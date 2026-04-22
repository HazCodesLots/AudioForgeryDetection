import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau, LinearLR, SequentialLR
import torch.utils.checkpoint as checkpoint
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm
import os
import json
from datetime import datetime
from typing import Tuple, Optional
import random
import gc
import pandas as pd
from pathlib import Path
import librosa



class KANLayer(nn.Module):
    """
    Optimized KAN layer for significantly faster training and lower memory usage.
    Uses 'Efficient-KAN' principles with precomputed grid denominators and 
    stable initialization to prevent gradient explosions.
    """
    def __init__(self, in_features, out_features, grid_size=5, spline_order=3, grid_range=(-1, 1), base_activation=nn.SiLU):
        super(KANLayer, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.base_activation = base_activation()
        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=math.sqrt(5))

        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.spline_weight, a=math.sqrt(5))

        grid = torch.linspace(grid_range[0], grid_range[1], grid_size + 1).unsqueeze(0)
        h = (grid_range[1] - grid_range[0]) / grid_size
        extended_grid = torch.cat([
            grid[:, 0:1] - h * torch.arange(spline_order, 0, -1).unsqueeze(0).float(),
            grid,
            grid[:, -1:] + h * torch.arange(1, spline_order + 1).unsqueeze(0).float()
        ], dim=1)
        self.register_buffer("grid", extended_grid)
        
        self.spline_coeffs = nn.Parameter(
            torch.randn(out_features, in_features, grid_size + spline_order) * 0.001
        )
        self.dropout = nn.Dropout(0.1)

    def b_splines(self, x: torch.Tensor):
        """
        Compute B-spline basis functions with proper shape handling.
        x: (batch, in_features)
        Returns: (batch, in_features, num_basis)
        """
        assert x.dim() == 2 and x.size(-1) == self.in_features, \
            f"Expected shape (*, {self.in_features}), got {x.shape}"
        
        batch_size = x.size(0)
        grid = self.grid.squeeze(0)  # (num_grid_points)
        x = x.unsqueeze(-1)  # (batch, in_features, 1)
        
        # Compute basis: (batch, in_features, num_intervals)
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        
        for k in range(1, self.spline_order + 1):
            left = grid[:-k-1]
            right = grid[k:-1]
            denom_l = (right - left).clamp(min=1e-8)
            
            right_k = grid[k+1:]
            left_k = grid[1:-k]
            denom_r = (right_k - left_k).clamp(min=1e-8)
            
            bases = (
                (x - left) / denom_l * bases[..., :-1]
            ) + (
                (grid[k+1:] - x) / denom_r * bases[..., 1:]
            )
        
        return bases.contiguous()

    def forward(self, x: torch.Tensor):
        """
        Forward pass for KAN layer.
        x: (batch, in_features)
        Returns: (batch, out_features)
        """
        if x.dim() != 2:
            raise ValueError(f"Expected 2D input, got {x.dim()}D: {x.shape}")
        
        x = torch.tanh(x)
        x = self.dropout(x)
        base_output = F.linear(self.base_activation(x), self.base_weight)
        
        bases = self.b_splines(x)
        
        scaled_coeffs = self.spline_coeffs.clamp(-5.0, 5.0) * self.spline_weight.unsqueeze(-1)
        
        spline_output = torch.einsum("bij,oij->bo", bases, scaled_coeffs)
        
        return base_output + spline_output



class PreEmphasis(nn.Module):
    def __init__(self, coef=0.97):
        super(PreEmphasis, self).__init__()
        self.coef = coef
        # Create filter: [-coef, 1.0] shape (1, 1, 2) for conv1d
        self.register_buffer('flipped_filter', 
            torch.tensor([-self.coef, 1.0]).view(1, 1, 2).float())

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Apply pre-emphasis filter.
        x: (B, 1, T) after frontend receives it
        """
        # Ensure 3D: (B, 1, T)
        if x.dim() == 2:
            x = x.unsqueeze(1)
        
        # Pad at the beginning for causal filtering
        x_padded = torch.nn.functional.pad(x, (1, 0), mode='replicate')
        
        # Apply conv1d: (B, 1, T+1) -> (B, 1, T)
        x_preemphasized = torch.nn.functional.conv1d(x_padded, self.flipped_filter)
        
        return x_preemphasized


class SincConv(nn.Module):
    """
    Sinc-based convolution layer for raw waveform processing (SincNet).
    Learns band-pass filters based on cutoff frequencies.
    """
    @staticmethod
    def to_mel(hz):
        return 2595 * np.log10(1 + hz / 700)

    @staticmethod
    def to_hz(mel):
        return 700 * (10**(mel / 2595) - 1)

    def __init__(self, out_channels, kernel_size, sample_rate=16000, min_low_hz=50, min_band_hz=50):
        super(SincConv, self).__init__()

        if kernel_size % 2 == 0:
            kernel_size += 1
        
        self.sample_rate = sample_rate
        self.kernel_size = kernel_size
        self.out_channels = out_channels
        self.min_low_hz = min_low_hz
        self.min_band_hz = min_band_hz

        num_filters = out_channels
        low_hz = 30
        high_hz = self.sample_rate / 2 - (min_low_hz + min_band_hz)

        mel = np.linspace(self.to_mel(low_hz), self.to_mel(high_hz), num_filters + 1)
        hz = self.to_hz(mel)

        self.low_hz_ = nn.Parameter(torch.from_numpy(hz[:-1]).float().view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.from_numpy(np.diff(hz)).float().view(-1, 1))

        self.register_buffer('window_', torch.hamming_window(self.kernel_size, periodic=False).view(1, -1))

        n = (self.kernel_size - 1) / 2.0
        self.register_buffer('n_', (torch.arange(-n, n + 1).view(1, -1) / self.sample_rate))

    def forward(self, waveforms):
        low = self.min_low_hz + torch.abs(self.low_hz_)
        high = torch.clamp(
            low + self.min_band_hz + torch.abs(self.band_hz_),
            self.min_low_hz,
            self.sample_rate / 2 - 1.0
        )

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        mid = self.kernel_size // 2
        n_safe = self.n_.abs().clamp(min=1.0 / self.sample_rate)

        filters_low = torch.sin(2 * math.pi * f_times_t_low) / (math.pi * n_safe)
        filters_high = torch.sin(2 * math.pi * f_times_t_high) / (math.pi * n_safe)

        filters_low[:, mid] = 2.0 * low.squeeze(-1) / self.sample_rate
        filters_high[:, mid] = 2.0 * high.squeeze(-1) / self.sample_rate

        filters = (filters_high - filters_low) * self.window_

        norms = filters.norm(p=2, dim=1, keepdim=True).clamp(min=1e-8)
        filters = filters / norms

        return F.conv1d(waveforms, filters.view(self.out_channels, 1, self.kernel_size))


class RawFrontend(nn.Module):
    def __init__(self, out_channels=128, kernel_size=251, sample_rate=16000):
        super(RawFrontend, self).__init__()
        self.pre_emphasis = PreEmphasis(0.97)
        self.sinc_conv = SincConv(out_channels, kernel_size, sample_rate)
        self.pool = nn.MaxPool1d(kernel_size=3, stride=3)
        self.batch_norm = nn.BatchNorm1d(out_channels)

    def forward(self, x):
        # x: (B, 1, samples)
        x = self.pre_emphasis(x)
        x = self.sinc_conv(x)
        x = torch.abs(x)
        x = self.pool(x)
        x = self.batch_norm(x)
        return x # (B, 128, T')


class AudioProcessor:

    def __init__(self, sample_rate: int = 16000, max_length_seconds: float = 4.0):
        self.sample_rate = sample_rate
        self.max_length_seconds = max_length_seconds
        self.max_length_samples = int(sample_rate * max_length_seconds)

    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        """Load audio file using librosa instead of torchaudio."""
        if librosa is None:
            raise RuntimeError(
                "librosa is required for audio loading. "
                "Install with: pip install librosa soundfile"
            )
        
        try:
            # Load audio using librosa
            waveform, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            
            # Convert numpy array to torch tensor
            waveform = torch.from_numpy(waveform).float()
            
            return waveform, self.sample_rate
        except Exception as e:
            raise RuntimeError(f"Failed to load audio from {audio_path}: {str(e)}")

    def pad_or_crop(self, audio: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        if length is None:
            length = self.max_length_samples

        current_length = audio.shape[0]

        if current_length > length:
            start_idx = (current_length - length) // 2
            audio = audio[start_idx:start_idx + length]
        elif current_length < length:
            pad_amount = length - current_length
            audio = torch.nn.functional.pad(audio, (0, pad_amount))
        return audio

    def process(self, audio: torch.Tensor) -> torch.Tensor:
        """Pad, crop, and normalize raw audio for AASIST3_Raw.

        NOTE: Do NOT apply pre-emphasis here. The RawFrontend inside the model
        already handles this as its first step. Applying it twice over-filters
        the signal and causes gradient explosions in SincConv.
        """
        # 1. Pad/crop to fixed length
        audio = self.pad_or_crop(audio)

        # 2. Instance normalization (zero mean, unit variance)
        mean = audio.mean()
        std = torch.clamp(audio.std(), min=1e-5)
        audio = (audio - mean) / std

        return audio.unsqueeze(0)  # (1, samples)

    def create_sliding_windows(self, audio: torch.Tensor, window_seconds: float = 4.0, overlap_seconds: float = 2.0) -> torch.Tensor:
        window_samples = int(self.sample_rate * window_seconds)
        stride_samples = int(self.sample_rate * (window_seconds - overlap_seconds))

        audio_length = audio.shape[0]

        if audio_length <= window_samples:
            return self.pad_or_crop(audio, window_samples).unsqueeze(0)

        num_windows = 1 + (audio_length - window_samples) // stride_samples

        if (audio_length - window_samples) % stride_samples != 0:
            num_windows += 1

        windows = []
        for i in range(num_windows):
            start = i * stride_samples
            end = start + window_samples

            if end > audio_length:
                window = audio[start:]
                window = torch.nn.functional.pad(window, (0, window_samples - window.shape[0]), mode='constant', value=0)
            else:
                window = audio[start:end]
            
            windows.append(window)
        
        return torch.stack(windows)


class RawASV5Dataset(torch.utils.data.Dataset):
    """
    Dataset class for ASVspoof5 Raw Waveforms.
    Loads .flac files on the fly and returns raw waveform tensors.
    """
    def __init__(self, audio_dir, protocol_file, max_len=64600, is_train=False):
        self.audio_dir = Path(audio_dir)
        self.max_len = max_len
        self.is_train = is_train
        self.processor = AudioProcessor(sample_rate=16000, max_length_seconds=max_len/16000)
        
        print(f"Loading protocol: {protocol_file}")
        df = pd.read_csv(protocol_file, sep=' ', header=None)
        print(f"Protocol shape: {df.shape}")
        print(f"First row: {df.iloc[0].tolist() if len(df) > 0 else 'Empty'}")
        
        self.file_ids = df[1].values
        
        # Detect label column (usually column 6 or 8)
        label_col = None
        for col in [8, 6, 7]:
            if col in df.columns:
                try:
                    test_val = df[col].iloc[0]
                    if test_val in ['spoof', 'bonafide', '-']:
                        label_col = col
                        print(f"Detected label column: {col} (value: {test_val})")
                        break
                except:
                    continue
        
        if label_col is None:
            raise ValueError(f"Could not find label column in protocol. Columns: {df.columns.tolist()}")
        
        # Map labels: spoof=1, bonafide=0, -=0 (unknown treated as bonafide)
        self.labels = df[label_col].apply(
            lambda x: 1 if x == 'spoof' else 0
        ).values

    def get_class_weights(self):
        """Automatically calculate class weights for balancing."""
        count_0 = np.sum(self.labels == 0) # Bonafide
        count_1 = np.sum(self.labels == 1) # Spoof
        total = count_0 + count_1
        
        w_bonafide = total / (2 * count_0) if count_0 > 0 else 1.0
        w_spoof = total / (2 * count_1) if count_1 > 0 else 1.0
        return [w_bonafide, w_spoof]

    def __len__(self):
        return len(self.file_ids)

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        label = self.labels[idx]
        file_path = self.audio_dir / f"{file_id}.flac"
        
        try:
            waveform, _ = self.processor.load_audio(str(file_path))
            if self.is_train:
                # Random time stretching
                if random.random() < 0.3:
                    rate = random.uniform(0.95, 1.05)
                    waveform_np = waveform.numpy()
                    waveform_np = librosa.effects.time_stretch(waveform_np, rate=rate)
                    # Ensure same length
                    if waveform_np.shape[-1] > self.max_len:
                        waveform_np = waveform_np[..., :self.max_len]
                    elif waveform_np.shape[-1] < self.max_len:
                        waveform_np = np.pad(waveform_np, (0, self.max_len - waveform_np.shape[-1]))
                    waveform = torch.from_numpy(waveform_np).float()

                # Random gain with larger range
                if random.random() < 0.5:
                    gain = random.uniform(0.7, 1.3)
                    waveform = waveform * gain
                    waveform = torch.clamp(waveform, -1.0, 1.0)  # Prevent clipping artifacts
                
                # Larger additive noise
                if random.random() < 0.4:
                    noise = torch.randn_like(waveform) * random.uniform(0.001, 0.005)
                    waveform = waveform + noise
                
                # Spec-style noise bursts (random silence regions)
                if random.random() < 0.2:
                    num_bursts = random.randint(1, 3)
                    for _ in range(num_bursts):
                        v_len = waveform.shape[-1]
                        if v_len > 4000:
                            burst_start = random.randint(0, v_len - 4000)
                            burst_len = random.randint(1000, 4000)
                            if burst_start + burst_len <= v_len:
                                waveform[burst_start:burst_start + burst_len] *= random.uniform(0.0, 0.3)

            waveform = self.processor.process(waveform)  # (1, samples)

            if waveform.shape[-1] > self.max_len:
                waveform = waveform[..., :self.max_len]
            elif waveform.shape[-1] < self.max_len:
                waveform = F.pad(waveform, (0, self.max_len - waveform.shape[-1]))
            return waveform, torch.tensor(label).long()

        except Exception as e:
            print(f"Warning: Failed to load {file_path}: {e}")
            return torch.zeros((1, self.max_len)), torch.tensor(label).long()



class PositionalEmbedding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000, temperature: int = 10000):
        super(PositionalEmbedding, self).__init__()

        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(temperature) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.dim() == 3:
            seq_len = x.size(1)
            return x + self.pe[:seq_len, :].unsqueeze(0)
        elif x.dim() == 2:
            seq_len = x.size(0)
            return x + self.pe[:seq_len, :]
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")



class ConvUnit(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, padding=1):
        super(ConvUnit, self).__init__()
        
        self.batch_norm = nn.BatchNorm1d(in_channels)
        self.selu = nn.SELU()
        self.conv = nn.Conv1d(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=stride,
            padding=padding
        )
    
    def forward(self, x):
        x = self.batch_norm(x)
        x = self.selu(x)
        x = self.conv(x)
        return x


class ResidualBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3, stride=1, pool_size=3):
        super(ResidualBlock, self).__init__()
        
        self.conv1 = ConvUnit(
            in_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        
        self.conv2 = ConvUnit(
            out_channels, 
            out_channels, 
            kernel_size=kernel_size,
            stride=1,
            padding=kernel_size // 2
        )
        
        self.skip_connection = None
        if in_channels != out_channels or stride != 1:
            self.skip_connection = nn.Conv1d(
                in_channels, 
                out_channels,
                kernel_size=1,
                stride=1
            )
        
        self.maxpool = nn.MaxPool1d(
            kernel_size=pool_size, 
            stride=stride,
            padding=pool_size // 2
        ) if stride > 1 else None
    
    def forward(self, x):
        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.skip_connection is not None:
            identity = self.skip_connection(identity)
        
        out = out + identity
        
        if self.maxpool is not None:
            out = self.maxpool(out)
        
        return out


class FirstBlock(nn.Module):
    
    def __init__(self, in_channels, out_channels, kernel_size=3):
        super(FirstBlock, self).__init__()
        
        self.conv1 = ConvUnit(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
        self.conv2 = ConvUnit(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=kernel_size // 2
        )
        
        self.skip_connection = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=1
        ) if in_channels != out_channels else None
        
        self.maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
    
    def forward(self, x):

        identity = x
        
        out = self.conv1(x)
        out = self.conv2(out)
        
        if self.skip_connection is not None:
            identity = self.skip_connection(identity)
        
        out = out + identity
        out = self.maxpool(out)
        
        return out


class AASIST3Encoder(nn.Module):
    
    def __init__(
        self,
        in_channels=128,
        channels=[128, 128, 256, 256, 256, 256],
        output_dim=256
    ):
        super(AASIST3Encoder, self).__init__()
        
        assert len(channels) == 6, "Encoder must have exactly 6 blocks"
        
        self.pre_maxpool = nn.MaxPool1d(kernel_size=3, stride=1, padding=1)
        self.pre_bn = nn.BatchNorm1d(in_channels)
        self.pre_selu = nn.SELU()
        
        self.blocks = nn.ModuleList()
        
        self.blocks.append(
            FirstBlock(in_channels, channels[0], kernel_size=3)
        )
        
        for i in range(1, 6):
            stride = 2 if i in [1, 3, 5] else 1  
            self.blocks.append(
                ResidualBlock(
                    channels[i-1],
                    channels[i],
                    kernel_size=3,
                    stride=stride,
                    pool_size=3
                )
            )
        
        self.output_dim = channels[-1]
    
    def forward(self, x):
        x = self.pre_maxpool(x)
        x = self.pre_bn(x)
        x = self.pre_selu(x)
        
        for block in self.blocks:
            x = block(x)
        
        return x


class AASIST3_Raw(nn.Module):
    """
    AASIST3 Model Architecture for Raw Waveform input.
    Operates end-to-end with a learnable SincNet frontend.
    """
    def __init__(
        self,
        frontend_out_channels=128,
        encoder_channels=[128, 128, 256, 256, 256, 256],
        num_temporal_nodes=25,
        num_spatial_nodes=25,
        temporal_dim=64,
        spatial_dim=64,
        stack_dim=128,
        num_branches=4,
        pool_ratio=0.5,
        temperature=1.0,
        num_classes=2
    ):
        super(AASIST3_Raw, self).__init__()
        
        self.frontend = RawFrontend(out_channels=frontend_out_channels)
        
        self.encoder = AASIST3Encoder(
            in_channels=frontend_out_channels,
            channels=encoder_channels
        )
        
        encoder_out_dim = encoder_channels[-1]
        
        self.graph_formation = GraphFormation(
            encoder_dim=encoder_out_dim,
            num_temporal_nodes=num_temporal_nodes,
            num_spatial_nodes=num_spatial_nodes,
            temporal_dim=temporal_dim,
            spatial_dim=spatial_dim,
            pool_ratio=pool_ratio,
            temperature=temperature
        )
        
        self.backbone = MultiBranchArchitecture(
            temporal_dim=temporal_dim,
            spatial_dim=spatial_dim,
            num_temporal_nodes=num_temporal_nodes,
            num_spatial_nodes=num_spatial_nodes,
            stack_dim=stack_dim,
            num_branches=num_branches,
            pool_ratio=pool_ratio,
            temperature=temperature
        )
        
        self.output_head = AASIST3OutputHead(
            hidden_dim=self.backbone.hidden_dim,
            num_classes=num_classes,
            use_intermediate=True,
            intermediate_dim=128
        )

    def forward(self, x):
        # x: (B, 1, samples)
        features = self.frontend(x) # (B, 128, T')
        features = self.encoder(features) # (B, 256, T'')
        h_t, h_s = self.graph_formation(features)
        hidden = self.backbone(h_t, h_s)
        logits = self.output_head(hidden)
        return logits


class KAN_GAL(nn.Module):

    def __init__(self, in_dim, out_dim, num_nodes, temperature=1.0, dropout=0.2):
        super(KAN_GAL, self).__init__()

        self.in_dim = in_dim
        self.out_dim = out_dim
        self.num_nodes = num_nodes
        self.temperature = temperature

        self.dropout = nn.Dropout(dropout)
        self.kan_attention = KANLayer(in_dim, in_dim)
        self.W_att = nn.Parameter(torch.empty(in_dim, num_nodes))
        nn.init.xavier_uniform_(self.W_att)
        self.kan_attn_proj = KANLayer(in_dim, out_dim)
        self.kan_direct_proj = KANLayer(in_dim, out_dim)
        self.norm_in = nn.LayerNorm(in_dim)
        self.norm_product = nn.LayerNorm(in_dim)
        self.scaling = math.sqrt(in_dim)
        self.batch_norm = nn.BatchNorm1d(out_dim)
    
    def forward(self, h):
        batch_size, num_nodes, in_dim = h.shape
        h = self.norm_in(h)
        h = self.dropout(h)
        h_normed = F.normalize(h, p=2, dim=-1)
        h_expanded_i = h_normed.unsqueeze(2)
        h_expanded_j = h_normed.unsqueeze(1)

        node_products = h_expanded_i * h_expanded_j
        node_products = self.norm_product(node_products)
        node_products_flat = node_products.reshape(-1, self.in_dim)
        kan_out = self.kan_attention(node_products_flat)
        kan_out = kan_out.view(batch_size, num_nodes, num_nodes, self.in_dim)
        kan_out = torch.tanh(kan_out)
        
        attention_scores = torch.einsum('bnmd,dk->bnmk', kan_out, self.W_att)
        attention_scores = attention_scores.mean(dim=-1)
        attention_map = F.softmax(attention_scores / (self.scaling * self.temperature), dim=-1)
        
        h_attended = torch.bmm(attention_map, h)
        h_attended_flat = h_attended.reshape(-1, self.in_dim)
        kan2_out = self.kan_attn_proj(h_attended_flat)
        kan2_out = kan2_out.view(batch_size, num_nodes, self.out_dim)

        h_flat = h.reshape(-1, self.in_dim)
        kan3_out = self.kan_direct_proj(h_flat)
        kan3_out = kan3_out.view(batch_size, num_nodes, self.out_dim)

        output = kan2_out + kan3_out
        output = output.transpose(1,2)
        output = self.batch_norm(output)
        output = output.transpose(1,2)
        return output

class KAN_GraphPool(nn.Module):

    def __init__(self, in_dim, ratio=0.5, dropout=0.2):
        super(KAN_GraphPool, self).__init__()
        self.in_dim = in_dim
        self.ratio = ratio
        self.dropout = nn.Dropout(dropout)
        self.kan_score = KANLayer(in_dim, 1)

    def forward(self, h, k=None):
        batch_size, num_nodes, in_dim = h.shape
        if k is None:
            k = max(1, int(num_nodes * self.ratio))
        h_drop = self.dropout(h)
        h_flat = h_drop.reshape(-1, in_dim)
        scores_flat = self.kan_score(h_flat)
        scores = scores_flat.view(batch_size, num_nodes)
        scores = torch.sigmoid(scores)
        h_gated = h * scores.unsqueeze(-1)
        top_k_scores, top_k_indices = torch.topk(scores, k, dim=1)
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, in_dim)
        h_pooled = torch.gather(h_gated, 1, top_k_indices_expanded)
        return h_pooled

class KAN_HS_GAL(nn.Module):
    def __init__(self, temporal_dim, spatial_dim, num_temporal_nodes, num_spatial_nodes, stack_dim, temperature=1.0, dropout=0.2):
        super(KAN_HS_GAL, self).__init__()

        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.num_temporal_nodes = num_temporal_nodes
        self.num_spatial_nodes = num_spatial_nodes
        self.stack_dim = stack_dim
        self.hetero_dim = temporal_dim + spatial_dim
        self.temperature = temperature

        self.dropout = nn.Dropout(dropout)
        
        self.kan_temporal_proj = KANLayer(temporal_dim, temporal_dim)
        self.kan_spatial_proj = KANLayer(spatial_dim, spatial_dim)
        
        self.kan_primary_attn = KANLayer(self.hetero_dim, self.hetero_dim)
        
        self.W11 = nn.Parameter(torch.randn(self.hetero_dim))
        self.W12 = nn.Parameter(torch.randn(self.hetero_dim))
        self.W22 = nn.Parameter(torch.randn(self.hetero_dim))
        
        self.kan_stack_attn = KANLayer(self.hetero_dim, self.hetero_dim)
        
        self.W_m = nn.Parameter(torch.randn(self.hetero_dim, 1))
        nn.init.xavier_uniform_(self.W_m)
        
        self.kan_stack_update1 = KANLayer(self.hetero_dim, stack_dim)
        self.kan_stack_update2 = KANLayer(stack_dim, stack_dim)
        
        self.kan_hetero_update1 = KANLayer(self.hetero_dim, self.hetero_dim)
        self.kan_hetero_update2 = KANLayer(self.hetero_dim, self.hetero_dim)
        
        self.norm_t = nn.LayerNorm(temporal_dim)
        self.norm_s = nn.LayerNorm(spatial_dim)
        self.norm_st = nn.LayerNorm(self.hetero_dim)
        self.norm_product = nn.LayerNorm(self.hetero_dim)
        self.scaling = math.sqrt(self.hetero_dim)

        self.batch_norm = nn.BatchNorm1d(self.hetero_dim)
    
    def forward(self, h_t, h_s, S):
        batch_size = h_t.size(0)
        num_t = h_t.size(1)
        num_s = h_s.size(1)
        
        h_t = self.norm_t(h_t)
        h_s = self.norm_s(h_s)

        h_t_proj = self.kan_temporal_proj(h_t.reshape(-1, self.temporal_dim))
        h_t_proj = h_t_proj.view(batch_size, num_t, self.temporal_dim)
        
        h_s_proj = self.kan_spatial_proj(h_s.reshape(-1, self.spatial_dim))
        h_s_proj = h_s_proj.view(batch_size, num_s, self.spatial_dim)
        
        h_t_proj_padded = F.pad(h_t_proj, (0, self.spatial_dim))
        h_s_proj_padded = F.pad(h_s_proj, (self.temporal_dim, 0))
        
        h_st = torch.cat([h_t_proj_padded, h_s_proj_padded], dim=1)  # (B, num_t + num_s, hetero_dim)
        total_nodes = num_t + num_s
        
        h_st = self.norm_st(h_st)
        h_st = self.dropout(h_st)
        h_st_normed = F.normalize(h_st, p=2, dim=-1)
        h_st_i = h_st_normed.unsqueeze(2)
        h_st_j = h_st_normed.unsqueeze(1)
        node_products = h_st_i * h_st_j  # (B, total_nodes, total_nodes, hetero_dim)
        
        node_products = self.norm_product(node_products)
        node_products_flat = node_products.reshape(-1, self.hetero_dim)
        primary_attn_flat = self.kan_primary_attn(node_products_flat)
        primary_attn = primary_attn_flat.view(batch_size, total_nodes, total_nodes, self.hetero_dim)
        A = torch.tanh(primary_attn)
        
        # Create weight map based on temporal/spatial node types
        idx = torch.arange(total_nodes, device=h_st.device)
        is_temporal = idx < num_t
        is_temporal_2d = is_temporal.unsqueeze(0).expand(total_nodes, -1)
        
        W_map = torch.zeros(total_nodes, total_nodes, self.hetero_dim, 
                           device=h_st.device, dtype=h_st.dtype)
        is_TT = is_temporal_2d & is_temporal_2d.t()
        is_SS = (~is_temporal_2d) & (~is_temporal_2d.t())
        is_TS = ~(is_TT | is_SS)
        
        W_map[is_TT] = self.W11
        W_map[is_SS] = self.W22
        W_map[is_TS] = self.W12
        
        B = (A * W_map.unsqueeze(0)).sum(dim=-1)
        B_hat = F.softmax(B / (self.scaling * self.temperature), dim=-1)
        
        S_expanded = S.unsqueeze(1).expand(-1, total_nodes, -1)
        S_padded = F.pad(S_expanded, (0, self.hetero_dim - self.stack_dim))
        
        h_st_stack = h_st * S_padded
        h_st_stack_flat = h_st_stack.reshape(-1, self.hetero_dim)
        stack_attn_flat = self.kan_stack_attn(h_st_stack_flat)
        stack_attn = stack_attn_flat.view(batch_size, total_nodes, self.hetero_dim)
        stack_attn = torch.tanh(stack_attn)
        
        stack_scores = torch.matmul(stack_attn, self.W_m).squeeze(-1)  # (B, total_nodes)
        A_m = F.softmax(stack_scores / (self.scaling * self.temperature), dim=-1)
        h_st_weighted = (h_st * A_m.unsqueeze(-1)).sum(dim=1)
        
        S_update1 = self.kan_stack_update1(h_st_weighted)
        S_update2 = self.kan_stack_update2(S)
        S_new = S_update1 + S_update2
        
        h_st_attended = torch.bmm(B_hat, h_st)
        
        h_st_update1_flat = h_st_attended.reshape(-1, self.hetero_dim)
        h_st_update1 = self.kan_hetero_update1(h_st_update1_flat)
        h_st_update1 = h_st_update1.view(batch_size, total_nodes, self.hetero_dim)
        
        h_st_update2_flat = h_st.reshape(-1, self.hetero_dim)
        h_st_update2 = self.kan_hetero_update2(h_st_update2_flat)
        h_st_update2 = h_st_update2.view(batch_size, total_nodes, self.hetero_dim)
        
        h_st_new = h_st_update1 + h_st_update2
        h_st_new = h_st_new.transpose(1, 2)
        h_st_new = self.batch_norm(h_st_new)
        h_st_new = h_st_new.transpose(1, 2)
        
        h_t_new = h_st_new[:, :num_t, :self.temporal_dim]
        h_s_new = h_st_new[:, num_t:, self.temporal_dim:]
        
        return h_t_new, h_s_new, S_new


class GraphPositionalEmbedding(nn.Module):
    def __init__(self, num_nodes, embedding_dim):

        super(GraphPositionalEmbedding, self).__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim

        self.pos_embedding = nn.Parameter(torch.randn(1, num_nodes, embedding_dim))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, x):
        batch_size, num_nodes, _ = x.shape
        pos_emb = self.pos_embedding[:, :num_nodes, :].expand(batch_size, -1, -1)
        return x + pos_emb


class GraphFormation(nn.Module):
    """
    Graph Formation Module for AASIST3.
    
    Converts encoder output into two separate graph representations:
    1. Temporal Graph: Models temporal dependencies (frame-level nodes)
    2. Spatial Graph: Models spectral dependencies (frequency-band nodes)
    
    Flow:
    encoder_output (B, C, T) 
        → temporal pooling (B, N_t, C) → projection → temporal graph (B, N_t, temporal_dim)
        → spatial pooling  (B, N_s, T) → projection → spatial graph  (B, N_s, spatial_dim)
    """

    def __init__(
        self, 
        encoder_dim, 
        num_temporal_nodes=50, 
        num_spatial_nodes=50, 
        temporal_dim=64, 
        spatial_dim=64, 
        pool_ratio=0.5, 
        temperature=1.0
    ):
        super(GraphFormation, self).__init__()
        
        self.encoder_dim = encoder_dim
        self.num_temporal_nodes = num_temporal_nodes
        self.num_spatial_nodes = num_spatial_nodes
        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.pool_ratio = pool_ratio
        self.temperature = temperature
        
        self.temporal_projection = nn.Linear(encoder_dim, temporal_dim)
        self.spatial_pool_size = 32
        self.spatial_projection = nn.Linear(self.spatial_pool_size, spatial_dim)
        self.pe_temporal = PositionalEmbedding(temporal_dim, num_temporal_nodes)
        self.pe_spatial = PositionalEmbedding(spatial_dim, num_spatial_nodes)
        
        self.kan_gal_temporal = KAN_GAL(
            in_dim=temporal_dim, 
            out_dim=temporal_dim, 
            num_nodes=num_temporal_nodes, 
            temperature=temperature
        )
        self.kan_gal_spatial = KAN_GAL(
            in_dim=spatial_dim, 
            out_dim=spatial_dim, 
            num_nodes=num_spatial_nodes, 
            temperature=temperature
        )
        
        self.kan_pool_temporal = KAN_GraphPool(in_dim=temporal_dim, ratio=pool_ratio)
        self.kan_pool_spatial = KAN_GraphPool(in_dim=spatial_dim, ratio=pool_ratio)
        
        self.pooled_temporal_nodes = max(1, int(num_temporal_nodes * pool_ratio))
        self.pooled_spatial_nodes = max(1, int(num_spatial_nodes * pool_ratio))

    def _temporal_max_pooling(self, x):
        """
        Pool encoder output along time axis to create temporal nodes.
        
        Input:  (B, C, T) - encoder output
        Output: (B, N_t, C) - temporal node features
        """
        x_abs = torch.abs(x)  # (B, C, T)
        
        pooled = F.adaptive_max_pool1d(x_abs, self.num_temporal_nodes)
        temporal_features = pooled.transpose(1, 2)
        return temporal_features

    def _spatial_max_pooling(self, x):
        """
        Pool encoder output along channel axis to create spatial nodes.
        
        Input:  (B, C, T) - encoder output
        Output: (B, N_s, T) - spatial node features (one per frequency band)
        """
        x_abs = torch.abs(x)  # (B, C, T)
        
        x_transposed = x_abs.transpose(1, 2)
        
        pooled = F.adaptive_max_pool1d(x_transposed, self.num_spatial_nodes)
        
        spatial_features = pooled.transpose(1, 2)
        
        return spatial_features

    def forward(self, encoder_output):
        """
        Create temporal and spatial graphs from encoder output.
        
        Args:
            encoder_output: (B, encoder_dim, T) - output from AASIST3Encoder
            
        Returns:
            h_t: (B, pooled_temporal_nodes, temporal_dim) - pooled temporal graph
            h_s: (B, pooled_spatial_nodes, spatial_dim) - pooled spatial graph
        """
        batch_size = encoder_output.size(0)
        temporal_features = self._temporal_max_pooling(encoder_output)
        temporal_features = self.temporal_projection(temporal_features)
        temporal_features = self.pe_temporal(temporal_features)
        temporal_graph = self.kan_gal_temporal(temporal_features)
        h_t = self.kan_pool_temporal(temporal_graph, k=self.pooled_temporal_nodes)
        spatial_features = self._spatial_max_pooling(encoder_output)
        spatial_features = F.adaptive_max_pool1d(spatial_features, self.spatial_pool_size)
        spatial_features = self.spatial_projection(spatial_features)
        spatial_features = self.pe_spatial(spatial_features)
        spatial_graph = self.kan_gal_spatial(spatial_features)
        h_s = self.kan_pool_spatial(spatial_graph, k=self.pooled_spatial_nodes)
        return h_t, h_s


class BranchModule(nn.Module):
    def __init__(self, temporal_dim, spatial_dim, num_temporal_nodes, num_spatial_nodes, stack_dim, pool_ratio=0.5, temperature=1.0, use_checkpoint=False):
        super(BranchModule, self).__init__()
        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.pool_ratio = pool_ratio
        self.use_checkpoint = use_checkpoint

        self.hs_gal_1 = KAN_HS_GAL(
            temporal_dim=temporal_dim, 
            spatial_dim=spatial_dim, 
            num_temporal_nodes=num_temporal_nodes, 
            num_spatial_nodes=num_spatial_nodes,
            stack_dim=stack_dim, 
            temperature=temperature
        )
        self.pooled_temporal = KAN_GraphPool(temporal_dim, ratio=pool_ratio)
        self.pooled_spatial = KAN_GraphPool(spatial_dim, ratio=pool_ratio)

        self.pooled_temporal_nodes = max(1, int(num_temporal_nodes * pool_ratio))
        self.pooled_spatial_nodes = max(1, int(num_spatial_nodes * pool_ratio))

        self.hs_gal_2 = KAN_HS_GAL(
            temporal_dim=temporal_dim, 
            spatial_dim=spatial_dim, 
            num_temporal_nodes=self.pooled_temporal_nodes, 
            num_spatial_nodes=self.pooled_spatial_nodes,
            stack_dim=stack_dim, 
            temperature=temperature
        )

    def forward(self, h_t, h_s, S):
        if self.use_checkpoint and h_t.requires_grad:
            return torch.utils.checkpoint.checkpoint(
                self._forward_impl, h_t, h_s, S, use_reentrant=False
            )
        else:
            return self._forward_impl(h_t, h_s, S)

    def _forward_impl(self, h_t, h_s, S):
        h_t_2, h_s_2, S_2 = self.hs_gal_1(h_t, h_s, S)

        h_t_pooled = self.pooled_temporal(h_t_2, k=self.pooled_temporal_nodes)
        h_s_pooled = self.pooled_spatial(h_s_2, k=self.pooled_spatial_nodes)

        h_t_3, h_s_3, S_3 = self.hs_gal_2(h_t_pooled, h_s_pooled, S_2)

        return h_t_3, h_s_3, S_3


class MultiBranchArchitecture(nn.Module):
    def __init__(self, temporal_dim, spatial_dim, num_temporal_nodes, num_spatial_nodes, stack_dim, num_branches=4, pool_ratio=0.5, temperature=1.0, dropout_p1=0.4, dropout_p2=0.5):
        super(MultiBranchArchitecture, self).__init__()

        self.num_branches = num_branches
        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.stack_dim = stack_dim

        self.stack_node_init = nn.Parameter(torch.randn(1, stack_dim))
        nn.init.normal_(self.stack_node_init, mean=0.0, std=0.02)

        self.branches = nn.ModuleList()
        current_temporal_nodes = num_temporal_nodes
        current_spatial_nodes = num_spatial_nodes

        for i in range(num_branches):
            use_cp = True 
            branch = BranchModule(
                temporal_dim=temporal_dim, 
                spatial_dim=spatial_dim, 
                num_temporal_nodes=current_temporal_nodes, 
                num_spatial_nodes=current_spatial_nodes, 
                stack_dim=stack_dim, 
                pool_ratio=pool_ratio, 
                temperature=temperature,
                use_checkpoint=use_cp
            )
            self.branches.append(branch)

            current_temporal_nodes = branch.pooled_temporal_nodes
            current_spatial_nodes = branch.pooled_spatial_nodes

        self.dropout1 = nn.Dropout(dropout_p1)
        self.hidden_dim = 2 * temporal_dim + 2 * spatial_dim + stack_dim

    def forward(self, h_t_init, h_s_init):
        batch_size = h_t_init.size(0)
        S_init = self.stack_node_init.expand(batch_size, -1)

        temporal_outputs = []
        spatial_outputs = []
        stack_outputs = []

        h_t_current = h_t_init
        h_s_current = h_s_init
        S_current = S_init

        for i, branch in enumerate(self.branches):
            h_t_out, h_s_out, S_out = branch(h_t_current, h_s_current, S_current)

            temporal_outputs.append(h_t_out)
            spatial_outputs.append(h_s_out)
            stack_outputs.append(S_out)

            h_t_current = h_t_out
            h_s_current = h_s_out
            S_current = S_out
        
        max_temporal_nodes = max(t.size(1) for t in temporal_outputs)
        max_spatial_nodes = max(s.size(1) for s in spatial_outputs)

        temporal_padded = []
        for h_t in temporal_outputs:
            if h_t.size(1) < max_temporal_nodes:
                pad_size = max_temporal_nodes - h_t.size(1)
                h_t_padded = F.pad(h_t, (0, 0, 0, pad_size))
            else:
                h_t_padded = h_t
            temporal_padded.append(h_t_padded)

        H_t = torch.stack(temporal_padded, dim=1)

        spatial_padded = []
        for h_s in spatial_outputs:
            if h_s.size(1) < max_spatial_nodes:
                pad_size = max_spatial_nodes - h_s.size(1)
                h_s_padded = F.pad(h_s, (0, 0, 0, pad_size))
            else:
                h_s_padded = h_s
            spatial_padded.append(h_s_padded)

        H_s = torch.stack(spatial_padded, dim=1)
        S_f = torch.stack(stack_outputs, dim=1)

        H_max_t = H_t.max(dim=1)[0].max(dim=1)[0]
        H_mean_t = H_t.mean(dim=1).mean(dim=1)

        H_max_s = H_s.max(dim=1)[0].max(dim=1)[0]
        H_mean_s = H_s.mean(dim=1).mean(dim=1)

        S_max_f = S_f.max(dim=1)[0]

        hidden_features = torch.cat([H_max_t, H_mean_t, H_max_s, H_mean_s, S_max_f], dim=1)
        hidden_features = self.dropout1(hidden_features)  # p=0.4 once
        return hidden_features


class AASIST3OutputHead(nn.Module):
    def __init__(self, hidden_dim, num_classes=2, use_intermediate=False, intermediate_dim=256):
        super(AASIST3OutputHead, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_classes = num_classes
        self.use_intermediate = use_intermediate

        if use_intermediate:
            self.intermediate_kan = KANLayer(hidden_dim, intermediate_dim)
            self.output_kan = KANLayer(intermediate_dim, num_classes)
        else:
            self.output_kan = KANLayer(hidden_dim, num_classes)

    def forward(self, hidden_features):
        if self.use_intermediate:
            x = self.intermediate_kan(hidden_features)
            logits = self.output_kan(x)
        else:
            logits = self.output_kan(hidden_features)

        return logits


class AASIST3OutputWithEmbedding(nn.Module):
    def __init__(self, hidden_dim, embedding_dim=256, num_classes=2):
        super(AASIST3OutputWithEmbedding, self).__init__()

        self.hidden_dim = hidden_dim
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes

        self.embedding_kan = KANLayer(hidden_dim, embedding_dim)
        self.classifier_kan = KANLayer(embedding_dim, num_classes)
        self.bn_embedding = nn.BatchNorm1d(embedding_dim)

    def forward(self, hidden_features, return_embedding=False):
        embedding = self.embedding_kan(hidden_features)
        embedding = self.bn_embedding(embedding)
        
        logits = self.classifier_kan(embedding)
        if return_embedding:
            return logits, embedding
        else:
            return logits


class AASIT3MultiTaskOutput(nn.Module):
    def __init__(self, hidden_dim, num_attack_types=19, use_quality_head=False):
        super(AASIT3MultiTaskOutput, self).__init__()

        self.hidden_dim = hidden_dim
        self.num_attack_types = num_attack_types
        self.use_quality_head = use_quality_head

        self.binary_head = KANLayer(hidden_dim, 2)
        self.attack_head = KANLayer(hidden_dim, num_attack_types)

        if use_quality_head:
            self.quality_head = KANLayer(hidden_dim, 1)

    def forward(self, hidden_features):
        outputs = {}
        outputs['binary'] = self.binary_head(hidden_features)
        outputs['attack_type'] = self.attack_head(hidden_features)
        if self.use_quality_head:
            outputs['quality'] = self.quality_head(hidden_features)
        return outputs


class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        
        if self.alpha is not None:
            # Handle alpha as a vector/tensor for class balancing
            # Ensure alpha is on the correct device
            alpha_t = self.alpha.to(inputs.device).gather(0, targets.data)
            focal_loss = alpha_t * (1 - pt) ** self.gamma * ce_loss
        else:
            focal_loss = (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):
    def __init__(
        self,
        focal_gamma=2.0,
        ce_weight=0.5,
        focal_weight=0.5,
        label_smoothing=0.05
    ):
        super(CombinedLoss, self).__init__()
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        self.label_smoothing = label_smoothing
        self.focal_loss = FocalLoss(alpha=None, gamma=focal_gamma)

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(
            inputs, targets,
            label_smoothing=self.label_smoothing
        )
        focal_loss = self.focal_loss(inputs, targets)
        return self.ce_weight * ce_loss + self.focal_weight * focal_loss


class MetricsCalculation:
    @staticmethod
    def compute_eer(labels, scores):
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr
        
        roc_interp = interp1d(fpr, tpr)
        eer_fpr = brentq(lambda x: 1. - x - roc_interp(x), 0., 1.)
        eer = 100 * (1 - roc_interp(eer_fpr))
        
        idx = np.nanargmin(np.abs(fnr - fpr))
        threshold = thresholds[idx]
        
        return eer, threshold
    
    @staticmethod
    def compute_min_dcf(labels, scores, p_target=0.05, c_miss=1, c_fa=1):
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr
        
        dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
        min_dcf = np.min(dcf)
        
        return min_dcf
    
    @staticmethod
    def compute_accuracy(labels, predictions):
        return 100 * (labels == predictions).sum() / len(labels)
    
    @staticmethod
    def compute_all_metrics(labels, scores, predictions=None):
        if predictions is None:
            predictions = (scores > 0.5).astype(int)
        
        eer, eer_threshold = MetricsCalculation.compute_eer(labels, scores)
        min_dcf = MetricsCalculation.compute_min_dcf(labels, scores)
        accuracy = MetricsCalculation.compute_accuracy(labels, predictions)
        
        # Class-specific accuracies
        spoof_mask = (labels == 1)
        bonafide_mask = (labels == 0)
        
        spoof_acc = 100 * (predictions[spoof_mask] == labels[spoof_mask]).sum() / (spoof_mask.sum() + 1e-8)
        bonafide_acc = 100 * (predictions[bonafide_mask] == labels[bonafide_mask]).sum() / (bonafide_mask.sum() + 1e-8)
        
        tp = ((predictions == 1) & (labels == 1)).sum()
        fp = ((predictions == 1) & (labels == 0)).sum()
        fn = ((predictions == 0) & (labels == 1)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'eer': float(eer),
            'eer_threshold': float(eer_threshold),
            'min_dcf': float(min_dcf),
            'accuracy': float(accuracy),
            'bonafide_acc': float(bonafide_acc),
            'spoof_acc': float(spoof_acc),
            'precision': float(precision),
            'recall': float(recall),
            'f1': float(f1)
        }


class TrainAASIST3:
    
    def __init__(self, model, optimizer, criterion, device, scheduler=None,
                 checkpoint_dir='.', experiment_name='aasist3',
                 accumulation_steps=4, use_amp=True, max_grad_norm=1.0):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp
        self.max_grad_norm = max_grad_norm
        self.freeze_frontend_epochs = 0 # Will be updated via property or externally
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        self.last_val_metrics = None
        self.gap_threshold = 20.0
        self.skipped_steps = 0
        self.results_dir     = os.path.join(checkpoint_dir, experiment_name)
        self.checkpoint_dir  = self.results_dir
        self.weights_dir     = os.path.join(self.results_dir, 'Checkpoints')
        self.metrics_dir     = os.path.join(self.results_dir, 'Metrics')
        os.makedirs(self.weights_dir, exist_ok=True)
        os.makedirs(self.metrics_dir,  exist_ok=True)
        
        self.experiment_name = experiment_name
        self.current_epoch = 0
        self.training_config = {}
        self.best_eer = float('inf')
        self.best_min_dcf = float('inf')
        self.history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_eer': [],
            'val_min_dcf': [],
            'val_accuracy': []
        }
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        total_correct = 0
        total_samples = 0
        total_bonafide_correct = 0
        total_bonafide = 0
        total_spoof_correct = 0
        total_spoof = 0
        num_batches = 0
        epoch_grad_norms = []
        
        red_flag_triggered = False
        red_flag_count = 0
        max_gap_observed = 0.0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch+1} [Train]')
        
        self.optimizer.zero_grad()
        last_grad_norm = 0.0
        
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 2:
                audio, labels = batch
                audio = audio.to(self.device)
                labels = labels.to(self.device)
            else:
                continue
            
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(audio)
                loss = self.criterion(logits, labels)
                loss = loss / self.accumulation_steps
            
            self.scaler.scale(loss).backward()
            
            if (batch_idx + 1) % self.accumulation_steps == 0 or (batch_idx + 1) == len(train_loader):
                self.scaler.unscale_(self.optimizer)
                
                if 'cuda' in str(self.device):
                    torch.cuda.empty_cache()
                gc.collect()
                
                last_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=self.max_grad_norm).item()

                epoch_grad_norms.append(last_grad_norm)
                
                if math.isnan(last_grad_norm) or math.isinf(last_grad_norm):
                    grad_type = 'NaN' if math.isnan(last_grad_norm) else 'Inf'
                    print(f"\n[WARNING] {grad_type} gradient at epoch {self.current_epoch+1}, batch {batch_idx+1}.")
                elif last_grad_norm == 0.0:
                    print(f"\n[WARNING] Zero gradient at epoch {self.current_epoch+1}, batch {batch_idx+1}.")

                old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                if self.scaler.get_scale() < old_scale:
                    self.skipped_steps += 1
                
                self.optimizer.zero_grad()
                
            if batch_idx % 100 == 0:
                torch.cuda.empty_cache()
                if batch_idx % 500 == 0:
                    gc.collect()
            
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                total_samples += labels.size(0)
                
                spoof_mask = (labels == 1)
                bonafide_mask = (labels == 0)
                total_spoof_correct += (preds[spoof_mask] == labels[spoof_mask]).sum().item()
                total_spoof += spoof_mask.sum().item()
                total_bonafide_correct += (preds[bonafide_mask] == labels[bonafide_mask]).sum().item()
                total_bonafide += bonafide_mask.sum().item()
                
                total_correct = total_bonafide_correct + total_spoof_correct
                current_acc = 100 * total_correct / total_samples
            
            total_loss += loss.item() * self.accumulation_steps
            num_batches += 1
            
            pbar_metrics = {
                'loss': f'{loss.item()*self.accumulation_steps:.4f}',
                'acc': f'{current_acc:.1f}%',
                'grad': f'{last_grad_norm:.3f}',
                'skip': self.skipped_steps
            }
            if self.last_val_metrics:
                last_val_acc = self.last_val_metrics.get('accuracy', 0)
                gap = current_acc - last_val_acc
                pbar_metrics['last_val'] = f"{last_val_acc:.1f}%"
                pbar_metrics['gap'] = f"{gap:+.1f}%"
                
                if gap > self.gap_threshold:
                    red_flag_count += 1
                    max_gap_observed = max(max_gap_observed, gap)
                    if not red_flag_triggered:
                        pbar.write(f"\n[RED FLAG] Overfitting detected at batch {batch_idx+1}/{len(train_loader)}! Gap: {gap:.1f}% (Train: {current_acc:.1f}%, Val: {last_val_acc:.1f}%)")
                        pbar.write(f"Note: Subsequent red flags this epoch will be logged silently to JSON.")
                        red_flag_triggered = True
            
            pbar.set_postfix(pbar_metrics)
        
        avg_loss = total_loss / num_batches
        avg_acc = 100 * total_correct / total_samples if total_samples > 0 else 0
        bonafide_acc = 100 * total_bonafide_correct / total_bonafide if total_bonafide > 0 else 0
        spoof_acc = 100 * total_spoof_correct / total_spoof if total_spoof > 0 else 0
        valid_norms = [n for n in epoch_grad_norms if math.isfinite(n)]
        grad_stats = {
            'avg': np.mean(valid_norms) if valid_norms else 0.0,
            'min': np.min(valid_norms) if valid_norms else 0.0,
            'max': np.max(valid_norms) if valid_norms else 0.0
        }
        
        overfit_stats = {
            'red_flag_count': red_flag_count,
            'max_gap_observed': max_gap_observed
        }
        
        return avg_loss, avg_acc, bonafide_acc, spoof_acc, grad_stats, overfit_stats
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_labels = []
        all_scores = []
        all_predictions = []
        
        num_samples = 0
        correct_preds = 0
        pbar = tqdm(val_loader, desc=f'Epoch {self.current_epoch+1} [Val]')
        
        with torch.no_grad():
            for batch in pbar:
                if len(batch) == 2:
                    audio, labels = batch
                    audio = audio.to(self.device)
                    labels = labels.to(self.device)
                else:
                    continue
                
                with torch.amp.autocast('cuda', enabled=self.use_amp):
                    logits = self.model(audio)
                    loss = self.criterion(logits, labels)
                
                total_loss += loss.item()
                num_batches += 1
                
                probs = F.softmax(logits, dim=1)
                spoofed_scores = probs[:, 1]  
                predictions = logits.argmax(dim=1)
                
                num_samples += labels.size(0)
                correct_preds += (predictions == labels).sum().item()
                
                all_labels.append(labels.cpu().numpy())
                all_scores.append(spoofed_scores.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
                
                pbar.set_postfix({
                    'loss': f'{total_loss/num_batches:.4f}',
                    'acc': f'{100*correct_preds/num_samples:.1f}%'
                })
        
        all_labels = np.concatenate(all_labels)
        all_scores = np.concatenate(all_scores)
        all_predictions = np.concatenate(all_predictions)
        
        metrics = MetricsCalculation.compute_all_metrics(
            all_labels, all_scores, all_predictions
        )
        
        avg_loss = total_loss / num_batches
        metrics['loss'] = avg_loss
        
        return metrics
    
    def fit(self, train_loader, val_loader, num_epochs, early_stopping_patience=10, start_epoch=0):

        print(f"Starting Training: {self.experiment_name}")
        if start_epoch > 0:
            print(f"  Resuming from epoch {start_epoch + 1} → {num_epochs}")
        patience_counter = 0
        
        for epoch in range(start_epoch, num_epochs):
            self.current_epoch = epoch

            # Resume safety: ensure frontend is unfrozen if we start past the warmup period
            if epoch >= self.freeze_frontend_epochs and self.freeze_frontend_epochs > 0:
                for param in self.model.frontend.parameters():
                    if not param.requires_grad:
                        print(f"\n[Frontend] Ensuring SincConv frontend is unfrozen (Epoch {epoch+1} >= Warmup {self.freeze_frontend_epochs})")
                        param.requires_grad = True
                        break # Only print once
                # Re-ensure all params are unfrozen silently
                for param in self.model.frontend.parameters():
                    param.requires_grad = True

            # Unfreeze the SincConv frontend after warmup period
            if epoch == self.freeze_frontend_epochs and self.freeze_frontend_epochs > 0:
                print(f"\n[Frontend] Unfreezing SincConv frontend at epoch {epoch + 1}. Full end-to-end training begins.")
                for param in self.model.frontend.parameters():
                    param.requires_grad = True
            
            train_loss, train_acc, train_bn_acc, train_sp_acc, grad_stats, overfit_stats = self.train_epoch(train_loader)
            
            val_metrics = self.validate(val_loader)
            self.last_val_metrics = val_metrics
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_eer'].append(val_metrics['eer'])
            self.history['val_min_dcf'].append(val_metrics['min_dcf'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            
            gap = train_acc - val_metrics['accuracy']
            print(f"\n" + "-"*40)
            print(f" EPOCH {epoch+1}/{num_epochs} METRICS SUMMARY")
            print(f" " + "-"*40)
            print(f" Metric      | Training | Validation")
            print(f" " + "-"*38)
            print(f" Loss        | {train_loss:8.4f} | {val_metrics['loss']:10.4f}")
            print(f" Total Acc   | {train_acc:7.2f}% | {val_metrics['accuracy']:9.2f}%")
            print(f" Bonafide Acc| {train_bn_acc:7.2f}% | {val_metrics['bonafide_acc']:9.2f}%")
            print(f" Spoof Acc   | {train_sp_acc:7.2f}% | {val_metrics['spoof_acc']:9.2f}%")
            print(f" Train-Val G | {gap:^8.2f}% | {'N/A':^10}")
            print(f" EER         | {'N/A':^8} | {val_metrics['eer']:9.2f}%")
            print(f" minDCF      | {'N/A':^8} | {val_metrics['min_dcf']:10.4f}")
            print(f" " + "-"*40)
            print(f" GRADIENT STATS: Avg: {grad_stats['avg']:.3f} | Min: {grad_stats['min']:.3f} | Max: {grad_stats['max']:.3f}")
            print(f" " + "-"*40)
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['eer'])
                else:
                    self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Learning Rate: {current_lr:.6f}")
            
            is_best = (val_metrics['eer'] < self.best_eer)
            self.save_checkpoint(epoch + 1, is_best=is_best)
            
            current_lr = self.optimizer.param_groups[0]['lr']
            train_info = {
                'loss': train_loss,
                'accuracy': train_acc,
                'bonafide_acc': train_bn_acc,
                'spoof_acc': train_sp_acc,
                'grad_stats': grad_stats,
                'overfit_stats': overfit_stats,
                'skipped_steps': self.skipped_steps
            }
            self.save_epoch_log(
                epoch + 1, val_metrics, train_info,
                is_best=is_best,
                patience_counter=patience_counter,
                early_stopping_patience=early_stopping_patience,
                learning_rate=current_lr,
                num_epochs=num_epochs
            )
            
            if is_best:
                self.best_eer = val_metrics['eer']
                self.best_min_dcf = val_metrics['min_dcf']
                print(f"  [BEST] New best model saved (EER: {self.best_eer:.2f}%)")
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"  [SAVED] Saved to Results/ - checkpoint + metrics log for epoch {epoch+1}")
            
            if patience_counter >= early_stopping_patience:
                print(f"\n[WARNING] Early stopping triggered after {epoch+1} epochs")
                break
        
        print(f"Training Complete!")
        print(f"Best EER: {self.best_eer:.2f}%")
        print(f"Best minDCF: {self.best_min_dcf:.4f}")
        
        self.save_history()
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint to Results/<experiment>/Checkpoints/."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict() if self.scheduler is not None else None,
            'best_eer': self.best_eer,
            'best_min_dcf': self.best_min_dcf,
            'history': self.history
        }
        
        filename = f"AASIST3_Epoch{epoch}.pth"
        filepath = os.path.join(self.weights_dir, filename)
        torch.save(checkpoint, filepath)
        
        if is_best:
            best_path = os.path.join(self.weights_dir, "AASIST3_best.pth")
            torch.save(checkpoint, best_path)

    def save_epoch_log(
        self, epoch, val_metrics, train_info,
        is_best=False, patience_counter=0,
        early_stopping_patience=10, learning_rate=None, num_epochs=None
    ):
        """Save exhaustive per-epoch metrics to Results/<experiment>/Metrics/.

        Everything printed to the terminal is captured here so that
        nothing is lost between runs.
        """
        ti = train_info if isinstance(train_info, dict) else {}
        train_acc  = float(ti.get('accuracy', 0))
        val_acc    = float(val_metrics.get('accuracy', 0))
        train_val_gap = float(train_acc - val_acc)

        log_data = {
            'experiment': self.experiment_name,
            'epoch': epoch,
            'total_epochs': num_epochs,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),

            'train': {
                'loss':         ti.get('loss', 0),
                'accuracy':     train_acc,
                'bonafide_acc': ti.get('bonafide_acc', 0),
                'spoof_acc':    ti.get('spoof_acc', 0),
                'skipped_amp_steps': ti.get('skipped_steps', self.skipped_steps),
            },

            'val': {
                'loss':         val_metrics.get('loss', 0),
                'accuracy':     val_acc,
                'bonafide_acc': val_metrics.get('bonafide_acc', 0),
                'spoof_acc':    val_metrics.get('spoof_acc', 0),
                'eer':          val_metrics.get('eer', 0),
                'eer_threshold':val_metrics.get('eer_threshold', 0),
                'min_dcf':      val_metrics.get('min_dcf', 0),
                'precision':    val_metrics.get('precision', 0),
                'recall':       val_metrics.get('recall', 0),
                'f1':           val_metrics.get('f1', 0),
            },

            'overfitting': {
                'train_val_gap':    train_val_gap,
                'red_flag_count':   ti.get('overfit_stats', {}).get('red_flag_count', 0),
                'max_gap_observed': ti.get('overfit_stats', {}).get('max_gap_observed', 0.0),
            },

            'gradient_stats': ti.get('grad_stats', {'avg': 0, 'min': 0, 'max': 0}),

            'optimisation': {
                'learning_rate': learning_rate,
            },

            'bests': {
                'is_best_epoch': bool(is_best),
                'best_eer_so_far':    float(self.best_eer if not is_best else val_metrics.get('eer', 0)),
                'best_min_dcf_so_far':float(self.best_min_dcf if not is_best else val_metrics.get('min_dcf', 0)),
            },

            'early_stopping': {
                'patience_counter':        patience_counter,
                'early_stopping_patience': early_stopping_patience,
                'patience_remaining':      early_stopping_patience - patience_counter,
            },

            'checkpoint_saved': os.path.join(
                self.weights_dir, f"AASIST3_Epoch{epoch}.pth"
            ),
        }

        filename = f"Epoch{epoch:03d}_metrics.json"
        filepath = os.path.join(self.metrics_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=4)
    
    def load_checkpoint(self, path):
        """Load model checkpoint from a full path or filename inside checkpoint_dir."""
        if os.path.isabs(path) or os.path.exists(path):
            filepath = path
        else:
            filepath = os.path.join(self.checkpoint_dir, path)
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        new_wd = self.training_config.get('weight_decay', 0.01)
        for group in self.optimizer.param_groups:
            group['weight_decay'] = new_wd
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_eer = checkpoint['best_eer']
        self.best_min_dcf = checkpoint['best_min_dcf']
        self.history = checkpoint['history']
        
        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
        print(f"  Best EER: {self.best_eer:.2f}%")
        print(f"  Best minDCF: {self.best_min_dcf:.4f}")
        return self.current_epoch
    
    def save_history(self):
        """Save run-level training history + best results to Results/<experiment>/Metrics/."""
        history_path = os.path.join(self.metrics_dir, 'training_metrics.json')
        full_report = {
            "metadata": self.training_config,
            "best_eer": float(self.best_eer),
            "best_min_dcf": float(self.best_min_dcf),
            "epochs_total": int(len(self.history['train_loss'])),
            "history": self.history,
            "last_updated": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        with open(history_path, 'w') as f:
            json.dump(full_report, f, indent=4)
        
        print(f"  Comprehensive report saved to: {history_path}")
        print(f"  Results folder: {self.results_dir}")


def count_parameters(model):
    """Count trainable parameters in model."""
    total, trainable = 0, 0
    for p in model.parameters():
        total += p.numel()
        if p.requires_grad:
            trainable += p.numel()
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }

def print_training_summary(args, model, criterion, device, optimizer):
    """Print training configuration and model summary."""

    print(f"{'TRAINING CONFIGURATION SUMMARY':^70}")
    
    params = count_parameters(model)
    
    summary_data = [
        ("Experiment Name", args.experiment_name),
        ("Device", device),
        ("Total Epochs", args.epochs),
        ("Batch Size", args.batch_size),
        ("Accumulation Steps", args.accumulation_steps),
        ("Effective Batch Size", args.batch_size * args.accumulation_steps),
        ("Learning Rate", args.lr),
        ("Use AMP", not args.disable_amp),
        ("Max Length (frames)", args.max_len),
        ("Weight Decay", optimizer.param_groups[0]['weight_decay']),
        ("Patience", args.patience),
        ("Loss Function", criterion.__class__.__name__),
        ("Optimizer", "AdamW"),
        ("-" * 20, "-" * 30),
        ("Total Parameters", f"{params['total']:,}"),
        ("Trainable Params", f"{params['trainable']:,}"),
        ("Frozen Params", f"{params['frozen']:,}"),
    ]
    
    for label, value in summary_data:
        print(f"{label:<25}: {value}")
    
    print(f"{'MODEL ARCHITECTURE DETAILS':^70}")
    
    if hasattr(model, 'encoder'):
        print(f"Encoder Blocks    : {len(getattr(model.encoder, 'blocks', []))}")
    
    if hasattr(model, 'backbone'):
        backbone = model.backbone
        if hasattr(backbone, 'num_branches'):
            print(f"Model Branches    : {backbone.num_branches}")
        if hasattr(backbone, 'temporal_dim'):
            print(f"Temporal Dim      : {backbone.temporal_dim}")
        if hasattr(backbone, 'spatial_dim'):
            print(f"Spatial Dim       : {backbone.spatial_dim}")
        if hasattr(backbone, 'stack_dim'):
            print(f"Stack Dim         : {backbone.stack_dim}")
            
    dropouts = {}
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            dropouts[name] = module.p
    
    if dropouts:
        print("\nDropout Rates:")
        for name, p in dropouts.items():
            print(f"  {name}: {p}")
            

    metadata = {
        "config": {label: str(value) for label, value in summary_data if not label.startswith("-")},
        "architecture": {
            "encoder_blocks": len(getattr(model.encoder, 'blocks', [])) if hasattr(model, 'encoder') else 6,
            "branches": getattr(model.backbone, 'num_branches', 4),
            "temporal_dim": getattr(model.backbone, 'temporal_dim', 64),
            "spatial_dim": getattr(model.backbone, 'spatial_dim', 64),
            "stack_dim": getattr(model.backbone, 'stack_dim', 128),
            "dropouts": dropouts
        },
        "parameters": params
    }
    return metadata

def print_model_summary(model, device, input_size=(1, 1, 64600)):
    print("MODEL FORWARD TEST")
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(input_size).to(device)
        try:
            output = model(dummy_input)
            print(f"Input shape:  {dummy_input.shape}")
            print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"Forward pass failed: {e}")
            import traceback
            traceback.print_exc()


def save_model_config(model, filepath):
    """Save model configuration to JSON."""
    config = {
        'model_type': model.__class__.__name__,
        'timestamp': datetime.now().isoformat(),
        'parameters': count_parameters(model)
    }
    
    with open(filepath, 'w') as f:
        json.dump(config, f, indent=2)



class AASIST3Inference:
    def __init__(self, model, device='cuda', sample_rate=16000):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
        self.processor = AudioProcessor(sample_rate=sample_rate)
    
    def predict_file(self, audio_path, return_score=True):

        audio, _ = self.processor.load_audio(audio_path)
        audio = self.processor.process(audio) # Returns (1, samples)
        
        audio = audio.unsqueeze(0).to(self.device) # (1, 1, samples)
        
        with torch.no_grad():
            logits = self.model(audio)
            probs = F.softmax(logits, dim=1)
            spoofed_prob = probs[0, 1].item()
        
        if return_score:
            return spoofed_prob
        else:
            return 1 if spoofed_prob > 0.5 else 0
    
    def predict_with_sliding_window(
        self,
        audio_path,
        window_size=64000,
        overlap=32000
    ):

        audio, _ = self.processor.load_audio(audio_path)
        
        scores = []
        stride = window_size - overlap
        
        for start in range(0, audio.shape[-1] - window_size + 1, stride):

            window = audio[start:start + window_size]
            window = self.processor.process(window)
            window = window.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(window)
                probs = F.softmax(logits, dim=1)
                scores.append(probs[0, 1].item())
        
        return np.mean(scores)


def run_raw_training():
    import argparse
    from torch.utils.data import DataLoader
    
    parser = argparse.ArgumentParser(description="AASIST3 Raw Waveform Training")
    # Updated to your new dataset location
    dataset_root = r"C:\Users\HazCodes\Documents\Datasets\ASVspoof5"
    
    parser.add_argument("--train_audio_dir", type=str, default=os.path.join(dataset_root, "flac_T"))
    parser.add_argument("--train_protocol", type=str, default=os.path.join(dataset_root, "ASVspoof5.train.tsv"))
    parser.add_argument("--dev_audio_dir", type=str, default=os.path.join(dataset_root, "flac_D"))
    parser.add_argument("--dev_protocol", type=str, default=os.path.join(dataset_root, "ASVspoof5.dev.track_1.tsv"))
    
    default_results = r"N:\ASV5"
    parser.add_argument("--checkpoint_dir", type=str, default=default_results,
                        help="Root results directory. Outputs go to <dir>/<experiment_name>/. "
                             f"Default: {default_results}")
    parser.add_argument("--experiment_name", type=str, default="aasist3_raw_v1")
    parser.add_argument("--epochs", type=int, default=30,
                        help="Total number of epochs to train.")
    parser.add_argument("--start_epoch", type=int, default=0,
                        help="Epoch to start/resume training from.")
    parser.add_argument("--resume", type=str, default=None,
                        help="Path to a .pth checkpoint to resume training from.")
    parser.add_argument("--batch_size", type=int, default=16)
    parser.add_argument("--lr", type=float, default=1e-4)
    parser.add_argument("--max_len", type=int, default=64600, help="~4s audio at 16kHz (ASVspoof standard: 64600 samples)")
    parser.add_argument("--num_workers", type=int, default=0)
    parser.add_argument("--patience", type=int, default=15)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--accumulation_steps", type=int, default=2)
    parser.add_argument("--disable_amp", action="store_true")
    parser.add_argument("--weight_decay", type=float, default=0.05)
    parser.add_argument("--max_grad_norm", type=float, default=1.0,
                        help="Max gradient norm for clipping. Default: 1.0")
    parser.add_argument("--freeze_frontend_epochs", type=int, default=2,
                        help="Freeze SincConv frontend for first N epochs to stabilize encoder training.")
    parser.add_argument("--use_ce_loss", action="store_true",
                        help="Use plain CrossEntropyLoss instead of CombinedLoss (FocalLoss).")
    parser.add_argument("--ce_pos_weight", type=float, default=3.0,
                        help="Pos weight for bonafide class in CrossEntropyLoss.")
    args = parser.parse_args()

    resume_epoch = 0
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading datasets...")
    train_dataset = RawASV5Dataset(args.train_audio_dir, args.train_protocol, max_len=args.max_len, is_train=True)
    dev_dataset = RawASV5Dataset(args.dev_audio_dir, args.dev_protocol, max_len=args.max_len, is_train=False)
    
    bonafide_w, spoof_w = train_dataset.get_class_weights()
    print(f"[Info] Class imbalance — Bonafide weight: {bonafide_w:.2f}, Spoof weight: {spoof_w:.2f}")

    if args.subset:
        train_indices = torch.randperm(len(train_dataset))[:args.subset]
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        
        dev_indices = torch.randperm(len(dev_dataset))[:args.subset]
        dev_dataset = torch.utils.data.Subset(dev_dataset, dev_indices)
        print(f"Using subset of {args.subset} training and validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print("Initializing AASIST3_Raw model...")
    model = AASIST3_Raw(frontend_out_channels=128).to(device)

    # Freeze the SincConv frontend for the first N epochs
    if args.freeze_frontend_epochs > 0:
        print(f"Freezing SincConv frontend for first {args.freeze_frontend_epochs} epochs...")
        for param in model.frontend.parameters():
            param.requires_grad = False
    
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)

    if args.use_ce_loss:
        ce_weight = torch.tensor([args.ce_pos_weight, 1.0]).to(device)
        criterion = nn.CrossEntropyLoss(weight=ce_weight, label_smoothing=0.05)
        print(f"Using CrossEntropyLoss with weights [bonafide={args.ce_pos_weight:.1f}, spoof=1.0]")
    else:
        # Fallback to combined loss
        criterion = CombinedLoss(class_weights=[bonafide_w, spoof_w], label_smoothing=0.05).to(device)
        print("Using CombinedLoss (FocalLoss + CE)")
    
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    metadata = print_training_summary(args, model, criterion, device, optimizer)
    print_model_summary(model, device, input_size=(1, 1, args.max_len))
    torch.cuda.empty_cache()

    trainer = TrainAASIST3(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment_name,
        accumulation_steps=args.accumulation_steps,
        use_amp=not args.disable_amp,
        max_grad_norm=args.max_grad_norm
    )
    trainer.training_config = metadata
    trainer.freeze_frontend_epochs = args.freeze_frontend_epochs
    if args.resume:
        resume_epoch = trainer.load_checkpoint(args.resume)
        print(f"  Will resume from epoch {resume_epoch + 1} → {args.epochs}")
    
    start_epoch = args.start_epoch if args.start_epoch > 0 else resume_epoch

    try:
        trainer.fit(
            train_loader,
            dev_loader,
            num_epochs=args.epochs,
            early_stopping_patience=args.patience,
            start_epoch=start_epoch
        )
    except Exception as e:
        import traceback
        error_msg = traceback.format_exc()
        print("FATAL ERROR CAUGHT:")
        print(error_msg)
        
        log_path = os.path.join(os.getcwd(), "debug_error.txt")
        with open(log_path, "w") as f:
            f.write(error_msg)
        raise e

if __name__ == "__main__":
    run_raw_training()