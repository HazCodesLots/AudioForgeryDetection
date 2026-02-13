import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math
import os
import json
import random
import pandas as pd
from datetime import datetime
from typing import Tuple, Optional
from pathlib import Path
from tqdm import tqdm
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

try:
    import torchaudio
except (ImportError, OSError):
    torchaudio = None

try:
    from transformers import Wav2Vec2Model
except ImportError:
    Wav2Vec2Model = None



class KANLayer(nn.Module):
    def __init__(self, in_features, out_features, grid_size=16, spline_order=4, grid_range=(-1, 1)):
        super(KANLayer, self).__init__()
        
        self.in_features = in_features
        self.out_features = out_features
        self.grid_size = grid_size
        self.spline_order = spline_order
        self.grid_range = grid_range

        self.num_grid_points = 2 * spline_order + grid_size + 1

        self.prelu_weight = nn.Parameter(torch.randn(out_features, in_features) * 0.25)

        self.base_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.base_weight, a=np.sqrt(5))

        self.spline_weight = nn.Parameter(torch.empty(out_features, in_features))
        nn.init.kaiming_uniform_(self.spline_weight, a=np.sqrt(5))

        num_coeffs = grid_size + spline_order
        self.spline_coeffs = nn.Parameter(torch.randn(out_features, in_features, num_coeffs) * 0.1)

        self.register_buffer('grid', self._create_grid())

    def _create_grid(self):
        alpha_1, alpha_2 = self.grid_range
        h = (alpha_2 - alpha_1) / (self.grid_size)

        theta_1 = -self.spline_order * h + alpha_1
        theta_2 = (self.grid_size + self.spline_order + 1) * h + alpha_1

        grid = torch.linspace(theta_1, theta_2, self.num_grid_points)
        return grid

    def _compute_bspline_basis(self, x: torch.Tensor):
        """
        Vectorized computation of B-spline bases using De Boor's recursion.
        Significantly faster than nested loops.
        """

        x = x.unsqueeze(-1)
        grid = self.grid
        
        bases = ((x >= grid[:-1]) & (x < grid[1:])).float()
        
        for k in range(1, self.spline_order + 1):
            left_den = grid[k:-1] - grid[:-k-1]
            left_term = ((x - grid[:-k-1]) / left_den) * bases[..., :-1]
            
            right_den = grid[k+1:] - grid[1:-k]
            right_term = ((grid[k+1:] - x) / right_den) * bases[..., 1:]
            
            bases = left_term + right_term
            
        return bases.contiguous()

    def _prelu(self, x: torch.Tensor):
        
        pos = F.relu(x)
        neg = x - pos
        
        return torch.einsum('ij,...j->i...j', self.prelu_weight, neg) + pos.unsqueeze(0)

    def forward(self, x: torch.Tensor):
        batch_shape = x.shape[:-1]
        x_clamped = torch.clamp(x, self.grid_range[0], self.grid_range[1])

        base_output = self._prelu(x_clamped)

        bspline_basis = self._compute_bspline_basis(x_clamped)
        spline_output = torch.einsum('ijk,...jk->i...j', self.spline_coeffs, bspline_basis)
        
        output = torch.einsum('ij,i...j->i...', self.base_weight, base_output) + \
                 torch.einsum('ij,i...j->i...', self.spline_weight, spline_output)

        dims = list(range(1, output.dim())) + [0]
        return output.permute(*dims).contiguous()


class PreEmphasis(nn.Module):
    def __init__(self, coef=0.97):
        super(PreEmphasis, self).__init__()
        self.coef = coef
        self.register_buffer('flipped_filter', torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() == 2:
            x = x.unsqueeze(1)
            squeeze_output = True
        else:
            squeeze_output = False

        x_padded = torch.nn.functional.pad(x, (1, 0), mode='replicate')
        x_preemphasized = torch.nn.functional.conv1d(x_padded, self.flipped_filter)

        if squeeze_output:
            x_preemphasized = x_preemphasized.squeeze(1)

        return x_preemphasized


class AudioProcessor:

    def __init__(self, sample_rate: int = 16000, max_length_seconds: float = 4.0):
        self.sample_rate = sample_rate
        self.max_length_seconds = max_length_seconds
        self.max_length_samples = int(sample_rate * max_length_seconds)

    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] != 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        return waveform.squeeze(0), self.sample_rate

    def process(self, audio: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        """Alias for pad_or_crop to maintain compatibility with inference classes."""
        return self.pad_or_crop(audio, length)

    def pad_or_crop(self, audio: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        if length is None:
            length = self.max_length_samples

        current_length = audio.shape[0]

        if current_length > length:
            start_idx = (current_length - length) // 2
            audio = audio[start_idx:start_idx + length]
        elif current_length < length:
            pad_amount = length - current_length
            audio = torch.nn.functional.pad(audio, (0, pad_amount), mode='constant', value=0)

        return audio

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


class ASV5MelDataset(torch.utils.data.Dataset):
    """
    Dataset class for ASVspoof5 Mel Spectrograms.
    Expects mel spectrograms in .npy format with shape (128, time).
    """
    def __init__(self, mel_dir, protocol_file, max_len=200, normalize=True):
        self.mel_dir = Path(mel_dir)
        self.max_len = max_len
        self.normalize = normalize
        
        print(f"Loading protocol: {protocol_file}")
        df = pd.read_csv(protocol_file, sep=' ', header=None)
        
        self.file_ids = df[1].values
        self.labels = df[8].apply(lambda x: 1 if x == 'spoof' else 0).values

    def __len__(self):
        return len(self.file_ids)

    def _pad_or_truncate(self, mel):
        time_len = mel.shape[1]
        if time_len < self.max_len:
            pad_len = self.max_len - time_len
            mel = np.pad(mel, ((0, 0), (0, pad_len)), mode='constant')
        elif time_len > self.max_len:
            start = (time_len - self.max_len) // 2
            mel = mel[:, start:start + self.max_len]
        return mel

    def __getitem__(self, idx):
        file_id = self.file_ids[idx]
        label = self.labels[idx]
        file_path = self.mel_dir / f"{file_id}.npy"
        try:
            mel = np.load(file_path)
            mel = self._pad_or_truncate(mel)
            if self.normalize:
                mean = mel.mean()
                std = mel.std() + 1e-9
                mel = (mel - mean) / std
            return torch.from_numpy(mel).float(), torch.tensor(label).long()
        except:
            return torch.zeros((128, self.max_len)), torch.tensor(label).long()


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000, temprature: int = 10000):
        super(PositionalEmbedding, self).__init__()

        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(temprature) / d_model))

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


class Wav2Vec2Frontend(nn.Module):

    def __init__(self, model_name='facebook/wav2vec2-xls-r-300m', output_dim=768, freeze_encoder=False, use_conv_projection=True):
        super(Wav2Vec2Frontend, self).__init__()

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.hidden_dim = self.wav2vec2.config.hidden_size

        if freeze_encoder:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

        self.use_conv_projection = use_conv_projection
        if use_conv_projection:
            self.projection = nn.Conv1d(self.hidden_dim, output_dim, kernel_size=1, bias=True)
        else:
            self.projection = nn.Linear(self.hidden_dim, output_dim)
        self.output_dim = output_dim

    def forward(self, x):
        if x.dim() == 3:
            x = x.squeeze(1)

        outputs = self.wav2vec2(x, output_hidden_states=False)
        features = outputs.last_hidden_state

        if self.use_conv_projection:
            features = features.transpose(1, 2)
            features = self.projection(features)
        else:
            features = self.projection(features)
            features = features.transpose(1,2)
        return features



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
        in_channels=70,
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
            stride = 3 if i in [1, 3, 5] else 1  
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


class AASIST3_Mel(nn.Module):
    """
    AASIST3 Model adapted for Mel Spectrogram input.
    """
    def __init__(
        self,
        in_channels=128,
        encoder_channels=[128, 128, 256, 256, 256, 256],
        num_temporal_nodes=100,
        num_spatial_nodes=100,
        temporal_dim=64,
        spatial_dim=64,
        stack_dim=128,
        num_branches=4,
        pool_ratio=0.5,
        temperature=1.0,
        num_classes=2
    ):
        super(AASIST3_Mel, self).__init__()
        
        self.encoder = AASIST3Encoder(
            in_channels=in_channels,
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
            num_classes=num_classes
        )

    def forward(self, x):
        features = self.encoder(x)
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
        self.temprature = temperature

        self.dropout = nn.Dropout(dropout)
        self.kan_attention = KANLayer(in_dim, in_dim)
        self.W_att = nn.Parameter(torch.empty(in_dim, num_nodes))
        nn.init.xavier_uniform_(self.W_att)
        self.kan_attn_proj = KANLayer(in_dim, out_dim)
        self.kan_direct_proj = KANLayer(in_dim, out_dim)
        self.batch_norm = nn.BatchNorm1d(out_dim)
    
    def forward(self, h):
        batch_size, num_nodes, in_dim = h.shape
        h = self.dropout(h)
        h_expanded_i = h.unsqueeze(2)
        h_expanded_j = h.unsqueeze(1)

        node_products = h_expanded_i * h_expanded_j
        node_products_flat = node_products.reshape(-1, self.in_dim)
        kan_out = self.kan_attention(node_products_flat)
        kan_out = kan_out.view(batch_size, num_nodes, num_nodes, self.in_dim)
        kan_out = torch.tanh(kan_out)
        
        attention_scores = torch.einsum('bnmd,dk->bnmk', kan_out, self.W_att)
        attention_scores = attention_scores.mean(dim=-1)
        attention_map = F.softmax(attention_scores / self.temprature, dim=-1)
        
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
    def __init__(self, temporal_dim, spatial_dim, num_nodes, stack_dim, temperature=1.0, dropout=0.2):
        super(KAN_HS_GAL, self).__init__()

        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.num_nodes = num_nodes
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
        
        self.batch_norm = nn.BatchNorm1d(self.hetero_dim)
    
    def forward(self, h_t, h_s, S):
        batch_size, num_nodes, _ = h_t.shape
        
        h_t_proj = self.kan_temporal_proj(h_t.reshape(-1, self.temporal_dim))
        h_t_proj = h_t_proj.view(batch_size, num_nodes, self.temporal_dim)
        
        h_s_proj = self.kan_spatial_proj(h_s.reshape(-1, self.spatial_dim))
        h_s_proj = h_s_proj.view(batch_size, num_nodes, self.spatial_dim)
        
        h_st = torch.cat([h_t_proj, h_s_proj], dim=-1)
        h_st = self.dropout(h_st)
        
        h_st_i = h_st.unsqueeze(2)
        h_st_j = h_st.unsqueeze(1)
        node_products = h_st_i * h_st_j
        
        node_products_flat = node_products.reshape(-1, self.hetero_dim)
        primary_attn_flat = self.kan_primary_attn(node_products_flat)
        primary_attn = primary_attn_flat.view(batch_size, num_nodes, num_nodes, self.hetero_dim)
        A = torch.tanh(primary_attn)
        
        idx = torch.arange(num_nodes, device=h_st.device)
        is_temporal = idx < (num_nodes // 2)
        is_temporal = is_temporal.unsqueeze(0).expand(num_nodes, -1)
        
        W_map = torch.zeros(num_nodes, num_nodes, self.hetero_dim, device=h_st.device)
        is_TT = is_temporal & is_temporal.t()
        is_SS = (~is_temporal) & (~is_temporal.t())
        is_TS = ~(is_TT | is_SS)
        
        W_map[is_TT] = self.W11
        W_map[is_SS] = self.W22
        W_map[is_TS] = self.W12
        
        B = (A * W_map.unsqueeze(0)).sum(dim=-1)
        B_hat = F.softmax(B / self.temperature, dim=-1)
        
        S_expanded = S.unsqueeze(1).expand(-1, num_nodes, -1)
        S_padded = F.pad(S_expanded, (0, self.hetero_dim - self.stack_dim))
        
        h_st_stack = h_st * S_padded
        h_st_stack_flat = h_st_stack.reshape(-1, self.hetero_dim)
        stack_attn_flat = self.kan_stack_attn(h_st_stack_flat)
        stack_attn = stack_attn_flat.view(batch_size, num_nodes, self.hetero_dim)
        stack_attn = torch.tanh(stack_attn)
        
        stack_scores = torch.matmul(stack_attn, self.W_m).squeeze(-1)
        A_m = F.softmax(stack_scores / self.temperature, dim=-1)
        h_st_weighted = (h_st * A_m.unsqueeze(-1)).sum(dim=1)
        
        S_update1 = self.kan_stack_update1(h_st_weighted)
        S_update2 = self.kan_stack_update2(S)
        S_new = S_update1 + S_update2
        
        h_st_attended = torch.bmm(B_hat, h_st)
        
        h_st_update1_flat = h_st_attended.reshape(-1, self.hetero_dim)
        h_st_update1 = self.kan_hetero_update1(h_st_update1_flat)
        h_st_update1 = h_st_update1.view(batch_size, num_nodes, self.hetero_dim)
        
        h_st_update2_flat = h_st.reshape(-1, self.hetero_dim)
        h_st_update2 = self.kan_hetero_update2(h_st_update2_flat)
        h_st_update2 = h_st_update2.view(batch_size, num_nodes, self.hetero_dim)
        
        h_st_new = h_st_update1 + h_st_update2
        h_st_new = h_st_new.transpose(1, 2)
        h_st_new = self.batch_norm(h_st_new)
        h_st_new = h_st_new.transpose(1, 2)
        
        h_t_new = h_st_new[:, :, :self.temporal_dim]
        h_s_new = h_st_new[:, :, self.temporal_dim:]
        
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

    def __init__(self, encoder_dim, num_temporal_nodes=100, num_spatial_nodes=100, temporal_dim=64, spatial_dim=64, pool_ratio=0.5, temperature=1.0):

        super(GraphFormation, self).__init__()
        self.encoder_dim = encoder_dim
        self.num_temporal_nodes = num_temporal_nodes
        self.num_spatial_nodes = num_spatial_nodes
        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.temporal_projection = nn.Linear(encoder_dim, temporal_dim)
        self.spatial_projection = nn.Linear(encoder_dim, spatial_dim)
        self.pe_temporal = PositionalEmbedding(temporal_dim, num_temporal_nodes)
        self.pe_spatial = PositionalEmbedding(spatial_dim, num_spatial_nodes)

        self.kan_gal_temporal = KAN_GAL(in_dim=temporal_dim, out_dim=temporal_dim, num_nodes=num_temporal_nodes, temperature=temperature)
        self.kan_gal_spatial = KAN_GAL(in_dim=spatial_dim, out_dim=spatial_dim, num_nodes=num_spatial_nodes, temperature=temperature)
        self.kan_pool_temporal = KAN_GraphPool(in_dim=temporal_dim, ratio=pool_ratio)
        self.kan_pool_spatial = KAN_GraphPool(in_dim=spatial_dim, ratio=pool_ratio)

        self.pooled_temporal_nodes = max(1, int(num_temporal_nodes * pool_ratio))
        self.pooled_spatial_nodes = max(1, int(num_spatial_nodes * pool_ratio))

    def _temporal_max_pooling(self, x):
        batch_size, channels, time = x.shape
        x_abs = torch.abs(x)
        pooled = F.adaptive_max_pool1d(x_abs, self.num_temporal_nodes)
        temporal_features = pooled.transpose(1,2)
        return temporal_features

    def _spatial_max_pooling(self, x):
        batch_size, channels, time = x.shape
        x_abs = torch.abs(x)
        x_transposed = x_abs.transpose(1,2)
        pooled = F.adaptive_max_pool1d(x_transposed.transpose(1,2), self.num_spatial_nodes)
        spatial_features = pooled.transpose(1,2)
        return spatial_features

    
    def forward(self, encoder_output):

        batch_size = encoder_output.size(0)

        temporal_features = self._temporal_max_pooling(encoder_output)
        temporal_features = self.temporal_projection(temporal_features)
        temporal_features = self.pe_temporal(temporal_features)
        temporal_graph = self.kan_gal_temporal(temporal_features)
        h_t = self.kan_pool_temporal(temporal_graph, k=self.pooled_temporal_nodes)
        
        spatial_features = self._spatial_max_pooling(encoder_output)
        spatial_features = self.spatial_projection(spatial_features)
        spatial_features = self.pe_spatial(spatial_features)
        spatial_graph = self.kan_gal_spatial(spatial_features)
        h_s = self.kan_pool_spatial(spatial_graph, k=self.pooled_spatial_nodes)

        return h_t, h_s


class BranchModule(nn.Module):
    def __init__(self, temporal_dim, spatial_dim, num_temporal_nodes, num_spatial_nodes, stack_dim, pool_ratio=0.5, temperature=1.0):

        super(BranchModule, self).__init__()

        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.pool_ratio = pool_ratio

        self.hs_gal_1 = KAN_HS_GAL(temporal_dim=temporal_dim, spatial_dim= spatial_dim, num_nodes=max(num_temporal_nodes, num_spatial_nodes), stack_dim=stack_dim, temperature=temperature)
        self.pooled_temporal = KAN_GraphPool(temporal_dim, ratio=pool_ratio)
        self.pooled_spatial = KAN_GraphPool(spatial_dim, ratio=pool_ratio)

        self.pooled_temporal_nodes = max(1, int(num_temporal_nodes * pool_ratio))
        self.pooled_spatial_nodes = max(1, int(num_spatial_nodes * pool_ratio))

        self.hs_gal_2 = KAN_HS_GAL(temporal_dim=temporal_dim, spatial_dim=spatial_dim, num_nodes=max(self.pooled_temporal_nodes, self.pooled_spatial_nodes), stack_dim=stack_dim, temperature=temperature)

    def forward(self, h_t, h_s, S):
        h_t_2, h_s_2, S_2 = self.hs_gal_1(h_t, h_s, S)

        h_t_pooled = self.pooled_temporal(h_t_2, k=self.pooled_temporal_nodes)
        h_s_pooled = self.pooled_spatial(h_s_2, k=self.pooled_spatial_nodes)

        h_t_3, h_s_3, S_3 = self.hs_gal_2(h_t_pooled, h_s_pooled, S_2)

        return h_t_3, h_s_3, S_3


class MultiBranchArchitecture(nn.Module):
    def __init__(self, temporal_dim, spatial_dim, num_temporal_nodes, num_spatial_nodes, stack_dim, num_branches=4, pool_ratio=0.5, temperature=1.0, dropout_p1=0.2, dropout_p2=0.5):
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
            branch = BranchModule(temporal_dim=temporal_dim, spatial_dim=spatial_dim, num_temporal_nodes=current_temporal_nodes, num_spatial_nodes=current_spatial_nodes, stack_dim=stack_dim, pool_ratio=pool_ratio, temperature=temperature)
            self.branches.append(branch)

            current_temporal_nodes = branch.pooled_temporal_nodes
            current_spatial_nodes = branch.pooled_spatial_nodes

        self.dropout1 = nn.Dropout(dropout_p1)
        self.dropout2 = nn.Dropout(dropout_p2)
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

        H_t = self.dropout1(H_t)
        H_s = self.dropout1(H_s)
        S_f = self.dropout1(S_f)

        H_max_t = H_t.max(dim=1)[0].max(dim=1)[0]
        H_mean_t = H_t.mean(dim=1).mean(dim=1)

        H_max_s = H_s.max(dim=1)[0].max(dim=1)[0]
        H_mean_s = H_s.mean(dim=1).mean(dim=1)

        S_max_f = S_f.max(dim=1)[0]

        hidden_features = torch.cat([H_max_t, H_mean_t, H_max_s, H_mean_s, S_max_f], dim=1)
        hidden_features = self.dropout2(hidden_features)

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


class AASSIT3MultiTaskOutput(nn.Module):
    def __init__(self, hidden_dim, num_attack_types=19, use_quality_head=False):
        super(AASSIT3MultiTaskOutput, self).__init__()

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
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss


class CombinedLoss(nn.Module):

    def __init__(
        self,
        class_weights=None,
        focal_alpha=0.25,
        focal_gamma=2.0,
        ce_weight=1.0,
        focal_weight=1.0
    ):
        super(CombinedLoss, self).__init__()
        
        self.ce_weight = ce_weight
        self.focal_weight = focal_weight
        
        if class_weights is not None:
            self.register_buffer('class_weights', torch.tensor(class_weights))
        else:
            self.class_weights = None
        
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, weight=self.class_weights)
        focal_loss = self.focal_loss(inputs, targets)
        total_loss = self.ce_weight * ce_loss + self.focal_weight * focal_loss
        
        return total_loss


class MetricsCalculation:
    @staticmethod
    def compute_eer(labels, scores):

        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr
        
        eer_threshold = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer = 100 * (1 - interp1d(fpr, tpr)(eer_threshold))
        
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
        bonafide_mask = (labels == 1)
        spoof_mask = (labels == 0)
        
        bonafide_acc = 100 * (predictions[bonafide_mask] == labels[bonafide_mask]).sum() / (bonafide_mask.sum() + 1e-8)
        spoof_acc = 100 * (predictions[spoof_mask] == labels[spoof_mask]).sum() / (spoof_mask.sum() + 1e-8)
        
        tp = ((predictions == 1) & (labels == 1)).sum()
        fp = ((predictions == 1) & (labels == 0)).sum()
        fn = ((predictions == 0) & (labels == 1)).sum()
        
        precision = tp / (tp + fp + 1e-8)
        recall = tp / (tp + fn + 1e-8)
        f1 = 2 * precision * recall / (precision + recall + 1e-8)
        
        return {
            'eer': eer,
            'eer_threshold': eer_threshold,
            'min_dcf': min_dcf,
            'accuracy': accuracy,
            'bonafide_acc': bonafide_acc,
            'spoof_acc': spoof_acc,
            'precision': precision,
            'recall': recall,
            'f1': f1
        }


class TrainAASIST3:
    
    def __init__(
        self,
        model,
        optimizer,
        criterion,
        device='cuda',
        scheduler=None,
        checkpoint_dir='checkpoints',
        experiment_name='aasist3',
        accumulation_steps=1,
        use_amp=True
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        self.scheduler = scheduler
        self.last_val_metrics = None
        self.gap_threshold = 20.0
        self.skipped_steps = 0
        
        self.checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
        self.weights_dir = os.path.join(self.checkpoint_dir, 'weights')
        os.makedirs(self.weights_dir, exist_ok=True)
        
        self.experiment_name = experiment_name
        
        self.current_epoch = 0
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
                
                last_grad_norm = torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=0.5).item()
                epoch_grad_norms.append(last_grad_norm)
                
                if math.isnan(last_grad_norm) or math.isinf(last_grad_norm):
                    print(f"\n[WARNING] Gradient explosion detected at epoch {self.current_epoch+1}, batch {batch_idx+1}! Norm: {last_grad_norm}")
                elif last_grad_norm == 0.0:
                    print(f"\n[WARNING] Zero gradient detected at epoch {self.current_epoch+1}, batch {batch_idx+1}! Potential vanishing gradient or dead layers.")
                
                old_scale = self.scaler.get_scale()
                self.scaler.step(self.optimizer)
                self.scaler.update()
                
                if self.scaler.get_scale() < old_scale:
                    self.skipped_steps += 1
                
                self.optimizer.zero_grad()
            
            with torch.no_grad():
                preds = logits.argmax(dim=1)
                total_samples += labels.size(0)
                
                bonafide_mask = (labels == 1)
                spoof_mask = (labels == 0)
                total_bonafide_correct += (preds[bonafide_mask] == labels[bonafide_mask]).sum().item()
                total_bonafide += bonafide_mask.sum().item()
                total_spoof_correct += (preds[spoof_mask] == labels[spoof_mask]).sum().item()
                total_spoof += spoof_mask.sum().item()
                
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
        
        grad_stats = {
            'avg': np.mean(epoch_grad_norms) if epoch_grad_norms else 0,
            'min': np.min(epoch_grad_norms) if epoch_grad_norms else 0,
            'max': np.max(epoch_grad_norms) if epoch_grad_norms else 0
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
    
    def fit(self, train_loader, val_loader, num_epochs, early_stopping_patience=10):

        print(f"Starting Training: {self.experiment_name}")
        patience_counter = 0
        
        for epoch in range(num_epochs):
            self.current_epoch = epoch
            
            train_loss, train_acc, train_bn_acc, train_sp_acc, grad_stats, overfit_stats = self.train_epoch(train_loader)
            
            val_metrics = self.validate(val_loader)
            self.last_val_metrics = val_metrics
            
            self.history['train_loss'].append(train_loss)
            self.history['train_acc'].append(train_acc)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_eer'].append(val_metrics['eer'])
            self.history['val_min_dcf'].append(val_metrics['min_dcf'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            
            # Overfitting Analysis Dashboard
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
            
            train_info = {
                'loss': train_loss,
                'accuracy': train_acc,
                'bonafide_acc': train_bn_acc,
                'spoof_acc': train_sp_acc,
                'grad_stats': grad_stats,
                'overfit_stats': overfit_stats,
                'skipped_steps': self.skipped_steps
            }
            self.save_epoch_log(epoch + 1, val_metrics, train_info)
            
            if is_best:
                self.best_eer = val_metrics['eer']
                self.best_min_dcf = val_metrics['min_dcf']
                print(f"  ✓ New best model saved (EER: {self.best_eer:.2f}%)")
                patience_counter = 0
            else:
                patience_counter += 1
            
            print(f"  ✓ Saved checkpoint and log for epoch {epoch+1}")
            
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                break
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"Best EER: {self.best_eer:.2f}%")
        print(f"Best minDCF: {self.best_min_dcf:.4f}")
        print(f"{'='*70}\n")
        
        self.save_history()
    
    def save_checkpoint(self, epoch, is_best=False):
        """Save model checkpoint with custom naming."""
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_eer': self.best_eer,
            'best_min_dcf': self.best_min_dcf,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        filename = f"KANweightsE{epoch}.pth"
        filepath = os.path.join(self.weights_dir, filename)
        torch.save(checkpoint, filepath)
        
        root_filename = f"AASIST3_Epoch{epoch}.pth"
        torch.save(checkpoint, root_filename)
        
        if is_best:
            best_path = os.path.join(self.weights_dir, "KANweights_best.pth")
            torch.save(checkpoint, best_path)
            torch.save(checkpoint, "aasist3_best.pth")

    def save_epoch_log(self, epoch, val_metrics, train_info):
        """Save detailed per-epoch metrics."""
        log_data = {
            'epoch': epoch,
            'timestamp': datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            'train_loss': train_info['loss'] if isinstance(train_info, dict) else train_info,
            'train_accuracy': train_info['accuracy'] if isinstance(train_info, dict) else 0,
            'train_bonafide_acc': train_info.get('bonafide_acc', 0) if isinstance(train_info, dict) else 0,
            'train_spoof_acc': train_info.get('spoof_acc', 0) if isinstance(train_info, dict) else 0,
            'grad_stats': train_info.get('grad_stats', {}) if isinstance(train_info, dict) else {},
            'overfit_stats': train_info.get('overfit_stats', {}) if isinstance(train_info, dict) else {},
            'skipped_steps': train_info.get('skipped_steps', 0) if isinstance(train_info, dict) else self.skipped_steps,
            'val_loss': val_metrics['loss'],
            'val_accuracy': val_metrics['accuracy'],
            'val_bonafide_acc': val_metrics.get('bonafide_acc', 0),
            'val_spoof_acc': val_metrics.get('spoof_acc', 0),
            'val_eer': val_metrics['eer'],
            'val_min_dcf': val_metrics['min_dcf'],
            'val_f1': val_metrics.get('f1', 0)
        }
        
        filename = f"Epoch{epoch}Log.json"
        filepath = os.path.join(self.weights_dir, filename)
        with open(filepath, 'w') as f:
            json.dump(log_data, f, indent=4)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        filepath = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device, weights_only=False)
        
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        
        if 'scheduler_state_dict' in checkpoint and self.scheduler is not None:
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        
        self.current_epoch = checkpoint['epoch']
        self.best_eer = checkpoint['best_eer']
        self.best_min_dcf = checkpoint['best_min_dcf']
        self.history = checkpoint['history']
        
        print(f"✓ Loaded checkpoint from epoch {self.current_epoch}")
        print(f"  Best EER: {self.best_eer:.2f}%")
        print(f"  Best minDCF: {self.best_min_dcf:.4f}")
    
    def save_history(self):
        """Save training history to JSON."""
        history_path = os.path.join(self.checkpoint_dir, 'training_history.json')
        with open(history_path, 'w') as f:
            json.dump(self.history, f, indent=2)


def count_parameters(model):
    """Count trainable parameters in model."""
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return {
        'total': total,
        'trainable': trainable,
        'frozen': total - trainable
    }


def print_training_summary(args, model, criterion, device):
    """Print training configuration and model summary."""
    print("\n" + "="*70)
    print(f"{'TRAINING CONFIGURATION SUMMARY':^70}")
    print("="*70)
    
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
        ("Weight Decay", 1e-4),
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
    
    print("\n" + "="*70)
    print(f"{'MODEL ARCHITECTURE DETAILS':^70}")
    print("="*70)
    
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
            
    dropouts = []
    for name, module in model.named_modules():
        if isinstance(module, nn.Dropout):
            dropouts.append(f"{name}: {module.p}")
    
    if dropouts:
        print("\nDropout Rates:")
        for drop in dropouts:
            print(f"  {drop}")
            
    print("="*70 + "\n")

def print_model_summary(model, input_size=(1, 128, 200)):
    print("MODEL FORWARD TEST")
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(input_size)
        try:
            output = model(dummy_input)
            print(f"Input shape:  {dummy_input.shape}")
            print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"Forward pass failed: {e}")


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

        waveform, _ = self.processor.load_audio(audio_path)
        audio = self.processor.process(waveform)
        
        audio = audio.unsqueeze(0).to(self.device).float()
        
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
        
        # Ensure audio is 1D for slicing if it came from load_audio
        if audio.dim() > 1:
            audio = audio.squeeze()

        for start in range(0, len(audio) - window_size + 1, stride):

            window = audio[start:start + window_size]
            window = self.processor.process(window)
            window = window.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(window)
                probs = F.softmax(logits, dim=1)
                scores.append(probs[0, 1].item())
        
        return np.mean(scores)


def run_mel_training():
    import argparse
    from torch.utils.data import DataLoader
    
    parser = argparse.ArgumentParser(description="AASIST3 Mel Training")
    parser.add_argument("--train_mel_dir", type=str, required=True, help="Directory containing training mel spectrograms")
    parser.add_argument("--train_protocol", type=str, required=True, help="Path to training protocol tsv file")
    parser.add_argument("--dev_mel_dir", type=str, required=True, help="Directory containing dev mel spectrograms")
    parser.add_argument("--dev_protocol", type=str, required=True, help="Path to dev protocol tsv file")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints", help="Directory to save checkpoints")
    parser.add_argument("--experiment_name", type=str, default="aasist3_mel_v1")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=5e-5)
    parser.add_argument("--max_len", type=int, default=200)
    parser.add_argument("--num_workers", type=int, default=4)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--subset", type=int, default=None)
    parser.add_argument("--accumulation_steps", type=int, default=1)
    parser.add_argument("--disable_amp", action="store_true")

    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    print("Loading datasets...")
    train_dataset = ASV5MelDataset(args.train_mel_dir, args.train_protocol, max_len=args.max_len)
    dev_dataset = ASV5MelDataset(args.dev_mel_dir, args.dev_protocol, max_len=args.max_len)

    if args.subset:
        train_indices = torch.randperm(len(train_dataset))[:args.subset]
        train_dataset = torch.utils.data.Subset(train_dataset, train_indices)
        
        dev_indices = torch.randperm(len(dev_dataset))[:args.subset]
        dev_dataset = torch.utils.data.Subset(dev_dataset, dev_indices)
        print(f"Using subset of {args.subset} training and validation samples.")

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True)
    dev_loader = DataLoader(dev_dataset, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, pin_memory=True)

    print("Initializing AASIST3_Mel model...")
    model = AASIST3_Mel(in_channels=128)
    optimizer = AdamW(model.parameters(), lr=args.lr, weight_decay=1e-4)
    criterion = CombinedLoss()
    scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)

    # Print training summary before starting
    print_training_summary(args, model, criterion, device)
    print_model_summary(model, input_size=(args.batch_size, 128, args.max_len))

    trainer = TrainAASIST3(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        device=device,
        scheduler=scheduler,
        checkpoint_dir=args.checkpoint_dir,
        experiment_name=args.experiment_name,
        accumulation_steps=args.accumulation_steps,
        use_amp=not args.disable_amp
    )

    trainer.fit(train_loader, dev_loader, num_epochs=args.epochs, early_stopping_patience=args.patience)

if __name__ == "__main__":
    run_mel_training()
