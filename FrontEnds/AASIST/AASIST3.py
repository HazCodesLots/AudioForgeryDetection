import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.optim import Adam, AdamW
from torch.optim.lr_scheduler import CosineAnnealingLR, ReduceLROnPlateau
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from tqdm import tqdm
import os
import json
from datetime import datetime
from typing import Tuple, Optional
import random



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

    def _compute_bspline_basis(self, x):
        x = x.unsqueeze(-1)
        grid = self.grid
        num_coeffs = self.grid_size + self.spline_order

        bases = []

        for i in range(num_coeffs):

            mask = (x >= grid[i]) & (x < grid[i + 1])
            basis = mask.float()

            for k in range(1, self.spline_order + 1):
                left_den = grid[i + k] - grid[i]
                if left_den > 1e-8:
                    left_term = ((x - grid[i]) / left_den) * basis
                else:
                    left_term = torch.zeros_like(basis)

                if i + k + 1 < len(grid):
                    right_den = grid[i + k + 1] - grid[i + 1]
                    if right_den > 1e-8:
                        next_mask = (x >= grid[i + 1]) & (x < grid[i + k + 1])
                        next_basis = next_mask.float()
                        right_term = ((grid[i + k + 1] - x) / right_den) * next_basis
                    else:
                        right_term = torch.zeros_like(basis)
                else:
                    right_term = torch.zeros_like(basis)

                basis = left_term + right_term

            bases.append(basis)

        return torch.cat(bases, dim=-1)

    def _prelu(self, x):

        x_expanded = x.unsqueeze(0)
        pos = F.relu(x_expanded)
        neg = torch.min(x_expanded, torch.zeros_like(x_expanded))

        prelu_weight = self.prelu_weight.view(self.out_features, *([1] * x.dim()), self.in_features)
        return pos + prelu_weight * neg

    def forward(self, x):
        batch_shape = x.shape[:-1]

        x_clamped = torch.clamp(x, self.grid_range[0], self.grid_range[1])

        base_output = self._prelu(x_clamped)

        bspline_basis = self._compute_bspline_basis(x_clamped)

        coeffs_exp = self.spline_coeffs.view(self.out_features, *([1] * len(batch_shape)), self.in_features, -1)
        
        basis_exp = bspline_basis.unsqueeze(0)

        spline_output = (coeffs_exp * basis_exp).sum(dim=-1)

        base_w =  self.base_weight.view(self.out_features, *([1] * len(batch_shape)), self.in_features)
        spline_w = self.spline_weight.view(self.out_features, *([1]* len(batch_shape)), self.in_features)

        output = base_w * base_output + spline_w * spline_output
        output = output.sum(dim=-1)
        output = output.permute(*range(1, len(output.shape)), 0)

        return output


class PreEmphasis(nn.Module:):
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

        if (audio_length - windows_samples) % stride_samples !=0:
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

import torch
import torch.nn as nn
import torch.nn.functional as F


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
                stride=stride
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
        self.batch_norm = nn.BatchNorm1d(num_nodes)
    
    def forward(self, h):
        batch_size = h.size(0)
        h = self.dropout(h)
        h_expanded_i = h.unsqueeze(2)
        h_expanded_j = h.unsqueeze(1)

        node_products = h_expanded_i * h_expanded_j
        original_shape = node_products.shape
        node_products_flat = node_products.view(-1, self.in_dim)
        kan_out = self.kan_attention(node_products_flat)
        kan_out = kan_out.view(batch_size, self.num_nodes, self.num_nodes, self.in_dim)
        kan_out = torch.tanh(kan_out)
        attention_scores = torch.einsum('bnmd,dk->bnmk', kan_out, self.W_att)
        attention_scores = attention_scores.mean(dim=-1)
        attention_map = F.softmax(attention_scores / self.temprature, dim=-1)
        h_attended = torch.bmm(attention_map, h)
        h_attended_flat = h_attended.view(-1, self.in_dim)
        kan2_out = self.kan_attn_proj(h_attended_flat)
        kan2_out = kan2_out.view(batch_size, self.num_nodes, self.out_dim)

        h_flat = h.view(-1, self.in_dim)
        kan3_out = self.kan_direct_proj(h_flat)
        kan3_out = kan3_out.view(batch_size, self.num_nodes, self.out_dim)

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

    def forward(self, k=None):
        batch_size, num_nodes, in_dim = h.shape
        if k is None:
            k = max(1, int(num_nodes * self.ratio))
        h_drop = self.dropout(h)
        h_flat = h_drop.view(-1, in_dim)
        scores_flat = self.kan_score(h_flat)
        scores = scores_flat.view(batch_size, num_nodes)
        scores = torch.sigmoid(scores)
        h_gated = h * scores.unsqueeze(-1)
        tok_k_scores, top_k_indices = torch.topk(scores, k, dim=1)
        top_k_indices_expanded = top_k_indices.unsqueeze(-1).expand(-1, -1, in_dim)
        h_pooled = torch.gather(h_gated, 1, top_k_indices_expanded)
        return h_pooled

class KAN_HS_GAL(nn.Module):
    def __init__(self, temporal_dim, spatial_dim, num_nodes, stack_dim, temprature=1.0, dropout=0.2):
        super(KAN_HS_GAL, self).__init__()

        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.num_nodes = num_nodes
        self.stack_dim = stack_dim
        self.hetero_dim = temporal_dim + spatial_dim
        self.temprature = temprature

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
        
        self.batch_norm = nn.BatchNorm1d(num_nodes)
    
    def forward(self, h_t, h_s, S):

        batch_size = h_t.size(0)
        
        h_t_proj = self.kan_temporal_proj(h_t.view(-1, self.temporal_dim))
        h_t_proj = h_t_proj.view(batch_size, self.num_nodes, self.temporal_dim)
        
        h_s_proj = self.kan_spatial_proj(h_s.view(-1, self.spatial_dim))
        h_s_proj = h_s_proj.view(batch_size, self.num_nodes, self.spatial_dim)
        
        h_st = torch.cat([h_t_proj, h_s_proj], dim=-1)
        
        h_st = self.dropout(h_st)
        
        h_st_i = h_st.unsqueeze(2)
        h_st_j = h_st.unsqueeze(1)
        node_products = h_st_i * h_st_j
        
        node_products_flat = node_products.view(-1, self.hetero_dim)
        primary_attn_flat = self.kan_primary_attn(node_products_flat)
        primary_attn = primary_attn_flat.view(batch_size, self.num_nodes, self.num_nodes, self.hetero_dim)
        A = torch.tanh(primary_attn)
        
        B = torch.zeros(batch_size, self.num_nodes, self.num_nodes, device=h_st.device)
        
        for i in range(self.num_nodes):
            for j in range(self.num_nodes):
                if i < self.temporal_dim and j < self.temporal_dim:
                    weight = self.W11
                elif i >= self.temporal_dim and j >= self.temporal_dim:
                    weight = self.W22
                else:
                    weight = self.W12
                
                B[:, i, j] = (A[:, i, j, :] * weight).sum(dim=-1)
        
        B_hat = F.softmax(B / self.temperature, dim=-1)
        
        S_expanded = S.unsqueeze(1).expand(-1, self.num_nodes, -1)
        S_padded = F.pad(S_expanded, (0, self.hetero_dim - self.stack_dim))
        
        h_st_stack = h_st * S_padded
        
        h_st_stack_flat = h_st_stack.view(-1, self.hetero_dim)
        stack_attn_flat = self.kan_stack_attn(h_st_stack_flat)
        stack_attn = stack_attn_flat.view(batch_size, self.num_nodes, self.hetero_dim)
        stack_attn = torch.tanh(stack_attn)
        
        stack_scores = torch.matmul(stack_attn, self.W_m).squeeze(-1)
        A_m = F.softmax(stack_scores / self.temperature, dim=-1)
        
        h_st_weighted = (h_st * A_m.unsqueeze(-1)).sum(dim=1)
        
        S_update1 = self.kan_stack_update1(h_st_weighted)
        S_update2 = self.kan_stack_update2(S)
        S_new = S_update1 + S_update2
        
        h_st_attended = torch.bmm(B_hat, h_st)
        
        h_st_update1_flat = h_st_attended.view(-1, self.hetero_dim)
        h_st_update1 = self.kan_hetero_update1(h_st_update1_flat)
        h_st_update1 = h_st_update1.view(batch_size, self.num_nodes, self.hetero_dim)
        
        h_st_update2_flat = h_st.view(-1, self.hetero_dim)
        h_st_update2 = self.kan_hetero_update2(h_st_update2_flat)
        h_st_update2 = h_st_update2.view(batch_size, self.num_nodes, self.hetero_dim)
        
        h_st_new = h_st_update1 + h_st_update2
        
        h_st_new = h_st_new.transpose(1, 2)
        h_st_new = self.batch_norm(h_st_new)
        h_st_new = h_st_new.transpose(1, 2)
        
        I_t = torch.eye(self.num_nodes, device=h_st.device)[:, :self.temporal_dim]
        zeros_s = torch.zeros(self.num_nodes, self.spatial_dim, device=h_st.device)
        M_t = torch.cat([I_t, zeros_s], dim=1)
        
        zeros_t = torch.zeros(self.num_nodes, self.temporal_dim, device=h_st.device)
        I_s = torch.eye(self.num_nodes, device=h_st.device)[:, :self.spatial_dim]
        M_s = torch.cat([zeros_t, I_s], dim=1)
        
        h_t_new = torch.matmul(h_st_new, M_t.t())
        h_s_new = torch.matmul(h_st_new, M_s.t())
        
        return h_t_new, h_s_new, S_new


class GraphPositionalEmbedding(nn.Module):
    def __init__(self, num_nodes, embedding_dim):

        super(PositionalEmbedding, self).__init__()

        self.num_nodes = num_nodes
        self.embedding_dim = embedding_dim

        self.pos_embedding = nn.Parameter(torch.randn(1, num_nodes, embedding_dim))
        nn.init.normal_(self.pos_embedding, mean=0.0, std=0.02)

    def forward(self, x):
        batch_size = x.size(0)
        pos_emb = self.pos_embedding.expand(batch_size, -1, -1)
        return x + pos_emb


class GraphFormation(nn.Module):

    def __init__(self, encoder_dim, num_temporal_nodes = 100, , num_spatial_nodes = 100, temporal_dim = 64, spatial_dim = 64, pool_ratio = 0.5, temperature = 1.0):

        super(GraphFormation, self).__init__()
        self.encoder_dim = encoder_dim
        self.num_temporal_nodes = num_temporal_nodes
        self.num_spatial_nodes = num_spatial_nodes
        self.temporal_dim = temporal_dim
        self.spatial_dim = spatial_dim
        self.temporal_projection = nn.Linear(encoder_dim, temporal_dim)
        self.spatial_projection = nn.Linear(encoder_dim, spatial_dim)
        self.pe_temporal = PositionalEmbedding(num_temporal_nodes, temporal_dim)
        self.pe_spatial = PositionalEmbedding(num_spatial_nodes, spatial_dim)

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
        pooled = f.adaptive_max_pool1d(x_transposed.transpose(1,2), self.num_spatial_nodes)
        spatial_features = pooled.transpose(1,2)
        return spatial_features

    
    def forward(self, encoder_output):

        batch_size = encoder_output.size(0)

        temporal_features = self._temporal_max_pooling(encoder_output)
        temporal_features = self._temporal_projection(temporal_features)
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

        self.hs_gal_1 = KAN_GAL(temporal_dim=temporal_dim, spatial_dim= spatial_dim, num_nodes=max(num_temporal_nodes, num_spatial_nodes), stack_dim=stack_dim, temperature=temperature)
        self.pooled_temporal = KAN_GraphPool(temporal_dim, ratio=pool_ratio)
        self.pooled_spatial = KAN_GraphPool(spatial_dim, ratio=pool_ratio)

        self.pooled_temporal_nodes = max(1, int(num_temporal_nodes * pool_ratio))
        self.pooled_spatial_nodes = max(1, int(num_spatial_nodes * pool_ratio))

        self.hs_gal_2 = KAN_HS_GAL(temporal_dim=temporal_dim, spatial_dim=spatial_dim, num_nodes=max(self.pooled_temporal_nodes, self.pooled_spatial_nodes), stack_dim=stack_dim, temperature=temperature)

    def forward(self, h_t, h_s, S):
        h_t_2, h_s_2, S_2 = self.hs_gal_1(h_t, h_s, S)

        h_t_pooled = self.pool_temporal(h_t_2, k=self.pooled_temporal_nodes)
        h_s_pooled = self.pool_spatial(h_s_2, k=self.pooled_spatial_nodes)

        h_t_3, h_s_3, S_3 = self.hs_gal_2(h_t_pooledm h_s_pooled, S_2)

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

        for i, branch in enumerate(self.brnaches):
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
                h_s_padded = F.pad(h_s, 0, 0, 0, pad_size)
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
        self.embedding_dim - embedding_dim
        self.num_classes = num_classes

        self.embedding_kan = KANLayer(hidden_dim, embedding_dim)
        self.classifier_kan = KANLayer(embedding_dim, num_classes)
        self.bn_embedding = nn.BatchNorm1d(embedding_dim)

    def forwad(self, hidden_features, return_embedding=False):
        embedding = self.embedding_kanh(hidden_features)
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
        outputs['attack_type'] = self.attack_type_head(hidden_features)
        if self.use_quality_head:
            outputs['quality'] = self.quality_head(hidden_features)
        return outputs


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
        
        eer, eer_threshold = MetricsCalculator.compute_eer(labels, scores)
        min_dcf = MetricsCalculator.compute_min_dcf(labels, scores)
        accuracy = MetricsCalculator.compute_accuracy(labels, predictions)
        
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
        experiment_name='aasist3'
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        
        self.checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
        os.makedirs(self.checkpoint_dir, exist_ok=True)
        
        self.experiment_name = experiment_name
        
        self.current_epoch = 0
        self.best_eer = float('inf')
        self.best_min_dcf = float('inf')
        self.history = {
            'train_loss': [],
            'val_loss': [],
            'val_eer': [],
            'val_min_dcf': [],
            'val_accuracy': []
        }
    
    def train_epoch(self, train_loader):
        self.model.train()
        total_loss = 0
        num_batches = 0
        
        pbar = tqdm(train_loader, desc=f'Epoch {self.current_epoch+1} [Train]')
        
        for batch_idx, batch in enumerate(pbar):
            if len(batch) == 2:
                audio, labels = batch
                audio = audio.to(self.device)
                labels = labels.to(self.device)
            else:
                continue
            
            self.optimizer.zero_grad()
            logits = self.model(audio)
            
            loss = self.criterion(logits, labels)
            
            loss.backward()
            
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            
            self.optimizer.step()
            
            total_loss += loss.item()
            num_batches += 1
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / num_batches
        return avg_loss
    
    def validate(self, val_loader):
        self.model.eval()
        total_loss = 0
        num_batches = 0
        
        all_labels = []
        all_scores = []
        all_predictions = []
        
        pbar = tqdm(val_loader, desc=f'Epoch {self.current_epoch+1} [Val]')
        
        with torch.no_grad():
            for batch in pbar:
                if len(batch) == 2:
                    audio, labels = batch
                    audio = audio.to(self.device)
                    labels = labels.to(self.device)
                else:
                    continue
                
                logits = self.model(audio)
                
                loss = self.criterion(logits, labels)
                total_loss += loss.item()
                num_batches += 1
                
                probs = F.softmax(logits, dim=1)
                spoofed_scores = probs[:, 1]  
                predictions = logits.argmax(dim=1)
                
                all_labels.append(labels.cpu().numpy())
                all_scores.append(spoofed_scores.cpu().numpy())
                all_predictions.append(predictions.cpu().numpy())
        
        all_labels = np.concatenate(all_labels)
        all_scores = np.concatenate(all_scores)
        all_predictions = np.concatenate(all_predictions)
        
        metrics = MetricsCalculator.compute_all_metrics(
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
            
            train_loss = self.train_epoch(train_loader)
            
            val_metrics = self.validate(val_loader)
            
            self.history['train_loss'].append(train_loss)
            self.history['val_loss'].append(val_metrics['loss'])
            self.history['val_eer'].append(val_metrics['eer'])
            self.history['val_min_dcf'].append(val_metrics['min_dcf'])
            self.history['val_accuracy'].append(val_metrics['accuracy'])
            
            print(f"\nEpoch {epoch+1}/{num_epochs}")
            print(f"  Train Loss: {train_loss:.4f}")
            print(f"  Val Loss:   {val_metrics['loss']:.4f}")
            print(f"  Val EER:    {val_metrics['eer']:.2f}%")
            print(f"  Val minDCF: {val_metrics['min_dcf']:.4f}")
            print(f"  Val Acc:    {val_metrics['accuracy']:.2f}%")
            
            if self.scheduler is not None:
                if isinstance(self.scheduler, ReduceLROnPlateau):
                    self.scheduler.step(val_metrics['eer'])
                else:
                    self.scheduler.step()
                
                current_lr = self.optimizer.param_groups[0]['lr']
                print(f"  Learning Rate: {current_lr:.6f}")
            
            if val_metrics['eer'] < self.best_eer:
                self.best_eer = val_metrics['eer']
                self.best_min_dcf = val_metrics['min_dcf']
                self.save_checkpoint('best_model.pth', val_metrics)
                print(f"  ✓ New best model saved (EER: {self.best_eer:.2f}%)")
                patience_counter = 0
            else:
                patience_counter += 1
            
            if (epoch + 1) % 5 == 0:
                self.save_checkpoint(f'checkpoint_epoch_{epoch+1}.pth', val_metrics)
            
            if patience_counter >= early_stopping_patience:
                print(f"\n⚠ Early stopping triggered after {epoch+1} epochs")
                break
        
        print(f"\n{'='*70}")
        print(f"Training Complete!")
        print(f"Best EER: {self.best_eer:.2f}%")
        print(f"Best minDCF: {self.best_min_dcf:.4f}")
        print(f"{'='*70}\n")
        
        self.save_history()
    
    def save_checkpoint(self, filename, metrics=None):
        """Save model checkpoint."""
        checkpoint = {
            'epoch': self.current_epoch,
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'best_eer': self.best_eer,
            'best_min_dcf': self.best_min_dcf,
            'history': self.history
        }
        
        if self.scheduler is not None:
            checkpoint['scheduler_state_dict'] = self.scheduler.state_dict()
        
        if metrics is not None:
            checkpoint['metrics'] = metrics
        
        filepath = os.path.join(self.checkpoint_dir, filename)
        torch.save(checkpoint, filepath)
    
    def load_checkpoint(self, filename):
        """Load model checkpoint."""
        filepath = os.path.join(self.checkpoint_dir, filename)
        checkpoint = torch.load(filepath, map_location=self.device)
        
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


def print_model_summary(model, input_size=(1, 64000)):
    print("MODEL SUMMARY")

    params = count_parameters(model)
    print(f"Total parameters:     {params['total']:,}")
    print(f"Trainable parameters: {params['trainable']:,}")
    print(f"Frozen parameters:    {params['frozen']:,}")
    
    model.eval()
    with torch.no_grad():
        dummy_input = torch.randn(input_size)
        try:
            output = model(dummy_input)
            print(f"\nInput shape:  {dummy_input.shape}")
            print(f"Output shape: {output.shape}")
        except Exception as e:
            print(f"\nForward pass failed: {e}")


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

        audio = self.processor.load_audio(audio_path)
        audio = self.processor.process(audio)
        
        audio = audio.unsqueeze(0).to(self.device)
        
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

        audio = self.processor.load_audio(audio_path)
        
        scores = []
        stride = window_size - overlap
        
        for start in range(0, len(audio) - window_size + 1, stride):

            window = audio[start:start + window_size]
            window = self.processor.process(window)
            window = window.unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                logits = self.model(window)
                probs = F.softmax(logits, dim=1)
                scores.append(probs[0, 1].item())
        
        return np.mean(scores)
