
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import math

class SincConv(nn.Module):
    def __init__(self, out_channels=70, kernel_size=129, sample_rate=16000):
        super(SincConv, self).__init__()

        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.sample_rate = sample_rate


        low_hz = 30
        high_hz = sample_rate / 2 - (low_hz)


        mel = np.linspace(self._to_mel(low_hz), self._to_mel(high_hz), out_channels + 1)
        hz = self._to_hz(mel)


        self.low_hz_ = nn.Parameter(torch.Tensor(hz[:-1]).view(-1, 1))
        self.band_hz_ = nn.Parameter(torch.Tensor(np.diff(hz)).view(-1, 1))


        n_lin = torch.linspace(0, (self.kernel_size / 2) - 1, steps=int((self.kernel_size / 2)))
        self.window_ = 0.54 - 0.46 * torch.cos(2 * math.pi * n_lin / self.kernel_size)

        n = (self.kernel_size - 1) / 2.0
        self.n_ = 2 * math.pi * torch.arange(-n, 0).view(1, -1) / self.sample_rate

    def _to_mel(self, hz):
        return 2595 * np.log10(1 + hz / 700)

    def _to_hz(self, mel):
        return 700 * (10 ** (mel / 2595) - 1)

    def forward(self, x):
        self.n_ = self.n_.to(x.device)
        self.window_ = self.window_.to(x.device)

        low = self.low_hz_
        high = torch.clamp(low + self.band_hz_, 
                          min=self.low_hz_.data[0].item(), 
                          max=self.sample_rate / 2)
        band = (high - low)[:, 0]

        f_times_t_low = torch.matmul(low, self.n_)
        f_times_t_high = torch.matmul(high, self.n_)

        n_half = self.n_ / 2
        n_half = torch.where(torch.abs(n_half) < 1e-7, torch.ones_like(n_half) * 1e-7, n_half)
        
        band_pass_left = ((torch.sin(f_times_t_high) - torch.sin(f_times_t_low)) / n_half) * self.window_
        band_pass_center = 2 * band.view(-1, 1)
        band_pass_right = torch.flip(band_pass_left, dims=[1])

        band_pass = torch.cat([band_pass_left, band_pass_center, band_pass_right], dim=1)
        band_pass = band_pass / (2 * band[:, None] + 1e-7)

        self.filters = band_pass.view(self.out_channels, 1, self.kernel_size)

        return F.conv1d(x, self.filters, stride=1, padding=self.kernel_size // 2, groups=1)


class ResBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=(2, 3), stride=1, padding=(1, 1)):
        super(ResBlock, self).__init__()

        self.bn1 = nn.BatchNorm2d(in_channels)
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size, stride, padding)

        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size, stride, padding)

        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, bias=False),
                nn.BatchNorm2d(out_channels)
            )

        self.maxpool = nn.MaxPool2d((1, 3))

    def selu(self, x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946
        x_clamped = torch.clamp(x, min=-10, max=10)
        return scale * torch.where(x > 0, x, alpha * (torch.exp(x_clamped) - 1))

    def forward(self, x):
        identity = x

        out = self.selu(self.bn1(x))
        out = self.conv1(out)
        out = self.selu(self.bn2(out))
        out = self.conv2(out)

        if out.shape != identity.shape:
            identity = self.shortcut(identity)
        
            if out.shape[2:] != identity.shape[2:]:
                identity = F.adaptive_avg_pool2d(identity, out.shape[2:])
        else:
            identity = self.shortcut(identity)

        out = out + identity
        out = self.maxpool(out)
        return out


class GraphAttentionLayer(nn.Module):
    def __init__(self, in_dim, out_dim):
        super(GraphAttentionLayer, self).__init__()
        self.in_dim = in_dim
        self.out_dim = out_dim

        self.W_att = nn.Linear(in_dim, out_dim, bias=False)
        self.W_res = nn.Linear(in_dim, out_dim, bias=False)
        self.W_map = nn.Linear(in_dim, 1, bias=False)

        self.bn = nn.BatchNorm1d(out_dim)

    def selu(self, x):
        alpha = 1.6732632423543772848170429916717
        scale = 1.0507009873554804934193349852946

        x_clamped = torch.clamp(x, min=-10, max=10)
        return scale * torch.where(x > 0, x, alpha * (torch.exp(x_clamped) - 1))

    def forward(self, x):

        batch_size, num_nodes, _ = x.size()


        x_i = x.unsqueeze(2).expand(-1, -1, num_nodes, -1)  # [B, N, N, D]
        x_j = x.unsqueeze(1).expand(-1, num_nodes, -1, -1)  # [B, N, N, D]


        x_prod = x_i * x_j  # [B, N, N, D]


        attn_scores = self.W_map(x_prod).squeeze(-1)  # [B, N, N]
        attn_weights = F.softmax(attn_scores, dim=-1)  # [B, N, N]


        aggregated = torch.matmul(attn_weights, x)  # [B, N, D]


        out = self.W_att(aggregated) + self.W_res(x)


        out = out.transpose(1, 2)  # [B, D, N]
        out = self.bn(out)
        out = out.transpose(1, 2)  # [B, N, D]
        out = self.selu(out)

        return out


class GraphPoolingLayer(nn.Module):
    def __init__(self, in_dim, ratio=0.8):
        super(GraphPoolingLayer, self).__init__()
        self.in_dim = in_dim
        self.ratio = ratio

        self.projection = nn.Linear(in_dim, 1)

    def forward(self, x):

        batch_size, num_nodes, _ = x.size()


        scores = self.projection(x).squeeze(-1)  # [B, N]


        k = max(1, int(num_nodes * self.ratio))
        top_scores, top_indices = torch.topk(scores, k, dim=1)  # [B, k]


        gate = torch.sigmoid(top_scores).unsqueeze(-1)  # [B, k, 1]

        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, k)
        selected_nodes = x[batch_indices, top_indices]  # [B, k, D]
        pooled = selected_nodes * gate

        return pooled


class RawGAT_ST(nn.Module):
    def __init__(self, num_classes=2):
        super(RawGAT_ST, self).__init__()

        self.sinc_layer = SincConv(out_channels=70, kernel_size=129)

        self.maxpool_init = nn.MaxPool2d((3, 3))
        self.res_blocks1 = nn.Sequential(
            ResBlock(1, 32),
            ResBlock(32, 32)
        )

        self.res_blocks2 = nn.Sequential(
            ResBlock(32, 64),
            ResBlock(64, 64),
            ResBlock(64, 64),
            ResBlock(64, 64)
        )

        self.spectral_gat = GraphAttentionLayer(64, 32)
        self.spectral_pool = GraphPoolingLayer(32, ratio=0.64)
        self.spectral_proj = nn.Linear(32, 32)
        self.temporal_gat = GraphAttentionLayer(64, 32)
        self.temporal_pool = GraphPoolingLayer(32, ratio=0.81)
        self.temporal_proj = nn.Linear(32, 32)

        self.st_gat = GraphAttentionLayer(32, 16)
        self.st_pool = GraphPoolingLayer(16, ratio=0.64)
        self.st_proj = nn.Linear(16, 1)
        self.fc = nn.Linear(7, num_classes)
        
        self._init_weights()

    def _init_weights(self):
        """LeCun normal initialization for SELU activation"""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.Linear):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)

    def forward(self, x):
        x = x.unsqueeze(1)  # [batch, 1, samples]

        x = torch.abs(self.sinc_layer(x))  # [batch, 70, samples]
        x = x.unsqueeze(1)  # [batch, 1, 70, samples]

        x = self.maxpool_init(x)  # [batch, 1, 23, samples//3]

        x = self.res_blocks1(x)  # [batch, 32, 23, T1]
        x = self.res_blocks2(x)  # [batch, 64, 23, T2]

        batch_size, channels, freq, time = x.size()

        spectral_feat = torch.max(torch.abs(x), dim=3)[0]  # [B, C, F]
        spectral_feat = spectral_feat.transpose(1, 2)  # [B, F, C]
        spectral_feat = self.spectral_gat(spectral_feat)  # [B, F, 32]
        spectral_feat = self.spectral_pool(spectral_feat)  # [B, F', 32]

        spectral_feat_proj = self.spectral_proj(spectral_feat)  # [B, F', 32]
        temporal_feat = torch.max(torch.abs(x), dim=2)[0]  # [B, C, T]
        temporal_feat = temporal_feat.transpose(1, 2)  # [B, T, C]
        temporal_feat = self.temporal_gat(temporal_feat)  # [B, T, 32]
        temporal_feat = self.temporal_pool(temporal_feat)  # [B, T', 32]

        temporal_feat_proj = self.temporal_proj(temporal_feat)  # [B, T', 32]

        target_nodes = 12

        if spectral_feat_proj.size(1) != target_nodes:
            spectral_feat_proj = F.interpolate(
                spectral_feat_proj.transpose(1, 2), 
                size=target_nodes, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)

        if temporal_feat_proj.size(1) != target_nodes:
            temporal_feat_proj = F.interpolate(
                temporal_feat_proj.transpose(1, 2), 
                size=target_nodes, 
                mode='linear', 
                align_corners=False
            ).transpose(1, 2)

        fused = spectral_feat_proj * temporal_feat_proj  # [B, 12, 32]

        st_feat = self.st_gat(fused)  # [B, 12, 16]
        st_feat = self.st_pool(st_feat)  # [B, N', 16]
        st_feat = self.st_proj(st_feat).squeeze(-1)  # [B, N']
        if st_feat.size(1) != 7:
            st_feat = F.adaptive_avg_pool1d(st_feat.unsqueeze(1), 7).squeeze(1)

        output = self.fc(st_feat)
        output = torch.clamp(output, min=-10, max=10)

        return output
