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

        filters = band_pass.view(self.out_channels, 1, self.kernel_size)
        return F.conv1d(x, filters, stride=1, padding=self.kernel_size // 2, groups=1)


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
        x_i = x.unsqueeze(2).expand(-1, -1, num_nodes, -1)
        x_j = x.unsqueeze(1).expand(-1, num_nodes, -1, -1)
        attn_scores = self.W_map(x_i * x_j).squeeze(-1)
        attn_weights = F.softmax(attn_scores, dim=-1)
        aggregated = torch.matmul(attn_weights, x)
        out = self.W_att(aggregated) + self.W_res(x)
        out = out.transpose(1, 2)
        out = self.bn(out)
        out = out.transpose(1, 2)
        out = self.selu(out)
        return out


class GraphPoolingLayer(nn.Module):
    def __init__(self, in_dim, ratio=0.8):
        super(GraphPoolingLayer, self).__init__()
        self.ratio = ratio
        self.projection = nn.Linear(in_dim, 1)

    def forward(self, x):
        batch_size, num_nodes, _ = x.size()
        scores = self.projection(x).squeeze(-1)
        k = max(1, int(num_nodes * self.ratio))
        top_scores, top_indices = torch.topk(scores, k, dim=1)
        gate = torch.sigmoid(top_scores).unsqueeze(-1)
        batch_indices = torch.arange(batch_size, device=x.device).unsqueeze(1).expand(-1, k)
        selected_nodes = x[batch_indices, top_indices]
        return selected_nodes * gate


class RawGAT_ST(nn.Module):
    def __init__(self, num_classes=2):
        super(RawGAT_ST, self).__init__()
        self.sinc_layer = SincConv(out_channels=70, kernel_size=129)
        self.maxpool_init = nn.MaxPool2d((3, 3))
        self.res_blocks1 = nn.Sequential(ResBlock(1, 32), ResBlock(32, 32))
        self.res_blocks2 = nn.Sequential(ResBlock(32, 64), ResBlock(64, 64), ResBlock(64, 64), ResBlock(64, 64))
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
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.Conv1d, nn.Linear)):
                nn.init.kaiming_normal_(m.weight, mode='fan_in', nonlinearity='linear')
                if hasattr(m, 'bias') and m.bias is not None: nn.init.zeros_(m.bias)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight); nn.init.zeros_(m.bias)

    def forward(self, x):
        if x.dim() == 2: x = x.unsqueeze(1)
        x = torch.abs(self.sinc_layer(x)).unsqueeze(1)
        x = self.maxpool_init(x)
        x = self.res_blocks1(x)
        x = self.res_blocks2(x)
        batch_size, channels, freq, time = x.size()
        spectral_feat = torch.max(torch.abs(x), dim=3)[0].transpose(1, 2)
        spectral_feat = self.spectral_pool(self.spectral_gat(spectral_feat))
        spectral_feat_proj = self.spectral_proj(spectral_feat)
        temporal_feat = torch.max(torch.abs(x), dim=2)[0].transpose(1, 2)
        temporal_feat = self.temporal_pool(self.temporal_gat(temporal_feat))
        temporal_feat_proj = self.temporal_proj(temporal_feat)
        
        target_nodes = 12
        if spectral_feat_proj.size(1) != target_nodes:
            spectral_feat_proj = F.interpolate(spectral_feat_proj.transpose(1, 2), size=target_nodes, mode='linear', align_corners=False).transpose(1, 2)
        if temporal_feat_proj.size(1) != target_nodes:
            temporal_feat_proj = F.interpolate(temporal_feat_proj.transpose(1, 2), size=target_nodes, mode='linear', align_corners=False).transpose(1, 2)
        
        fused = spectral_feat_proj * temporal_feat_proj
        st_feat = self.st_proj(self.st_pool(self.st_gat(fused))).squeeze(-1)
        if st_feat.size(1) != 7: st_feat = F.adaptive_avg_pool1d(st_feat.unsqueeze(1), 7).squeeze(1)
        return torch.clamp(self.fc(st_feat), min=-10, max=10)


class AudioProcessor:
    def __init__(self, sample_rate: int = 16000, max_length_seconds: float = 4.0):
        self.sample_rate = sample_rate
        self.max_length_samples = int(sample_rate * max_length_seconds)

    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        waveform, sr = torchaudio.load(audio_path)
        if waveform.shape[0] != 1: waveform = torch.mean(waveform, dim=0, keepdim=True)
        if sr != self.sample_rate: waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform)
        return waveform.squeeze(0), self.sample_rate

    def process(self, audio: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        if length is None: length = self.max_length_samples
        if len(audio) > length:
            start_idx = (len(audio) - length) // 2
            audio = audio[start_idx:start_idx + length]
        else:
            audio = F.pad(audio, (0, length - len(audio)), mode='constant', value=0)
        return audio


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, reduction='mean'):
        super().__init__()
        self.alpha, self.gamma, self.reduction = alpha, gamma, reduction
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none')
        pt = torch.exp(-ce_loss)
        f_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return f_loss.mean() if self.reduction == 'mean' else f_loss.sum() if self.reduction == 'sum' else f_loss


class CombinedLoss(nn.Module):
    def __init__(self, class_weights=None, focal_alpha=0.25, focal_gamma=2.0, ce_weight=1.0, focal_weight=1.0):
        super().__init__()
        self.ce_weight, self.focal_weight = ce_weight, focal_weight
        self.register_buffer('class_weights', torch.tensor(class_weights) if class_weights else None)
        self.focal_loss = FocalLoss(alpha=focal_alpha, gamma=focal_gamma)
    def forward(self, inputs, targets):
        return self.ce_weight * F.cross_entropy(inputs, targets, weight=self.class_weights) + self.focal_weight * self.focal_loss(inputs, targets)


class MetricsCalculation:
    @staticmethod
    def compute_eer(labels, scores):
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        eer = 100 * (1 - interp1d(fpr, tpr)(brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)))
        return eer, None
    @staticmethod
    def compute_all_metrics(labels, scores):
        eer, _ = MetricsCalculation.compute_eer(labels, scores)
        preds = (scores > 0.5).astype(int)
        acc = 100 * (labels == preds).sum() / len(labels)
        return {'eer': eer, 'accuracy': acc}


class TrainRawGATST:
    def __init__(self, model, optimizer, criterion, device='cuda', checkpoint_dir='checkpoints', experiment_name='rawgatst'):
        self.model, self.optimizer, self.criterion, self.device = model.to(device), optimizer, criterion, device
        self.weights_dir = os.path.join(checkpoint_dir, experiment_name, 'weights')
        os.makedirs(self.weights_dir, exist_ok=True)
        self.best_eer = float('inf')
    def fit(self, train_loader, val_loader, epochs):
        scaler = torch.amp.GradScaler('cuda')
        for epoch in range(epochs):
            self.model.train()
            for data, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                data, labels = data.to(self.device), labels.to(self.device)
                with torch.amp.autocast('cuda'):
                    loss = self.criterion(self.model(data), labels)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer); scaler.update(); self.optimizer.zero_grad()
            self.model.eval()
            labels_all, scores_all = [], []
            with torch.no_grad():
                for data, labels in val_loader:
                    logits = self.model(data.to(self.device))
                    scores = F.softmax(logits, dim=1)[:, 1]
                    labels_all.append(labels.numpy()); scores_all.append(scores.cpu().numpy())
            metrics = MetricsCalculation.compute_all_metrics(np.concatenate(labels_all), np.concatenate(scores_all))
            print(f"Val EER: {metrics['eer']:.2f}%")
            if metrics['eer'] < self.best_eer:
                self.best_eer = metrics['eer']
                torch.save(self.model.state_dict(), os.path.join(self.weights_dir, 'best.pth'))


class RawGATSTInference:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device); self.model.eval(); self.device = device
        self.processor = AudioProcessor()
    def predict_file(self, audio_path):
        waveform, _ = self.processor.load_audio(audio_path)
        audio = self.processor.process(waveform).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return F.softmax(self.model(audio), dim=1)[0, 1].item()


def run_training():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    model = RawGAT_ST()
    print(f"RawGAT-ST initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

if __name__ == "__main__":
    run_training()
