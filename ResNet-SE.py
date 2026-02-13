import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import scipy.signal
import scipy.fftpack
import soundfile as sf
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Tuple, Optional, Dict
from sklearn.metrics import roc_curve, auc
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from scipy.ndimage import zoom

try:
    import torchaudio
except (ImportError, OSError):
    torchaudio = None

try:
    from calculate_tDCF import load_asv_metrics, compute_min_tDCF
except ImportError:
    def load_asv_metrics(*args, **kwargs): return {}
    def compute_min_tDCF(*args, **kwargs): return 1.0


class SEBlock(nn.Module):
    def __init__(self, channels, reduction=16):
        super(SEBlock, self).__init__()
        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()
        y = self.squeeze(x).view(batch, channels)
        y = self.excitation(y).view(batch, channels, 1, 1)
        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.se = SEBlock(planes, reduction)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class SEBottleneck(nn.Module):
    expansion = 4
    def __init__(self, in_planes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(planes * self.expansion, reduction)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out = self.se(out)
        if self.downsample is not None:
            identity = self.downsample(x)
        out += identity
        return self.relu(out)


class TemporalAttention(nn.Module):
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.temporal_squeeze = nn.AdaptiveAvgPool2d((None, 1))
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, kernel_size=15, padding=7),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, kernel_size=15, padding=7),
            nn.Sigmoid()
        )
    def forward(self, x):
        y = self.temporal_squeeze(x).squeeze(-1)
        att = self.temporal_conv(y).unsqueeze(-1)
        return x * att.expand_as(x)


class SpectralAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_squeeze = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    def forward(self, x):
        freq_att = self.conv_squeeze(x.mean(dim=2, keepdim=True))
        return x * freq_att


class ResNetSE(nn.Module):
    def __init__(self, block, layers, num_classes=2, in_channels=4, reduction=16, base_channels=64):
        super(ResNetSE, self).__init__()
        self.inplanes = base_channels
        self.reduction = reduction
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        self.temporal_attn = TemporalAttention(base_channels, reduction=4)
        self.spectral_attn = SpectralAttention(base_channels)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, base_channels, layers[0])
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)
        self.fc = nn.Linear(base_channels * 8 * block.expansion, num_classes)
        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = [block(self.inplanes, planes, stride, downsample, self.reduction)]
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, reduction=self.reduction))
        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                if m.bias is not None: nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None: nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.spectral_attn(self.temporal_attn(x))
        x = self.maxpool(x)
        x = self.layer4(self.layer3(self.layer2(self.layer1(x))))
        x = self.fc(self.dropout(torch.flatten(self.avgpool(x), 1)))
        return x


def resnet18_se(num_classes=2, **kwargs): return ResNetSE(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)
def resnet34_se(num_classes=2, **kwargs): return ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)


class AudioProcessor:
    def __init__(self, sample_rate: int = 16000, max_len: int = 400, n_ceps: int = 70):
        self.sample_rate, self.max_len, self.n_ceps = sample_rate, max_len, n_ceps

    def delta(self, feat, N=2):
        denom = 2 * sum([i**2 for i in range(1, N + 1)])
        padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')
        d_feat = np.zeros_like(feat)
        for t in range(feat.shape[0]):
            for n in range(1, N + 1):
                d_feat[t] += n * (padded[t + N + n] - padded[t + N - n])
        return d_feat / denom

    def extract_lfcc(self, signal):
        signal = scipy.signal.lfilter([1, -0.97], 1, signal)
        f_len, f_step = int(0.025 * self.sample_rate), int(0.01 * self.sample_rate)
        num_frames = max(1, 1 + int((len(signal) - f_len) / f_step))
        pad_len = (num_frames - 1) * f_step + f_len
        if pad_len > len(signal): signal = np.pad(signal, (0, pad_len - len(signal)), mode='constant')
        mag_frames = np.absolute(np.fft.rfft(np.array([signal[i*f_step:i*f_step+f_len] * np.hamming(f_len) for i in range(num_frames)]), n=512))
        filterbank = np.zeros((self.n_ceps, 257))
        bins = np.floor(513 * np.linspace(0, self.sample_rate/2, self.n_ceps + 2) / self.sample_rate).astype(int)
        for m in range(1, self.n_ceps + 1):
            filterbank[m-1, bins[m-1]:bins[m]] = (np.arange(bins[m-1], bins[m]) - bins[m-1]) / (bins[m] - bins[m-1] + 1e-8)
            filterbank[m-1, bins[m]:bins[m+1]] = (bins[m+1] - np.arange(bins[m], bins[m+1])) / (bins[m+1] - bins[m] + 1e-8)
        static = scipy.fftpack.dct(np.log(np.dot(mag_frames**2 / 512, filterbank.T) + 1e-8), type=2, axis=1, norm='ortho')[:, :self.n_ceps]
        d1, d2 = self.delta(static), self.delta(self.delta(static))
        return np.stack([static, d1, d2], axis=0)

    def extract_phase_derivative(self, signal):
        _, _, Zxx = scipy.signal.stft(signal, fs=self.sample_rate, nperseg=int(0.025*self.sample_rate), noverlap=int(0.015*self.sample_rate))
        phase_delta = np.unwrap(np.diff(np.angle(Zxx), axis=1, prepend=np.angle(Zxx)[:, :1]), axis=1)
        return (phase_delta - phase_delta.mean()) / (phase_delta.std() + 1e-8)

    def process(self, signal):
        lfcc = self.extract_lfcc(signal)
        phase = self.extract_phase_derivative(signal).T
        if phase.shape[0] != lfcc.shape[1]:
            phase = interp1d(np.linspace(0, 1, phase.shape[0]), phase, axis=0, kind='linear')(np.linspace(0, 1, lfcc.shape[1]))
        if phase.shape[1] != self.n_ceps:
            phase = zoom(phase, (1, self.n_ceps / phase.shape[1]), order=1)
        phase = (phase - phase.mean(axis=0)) / (phase.std(axis=0) + 1e-8)
        feat = np.concatenate([lfcc, phase[np.newaxis, :, :]], axis=0)
        return feat[:, :self.max_len, :] if feat.shape[1] >= self.max_len else np.pad(feat, ((0,0),(0,self.max_len - feat.shape[1]),(0,0)))


class MetricsCalculation:
    @staticmethod
    def compute_eer(labels, scores):
        fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
        return 100 * (1 - interp1d(fpr, tpr)(brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.))), None
    @staticmethod
    def compute_all_metrics(labels, scores):
        eer, _ = MetricsCalculation.compute_eer(labels, scores)
        return {'eer': eer, 'accuracy': 100 * (labels == (scores > 0.5).astype(int)).sum() / len(labels)}


class TrainResNetSE:
    def __init__(self, model, optimizer, criterion, device='cuda', checkpoint_dir='checkpoints', experiment_name='resnet_se'):
        self.model, self.optimizer, self.criterion, self.device = model.to(device), optimizer, criterion, device
        self.weights_dir = Path(checkpoint_dir) / experiment_name / 'weights'
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.best_eer = float('inf')

    def fit(self, train_loader, val_loader, epochs):
        scaler = torch.amp.GradScaler('cuda')
        for epoch in range(epochs):
            self.model.train()
            for inputs, labels, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with torch.amp.autocast('cuda'):
                    loss = self.criterion(self.model(inputs), labels)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer); scaler.update(); self.optimizer.zero_grad()
            self.model.eval()
            scores, labels_val = [], []
            with torch.no_grad():
                for inputs, labels, _ in val_loader:
                    scores.extend(torch.softmax(self.model(inputs.to(self.device)), dim=1)[:, 1].cpu().numpy())
                    labels_val.extend(labels.numpy())
            metrics = MetricsCalculation.compute_all_metrics(np.array(labels_val), np.array(scores))
            print(f"Val EER: {metrics['eer']:.2f}%")
            if metrics['eer'] < self.best_eer:
                self.best_eer = metrics['eer']
                torch.save(self.model.state_dict(), self.weights_dir / 'best.pth')

class ResNetSEInference:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device); self.model.eval(); self.device = device
        self.processor = AudioProcessor()
    def predict_file(self, audio_path):
        signal, _ = sf.read(audio_path)
        feat = torch.from_numpy(self.processor.process(signal)).unsqueeze(0).to(self.device).float()
        with torch.no_grad(): return torch.softmax(self.model(feat), dim=1)[0, 1].item()

def run_training():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    model = resnet18_se()
    print(f"ResNet-SE initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

if __name__ == "__main__":
    run_training()
