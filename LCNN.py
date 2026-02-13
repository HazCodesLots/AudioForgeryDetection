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

class MaxFeatureMap2D(nn.Module):
    """
    Max-Feature-Map activation
    Key component of LCNN that reduces feature maps by taking max over pairs
    """
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        shape = list(x.size())
        batch_size = shape[0]
        channels = shape[1]

        new_shape = [batch_size, channels // 2, 2] + shape[2:]
        x = x.view(*new_shape)
        x, _ = torch.max(x, dim=2)
        return x


class MFMConv2d(nn.Module):
    """Convolutional layer with Max-Feature-Map activation"""
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0):
        super().__init__()
        self.conv = nn.Conv2d(
            in_channels, 
            out_channels * 2,
            kernel_size=kernel_size,
            stride=stride,
            padding=padding,
            bias=False
        )
        self.bn = nn.BatchNorm2d(out_channels * 2)
        self.mfm = MaxFeatureMap2D()
        
    def forward(self, x):
        x = self.conv(x)
        x = self.bn(x)
        x = self.mfm(x)
        return x


class LCNN(nn.Module):
    """
    Light CNN for audio deepfake detection
    Optimized for LFCC features
    """
    def __init__(self, n_lfcc=60, num_classes=2):
        super().__init__()
        
        self.conv1 = MFMConv2d(1, 48, kernel_size=5, stride=1, padding=2)
        self.pool1 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv2a = MFMConv2d(48, 48, kernel_size=1, stride=1, padding=0)
        self.conv2 = MFMConv2d(48, 96, kernel_size=3, stride=1, padding=1)
        self.pool2 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv3a = MFMConv2d(96, 96, kernel_size=1, stride=1, padding=0)
        self.conv3 = MFMConv2d(96, 128, kernel_size=3, stride=1, padding=1)
        self.pool3 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv4a = MFMConv2d(128, 128, kernel_size=1, stride=1, padding=0)
        self.conv4 = MFMConv2d(128, 192, kernel_size=3, stride=1, padding=1)
        self.pool4 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.conv5a = MFMConv2d(192, 192, kernel_size=1, stride=1, padding=0)
        self.conv5 = MFMConv2d(192, 256, kernel_size=3, stride=1, padding=1)
        self.pool5 = nn.MaxPool2d(kernel_size=2, stride=2)
        
        self.global_pool = nn.AdaptiveAvgPool2d((1, 1))
        
        self.fc1 = nn.Linear(256, 320)
        self.mfm_fc1 = MaxFeatureMap2D()
        self.dropout1 = nn.Dropout(0.7)
        
        self.fc2 = nn.Linear(160, num_classes)
        
    def forward(self, x):
        if x.dim() == 3:
            x = x.unsqueeze(1)
            
        x = self.conv1(x)
        x = self.pool1(x)
        
        x = self.conv2a(x)
        x = self.conv2(x)
        x = self.pool2(x)
        
        x = self.conv3a(x)
        x = self.conv3(x)
        x = self.pool3(x)
        
        x = self.conv4a(x)
        x = self.conv4(x)
        x = self.pool4(x)
        
        x = self.conv5a(x)
        x = self.conv5(x)
        x = self.pool5(x)
        
        x = self.global_pool(x)
        x = x.view(x.size(0), -1)
        
        x = self.fc1(x)
        x = self.mfm_fc1(x)
        x = x.view(x.size(0), -1)
        x = self.dropout1(x)
        
        x = self.fc2(x)
        return x


class AudioProcessor:
    def __init__(self, sample_rate: int = 16000, max_length_seconds: float = 4.0):
        self.sample_rate = sample_rate
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
    def __init__(self, class_weights=None, focal_alpha=0.25, focal_gamma=2.0, ce_weight=1.0, focal_weight=1.0):
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
        return self.ce_weight * ce_loss + self.focal_weight * focal_loss


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
        return np.min(dcf)
    
    @staticmethod
    def compute_all_metrics(labels, scores, predictions=None):
        if predictions is None:
            predictions = (scores > 0.5).astype(int)
        eer, eer_threshold = MetricsCalculation.compute_eer(labels, scores)
        min_dcf = MetricsCalculation.compute_min_dcf(labels, scores)
        accuracy = 100 * (labels == predictions).sum() / len(labels)
        
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


class TrainLCNN:
    def __init__(self, model, optimizer, criterion, device='cuda', scheduler=None, 
                 checkpoint_dir='checkpoints', experiment_name='lcnn', accumulation_steps=1, use_amp=True):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler
        self.accumulation_steps = accumulation_steps
        self.use_amp = use_amp
        self.scaler = torch.amp.GradScaler('cuda', enabled=use_amp)
        self.checkpoint_dir = os.path.join(checkpoint_dir, experiment_name)
        self.weights_dir = os.path.join(self.checkpoint_dir, 'weights')
        os.makedirs(self.weights_dir, exist_ok=True)
        self.best_eer = float('inf')
        self.history = {'train_loss': [], 'val_eer': []}

    def train_epoch(self, train_loader, epoch):
        self.model.train()
        total_loss = 0
        pbar = tqdm(train_loader, desc=f'Epoch {epoch+1} [Train]')
        self.optimizer.zero_grad()
        
        for batch_idx, (data, labels) in enumerate(pbar):
            data, labels = data.to(self.device), labels.to(self.device)
            with torch.amp.autocast('cuda', enabled=self.use_amp):
                logits = self.model(data)
                loss = self.criterion(logits, labels) / self.accumulation_steps
            
            self.scaler.scale(loss).backward()
            if (batch_idx + 1) % self.accumulation_steps == 0:
                self.scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 0.5)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad()
            
            total_loss += loss.item() * self.accumulation_steps
            pbar.set_postfix({'loss': f'{loss.item()*self.accumulation_steps:.4f}'})
        return total_loss / len(train_loader)

    def validate(self, val_loader, epoch):
        self.model.eval()
        all_labels, all_scores = [], []
        with torch.no_grad():
            for data, labels in tqdm(val_loader, desc=f'Epoch {epoch+1} [Val]'):
                data = data.to(self.device)
                logits = self.model(data)
                scores = F.softmax(logits, dim=1)[:, 1]
                all_labels.append(labels.numpy())
                all_scores.append(scores.cpu().numpy())
        
        all_labels = np.concatenate(all_labels)
        all_scores = np.concatenate(all_scores)
        return MetricsCalculation.compute_all_metrics(all_labels, all_scores)

    def fit(self, train_loader, val_loader, num_epochs):
        for epoch in range(num_epochs):
            train_loss = self.train_epoch(train_loader, epoch)
            metrics = self.validate(val_loader, epoch)
            print(f"Epoch {epoch+1} - Train Loss: {train_loss:.4f}, Val EER: {metrics['eer']:.2f}%")
            
            if metrics['eer'] < self.best_eer:
                self.best_eer = metrics['eer']
                torch.save(self.model.state_dict(), os.path.join(self.weights_dir, 'best.pth'))
            
            if self.scheduler:
                self.scheduler.step(metrics['eer'] if isinstance(self.scheduler, ReduceLROnPlateau) else None)


class LCNNInference:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device)
        self.model.eval()
        self.device = device
    
    def predict_features(self, features):
        """Predict from pre-extracted features (LFCC/Mel)"""
        if not isinstance(features, torch.Tensor):
            features = torch.from_numpy(features)
        if features.dim() == 2:
            features = features.unsqueeze(0)
        features = features.unsqueeze(1).to(self.device).float()
        with torch.no_grad():
            logits = self.model(features)
            probs = F.softmax(logits, dim=1)
            return probs[0, 1].item()


def run_training():
    import argparse
    parser = argparse.ArgumentParser(description="LCNN Training")
    parser.add_argument("--experiment_name", type=str, default="lcnn_v1")
    parser.add_argument("--checkpoint_dir", type=str, default="checkpoints")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--lr", type=float, default=1e-4)
    args = parser.parse_args()
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = LCNN()
    optimizer = AdamW(model.parameters(), lr=args.lr)
    criterion = CombinedLoss()
    
    print(f"LCNN initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

if __name__ == "__main__":
    run_training()
