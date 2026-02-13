import os
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from typing import Tuple, Optional, Dict, List
from torch.optim import Adam, AdamW
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

# Optional dependencies with graceful fallbacks
try:
    import torchaudio
except (ImportError, OSError):
    torchaudio = None

try:
    from transformers import Wav2Vec2Model, Wav2Vec2Config
except ImportError:
    Wav2Vec2Model = None
    Wav2Vec2Config = None

# Fallback for t-DCF if external script is missing
try:
    from calculate_tDCF import load_asv_metrics, compute_min_tDCF
except ImportError:
    def load_asv_metrics(*args, **kwargs): return {}
    def compute_min_tDCF(*args, **kwargs): return 1.0


class Wav2Vec2Frontend(nn.Module):
    """
    Wav2Vec2-based feature extractor for detection tasks.
    """
    def __init__(self, model_name='facebook/wav2vec2-xls-r-300m', output_dim=768, freeze_encoder=False, use_conv_projection=True):
        super(Wav2Vec2Frontend, self).__init__()
        
        self.use_conv_projection = use_conv_projection
        self.output_dim = output_dim

        if Wav2Vec2Model is None:
            print("Warning: Wav2Vec2Model not found. Transformers library is required for Wav2Vec2Frontend.")
            self.wav2vec2 = None
            return

        self.wav2vec2 = Wav2Vec2Model.from_pretrained(model_name)
        self.hidden_dim = self.wav2vec2.config.hidden_size

        if freeze_encoder:
            for param in self.wav2vec2.parameters():
                param.requires_grad = False

        if use_conv_projection:
            self.projection = nn.Conv1d(self.hidden_dim, output_dim, kernel_size=1, bias=True)
        else:
            self.projection = nn.Linear(self.hidden_dim, output_dim)

    def forward(self, x):
        # x shape: [batch, samples] or [batch, 1, samples]
        if x.dim() == 3:
            x = x.squeeze(1)

        outputs = self.wav2vec2(x, output_hidden_states=False)
        features = outputs.last_hidden_state # [batch, time, hidden_dim]

        if self.use_conv_projection:
            features = features.transpose(1, 2) # [batch, hidden_dim, time]
            features = self.projection(features)
        else:
            features = self.projection(features)
            features = features.transpose(1, 2)
            
        return features # [batch, output_dim, time]


class Wav2Vec2CM(nn.Module):
    """
    Complete Countermeasure model using Wav2Vec2 as a backbone.
    """
    def __init__(self, frontend: Wav2Vec2Frontend, num_classes: int = 2):
        super().__init__()
        self.frontend = frontend
        self.avgpool = nn.AdaptiveAvgPool1d(1)
        self.fc = nn.Linear(frontend.output_dim, num_classes)

    def forward(self, x):
        feat = self.frontend(x) # [B, D, T]
        feat = self.avgpool(feat).squeeze(-1) # [B, D]
        return self.fc(feat)


class AudioProcessor:
    def __init__(self, sample_rate: int = 16000, max_length_seconds: float = 4.0):
        self.sample_rate = sample_rate
        self.max_length_samples = int(sample_rate * max_length_seconds)

    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        if torchaudio is None:
            import soundfile as sf
            waveform, sr = sf.read(audio_path)
            waveform = torch.from_numpy(waveform).float()
            if waveform.dim() > 1: waveform = waveform.mean(dim=1)
        else:
            waveform, sr = torchaudio.load(audio_path)
            if waveform.shape[0] > 1: waveform = torch.mean(waveform, dim=0)
            else: waveform = waveform.squeeze(0)
            
        if sr != self.sample_rate:
            if torchaudio:
                waveform = torchaudio.transforms.Resample(sr, self.sample_rate)(waveform.unsqueeze(0)).squeeze(0)
            else:
                import librosa
                waveform = torch.from_numpy(librosa.resample(waveform.numpy(), orig_sr=sr, target_sr=self.sample_rate)).float()
        
        return waveform, self.sample_rate

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
        return 100 * (1 - interp1d(fpr, tpr)(brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.))), None
    @staticmethod
    def compute_all_metrics(labels, scores):
        eer, _ = MetricsCalculation.compute_eer(labels, scores)
        return {'eer': eer, 'accuracy': 100 * (labels == (scores > 0.5).astype(int)).sum() / len(labels)}


class TrainWav2Vec2:
    def __init__(self, model, optimizer, criterion, device='cuda', checkpoint_dir='checkpoints', experiment_name='wav2vec2_cm'):
        self.model, self.optimizer, self.criterion, self.device = model.to(device), optimizer, criterion, device
        self.weights_dir = Path(checkpoint_dir) / experiment_name / 'weights'
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.best_eer = float('inf')

    def fit(self, train_loader, val_loader, epochs):
        scaler = torch.amp.GradScaler('cuda')
        for epoch in range(epochs):
            self.model.train()
            for inputs, labels in tqdm(train_loader, desc=f"Epoch {epoch+1}"):
                inputs, labels = inputs.to(self.device), labels.to(self.device)
                with torch.amp.autocast('cuda'):
                    loss = self.criterion(self.model(inputs), labels)
                scaler.scale(loss).backward()
                scaler.step(self.optimizer); scaler.update(); self.optimizer.zero_grad()
            self.model.eval()
            scores, labels_val = [], []
            with torch.no_grad():
                for inputs, labels in val_loader:
                    scores.extend(torch.softmax(self.model(inputs.to(self.device)), dim=1)[:, 1].cpu().numpy())
                    labels_val.extend(labels.numpy())
            metrics = MetricsCalculation.compute_all_metrics(np.array(labels_val), np.array(scores))
            print(f"Val EER: {metrics['eer']:.2f}%")
            if metrics['eer'] < self.best_eer:
                self.best_eer = metrics['eer']
                torch.save(self.model.state_dict(), self.weights_dir / 'best.pth')


class Wav2Vec2Inference:
    def __init__(self, model, device='cuda'):
        self.model = model.to(device); self.model.eval(); self.device = device
        self.processor = AudioProcessor()
    def predict_file(self, audio_path):
        waveform, _ = self.processor.load_audio(audio_path)
        audio = self.processor.process(waveform).unsqueeze(0).to(self.device)
        with torch.no_grad():
            return torch.softmax(self.model(audio), dim=1)[0, 1].item()


def run_training():
    import argparse
    parser = argparse.ArgumentParser(description="Wav2Vec2 Training CLI")
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--lr", type=float, default=1e-5, help="Smaller LR recommended for Wav2Vec2 fine-tuning")
    parser.add_argument("--model_name", type=str, default='facebook/wav2vec2-xls-r-300m')
    args = parser.parse_args()
    
    frontend = Wav2Vec2Frontend(model_name=args.model_name)
    model = Wav2Vec2CM(frontend)
    print(f"Wav2Vec2 Toolkit initialized with {sum(p.numel() for p in model.parameters()):,} parameters.")

if __name__ == "__main__":
    run_training()
