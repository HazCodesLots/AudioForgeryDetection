import os
import json
import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import librosa
from sklearn.metrics import f1_score, average_precision_score, roc_auc_score
from tqdm import tqdm
import argparse

# =============================================================================
# ConvNeXt-SplitBand ASL
# =============================================================================
# Changes vs V1:
#   1. Multi-label Asymmetric Loss (ASL) replaces BCEWithLogitsLoss
#      - gamma_neg=4 suppresses easy negatives aggressively
#      - clamp(sigmoid(logit) - margin, min=0) avoids log(negative)
#   2. Rare-class Mixup — targeted λ-blend for classes with prevalence < 5%
#      - SpecAugment runs PER-SAMPLE before mixing (correct order)
#      - labels fused with max() not soft blend (cleaner with ASL)
#   3. LSE pooling per frequency band replaces hard GMP
#      - learnable β (init=5, clamped ≥ 0.5) lets the model tune softness
#   4. 256 mel bins, fmax=16kHz, n_fft=2048 (double V1 frequency resolution)
# =============================================================================


# ---------------------------------------------------------------------------
# 1. Asymmetric Loss (Multi-label)
# ---------------------------------------------------------------------------

class AsymmetricLoss(nn.Module):
    """
    Multi-label Asymmetric Loss (ASL).
    Zamir et al. "Asymmetric Loss For Multi-Label Classification" (ICCV 2021).

    For positives (y=1): standard focal with gamma_pos (default 0 — no down-weighting,
    because rare classes need every positive signal).
    For negatives (y=0): focal with gamma_neg (default 4) to suppress easy negatives.
    Probability shifting: shift negative probabilities down by `margin` and clamp at 0
    before computing log to avoid log(negative).

    Args:
        gamma_neg   (float): Focusing power for negatives. Default 4.
        gamma_pos   (float): Focusing power for positives. Default 0.
        margin      (float): Probability shift for negatives. Default 0.05.
        reduction   (str):   'mean' or 'sum'.
    """
    def __init__(self, gamma_neg=4, gamma_pos=0, margin=0.05, reduction='mean'):
        super().__init__()
        self.gamma_neg = gamma_neg
        self.gamma_pos = gamma_pos
        self.margin    = margin
        self.reduction = reduction

    def forward(self, logits, targets):
        # Sigmoid probabilities
        p = torch.sigmoid(logits)                   # (B, C)

        # Positive branch: standard focal
        p_pos = p
        loss_pos = targets * (torch.pow(1 - p_pos, self.gamma_pos)
                               * torch.log(p_pos.clamp(min=1e-8)))

        # Negative branch: shifted + clamped probability
        p_neg = (p - self.margin).clamp(min=0)      # shift then clamp → safe log
        loss_neg = (1 - targets) * (torch.pow(p_neg, self.gamma_neg)
                                     * torch.log((1 - p_neg).clamp(min=1e-8)))

        loss = -(loss_pos + loss_neg)
        return loss.mean() if self.reduction == 'mean' else loss.sum()


# ---------------------------------------------------------------------------
# 2. Backbone: ConvNeXt2D (unchanged from V1)
# ---------------------------------------------------------------------------

class DropPath(nn.Module):
    def __init__(self, drop_prob=0.):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        if self.drop_prob == 0. or not self.training:
            return x
        keep_prob = 1 - self.drop_prob
        shape = (x.shape[0],) + (1,) * (x.ndim - 1)
        random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
        random_tensor.floor_()
        return x.div(keep_prob) * random_tensor


class SEBlock(nn.Module):
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(dim, max(dim // reduction, 4))
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(max(dim // reduction, 4), dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=(2, 3))
        y = self.act(self.fc1(y))
        y = self.sigmoid(self.fc2(y)).unsqueeze(-1).unsqueeze(-1)
        return x * y


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-5):
        super().__init__()
        self.dwconv  = nn.Conv2d(dim, dim, kernel_size=9, padding=4, groups=dim)
        self.norm    = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act     = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma   = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.se      = SEBlock(dim)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.act(self.pwconv1(x))
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = self.se(x)
        return shortcut + self.drop_path(x)


class ConvNeXt2D(nn.Module):
    def __init__(self, input_channels=1, depths=[2, 2, 6, 2], dims=[64, 128, 256, 512],
                 drop_path_rate=0.2, layer_scale_init_value=1e-5):
        super().__init__()
        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=2, stride=2),
            nn.LayerNorm(dims[0])
        )
        self.stages     = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(depths)):
            self.stages.append(nn.Sequential(
                *[ConvNeXtBlock(dims[i], dp_rates[cur + j], layer_scale_init_value)
                  for j in range(depths[i])]
            ))
            cur += depths[i]
            if i < len(depths) - 1:
                self.downsamples.append(nn.Sequential(
                    nn.LayerNorm(dims[i], eps=1e-6),
                    nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
                ))
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.stem[0](x)
        x = x.permute(0, 2, 3, 1)
        x = self.stem[1](x)
        x = x.permute(0, 3, 1, 2)
        for i in range(len(self.stages)):
            x = self.stages[i](x)
            if i < len(self.downsamples):
                x = x.permute(0, 2, 3, 1)
                x = self.downsamples[i][0](x)
                x = x.permute(0, 3, 1, 2)
                x = self.downsamples[i][1](x)
        return x


# ---------------------------------------------------------------------------
# 3. LSE Pooling + SplitBandPoolingV2
# ---------------------------------------------------------------------------

class LSEPool2d(nn.Module):
    """
    Learnable-β LogSumExp pooling over H×W spatial dimensions.

    LSE(x; β) = (1/β) · log[ Σ exp(β·x_i) ]

    As β → ∞ this approaches hard-max (GMP).
    As β → 0 this approaches mean (GAP).
    β is a learnable parameter, clamped to [0.5, ∞) so it never collapses to mean.

    Args:
        init_beta (float): Initial β. 5.0 starts close to hard-max.
    """
    def __init__(self, init_beta=5.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))

    def forward(self, x):
        B, C = x.shape[:2]
        beta = self.beta.clamp(min=0.5)
        flat = x.view(B, C, -1)                           # (B, C, H*W)
        return (1.0 / beta) * torch.logsumexp(beta * flat, dim=-1)  # (B, C)


class SplitBandPoolingV2(nn.Module):
    """
    Frequency split + dual (GAP + LSE) pooling per band.

    Given backbone output (B, C, F, T):
      low  = x[:, :, :F//2, :]   →  [0  – ~4kHz]
      high = x[:, :, F//2:, :]   →  [~4kHz – 16kHz]

    For each half:
      GAP → (B, C)   — captures sustained energy
      LSE → (B, C)   — learnable soft-max, captures transient peaks

    Fused: cat([gap_low, lse_low, gap_high, lse_high]) → (B, 4C)
    Projected: Linear(4C → output_dim) + LayerNorm + GELU
    """
    def __init__(self, input_dim=512, output_dim=512, dropout=0.1, lse_beta=5.0):
        super().__init__()
        self.gap     = nn.AdaptiveAvgPool2d(1)
        self.lse_low  = LSEPool2d(init_beta=lse_beta)
        self.lse_high = LSEPool2d(init_beta=lse_beta)   # separate β per band
        self.proj = nn.Sequential(
            nn.Linear(input_dim * 4, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        F_dim = x.shape[2]
        mid   = F_dim // 2
        low   = x[:, :, :mid, :]
        high  = x[:, :, mid:, :]

        gap_low  = self.gap(low).view(x.size(0), -1)
        lse_low  = self.lse_low(low)
        gap_high = self.gap(high).view(x.size(0), -1)
        lse_high = self.lse_high(high)

        fused = torch.cat([gap_low, lse_low, gap_high, lse_high], dim=1)
        return self.proj(fused)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=23, dropout_rate=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(dropout_rate / 2),
            nn.Linear(128, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class ConvNeXtTagger(nn.Module):
    def __init__(self, convnext_params, pooling_params, mlp_params):
        super().__init__()
        self.convnext = ConvNeXt2D(**convnext_params)
        self.pool     = SplitBandPoolingV2(**pooling_params)
        self.mlp      = MLPClassifier(**mlp_params)

    def forward(self, x):
        return self.mlp(self.pool(self.convnext(x)))


# ---------------------------------------------------------------------------
# 4. Audio Frontend (256 mels, 16kHz fmax)
# ---------------------------------------------------------------------------

class MelSpectrogramTransform:
    def __init__(self, sample_rate=32000, n_fft=2048, hop_length=256,
                 n_mels=256, fmin=50, fmax=16000, eps=1e-6):
        self.sample_rate = sample_rate
        self.n_fft       = n_fft
        self.hop_length  = hop_length
        self.n_mels      = n_mels
        self.fmin        = fmin
        self.fmax        = fmax
        self.eps         = eps

    def __call__(self, waveform):
        y = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform
        if y.ndim > 1:
            y = y.squeeze()
        mel = librosa.feature.melspectrogram(
            y=y, sr=self.sample_rate, n_fft=self.n_fft, hop_length=self.hop_length,
            n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        norm    = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + self.eps)
        return torch.tensor(norm).unsqueeze(0).float()   # (1, n_mels, T)


class SpecAugment(nn.Module):
    """Applied PER-SAMPLE before Mixup."""
    def __init__(self, time_mask_param=30, freq_mask_param=20):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

    def forward(self, x):
        # x: (1, F, T)
        _, F_dim, T_dim = x.shape
        f  = int(np.random.uniform(0, self.freq_mask_param))
        f0 = int(np.random.uniform(0, max(F_dim - f, 1)))
        x[:, f0:f0+f, :] = 0
        t  = int(np.random.uniform(0, self.time_mask_param))
        t0 = int(np.random.uniform(0, max(T_dim - t, 1)))
        x[:, :, t0:t0+t] = 0
        return x


class EnhancedAudioFrontend(nn.Module):
    def __init__(self, n_mels=256):
        super().__init__()
        self.mel_transform = MelSpectrogramTransform(sample_rate=32000, n_mels=n_mels)
        self.spec_augment  = SpecAugment()

    def forward(self, waveform, training=True):
        mel = self.mel_transform(waveform)
        if training:
            mel = self.spec_augment(mel)   # aug before mixup
        return mel


# ---------------------------------------------------------------------------
# 5. Dataset with Rare-Class Index
# ---------------------------------------------------------------------------

LABEL_COLS = [
    '1-1_small-sounding-engine_presence', '1-2_medium-sounding-engine_presence',
    '1-3_large-sounding-engine_presence', '2-1_rock-drill_presence',
    '2-2_jackhammer_presence', '2-3_hoe-ram_presence', '2-4_pile-driver_presence',
    '3-1_non-machinery-impact_presence', '4-1_chainsaw_presence',
    '4-2_small-medium-rotating-saw_presence', '4-3_large-rotating-saw_presence',
    '5-1_car-horn_presence', '5-2_car-alarm_presence', '5-3_siren_presence',
    '5-4_reverse-beeper_presence', '6-1_stationary-music_presence',
    '6-2_mobile-music_presence', '6-3_ice-cream-truck_presence',
    '7-1_person-or-small-group-talking_presence', '7-2_person-or-small-group-shouting_presence',
    '7-3_large-crowd_presence', '7-4_amplified-speech_presence',
    '8-1_dog-barking-whining_presence'
]

RARE_PREVALENCE_THRESHOLD = 0.05   # classes with < 5% prevalence are "rare"


class SONYCUSTDataset(Dataset):
    def __init__(self, csv_path, audio_dir, split='train', frontend=None,
                 limit_samples=None):
        self.label_cols = LABEL_COLS
        df_raw   = pd.read_csv(csv_path)
        df_split = df_raw[df_raw['split'] == split].copy()
        df_split[self.label_cols] = df_split[self.label_cols].replace(-1, 0)
        row_sums = df_split[self.label_cols].sum(axis=1)
        df_split = df_split[row_sums <= 10]
        print(f"Aggregating labels for {split} split (Sanitized)...")
        self.df = df_split.groupby('audio_filename')[self.label_cols].max().reset_index()
        if limit_samples:
            self.df = self.df.head(limit_samples)

        self.audio_dir = audio_dir
        self.frontend  = frontend
        self.split     = split

        print(f"Building file index for {audio_dir}...")
        self.file_map = {}
        for root, _, files in os.walk(audio_dir):
            for f in files:
                if f.endswith('.wav'):
                    self.file_map[f] = os.path.join(root, f)
        print(f"Index built with {len(self.file_map)} unique audio files.")

        # Build rare-class index (used by RareClassMixup collate)
        labels      = self.df[self.label_cols].values
        n_samples   = len(labels)
        prevalences = labels.mean(axis=0)
        self.rare_class_ids = np.where(prevalences < RARE_PREVALENCE_THRESHOLD)[0]
        # Map: class_idx → list of dataset indices that have that class
        self.rare_class_index = {}
        for c in self.rare_class_ids:
            idxs = np.where(labels[:, c] > 0)[0].tolist()
            if idxs:
                self.rare_class_index[c] = idxs
        print(f"Rare classes (<{RARE_PREVALENCE_THRESHOLD:.0%}): "
              f"{len(self.rare_class_ids)} / {len(self.label_cols)}")

    def get_pos_weights(self):
        """Not used with ASL, kept for inspection/debugging."""
        labels      = self.df[self.label_cols].values.astype(np.float32)
        pos_counts  = np.where(labels > 0, 1.0, 0.0).sum(axis=0)
        neg_counts  = len(labels) - pos_counts
        pw          = neg_counts / (pos_counts + 1e-6)
        return torch.tensor(np.clip(pw, None, 10.0), dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row        = self.df.iloc[idx]
        audio_name = row['audio_filename']
        if audio_name not in self.file_map:
            raise FileNotFoundError(f"Audio file {audio_name} not found.")
        # Load at 32kHz to support 16kHz fmax without aliasing or empty bins
        wav, _  = librosa.load(self.file_map[audio_name], sr=32000, duration=10.0)
        wav     = torch.from_numpy(wav).float()
        mel     = self.frontend(wav, training=(self.split == 'train')) if self.frontend else wav
        labels  = np.where(row[self.label_cols].values.astype(np.float32) > 0, 1.0, 0.0)
        return mel, torch.tensor(labels, dtype=torch.float32), audio_name


# ---------------------------------------------------------------------------
# 6. Rare-Class Mixup Collate
# ---------------------------------------------------------------------------

def rare_class_mixup_collate(batch, dataset, max_length=626, p_mixup=0.5,
                              alpha=0.4):
    """
    Collate function that:
      1. Pads/truncates each mel to max_length.
      2. With probability p_mixup, mixes each sample with a randomly chosen
         rare-class sample using λ ~ Beta(alpha, alpha).
         Labels: max(label_a, label_b)  — hard union, safe with ASL.
      SpecAugment has already been applied per-sample inside the dataset,
      so we never corrupt the rare-class signal added by mixing.
    """
    mels, labels, names = [], [], []
    for mel, label, name in batch:
        # Pad / truncate time dim
        t = mel.shape[-1]
        if t > max_length:
            mel = mel[..., :max_length]
        elif t < max_length:
            mel = F.pad(mel, (0, max_length - t))
        mels.append(mel)
        labels.append(label)
        names.append(name)

    mels_t   = torch.stack(mels)
    labels_t = torch.stack(labels)

    if dataset.split != 'train' or not dataset.rare_class_index:
        return mels_t, labels_t, names

    # Mixup step
    B = len(mels_t)
    for i in range(B):
        if np.random.random() > p_mixup:
            continue
        # Pick a random rare class, then a random sample from that class
        rare_c  = np.random.choice(list(dataset.rare_class_index.keys()))
        mix_idx = np.random.choice(dataset.rare_class_index[rare_c])
        mix_mel, mix_label, _ = dataset[mix_idx]

        # Ensure mix_mel is padded/truncated to max_length
        t = mix_mel.shape[-1]
        if t > max_length:
            mix_mel = mix_mel[..., :max_length]
        elif t < max_length:
            mix_mel = F.pad(mix_mel, (0, max_length - t))

        lam = np.random.beta(alpha, alpha)
        mels_t[i]   = lam * mels_t[i] + (1 - lam) * mix_mel
        labels_t[i] = torch.maximum(labels_t[i], mix_label)  # hard union

    return mels_t, labels_t, names


class RareClassMixupCollate:
    """
    Picklable collate class for Windows multiprocessing.
    Captures necessary dataset state without using local closures.
    """
    def __init__(self, dataset, max_length=626):
        self.dataset_split = dataset.split
        self.rare_class_index = getattr(dataset, 'rare_class_index', {})
        self.dataset = dataset  # Needed because rare_class_mixup_collate calls dataset[mix_idx]
        self.max_length = max_length

    def __call__(self, batch):
        return rare_class_mixup_collate(batch, self.dataset, max_length=self.max_length)


# ---------------------------------------------------------------------------
# 7. Evaluation Utilities
# ---------------------------------------------------------------------------

def comprehensive_evaluation(predictions, targets, threshold=0.5):
    aps  = [average_precision_score(targets[:, i], predictions[:, i])
            for i in range(targets.shape[1]) if targets[:, i].sum() > 0]
    aucs = [roc_auc_score(targets[:, i], predictions[:, i])
            for i in range(targets.shape[1]) if len(np.unique(targets[:, i])) > 1]
    binary_preds = (predictions > threshold).astype(int)
    return {
        'mAP':      np.mean(aps) if aps else 0.0,
        'AUC':      np.mean(aucs) if aucs else 0.0,
        'F1_micro': f1_score(targets, binary_preds, average='micro', zero_division=0),
        'F1_macro': f1_score(targets, binary_preds, average='macro', zero_division=0),
    }


def calibrate_thresholds(predictions, targets, sweep=np.arange(0.05, 0.96, 0.05)):
    """
    For each class, find the threshold in `sweep` that maximises per-class F1
    on the provided predictions/targets (should be val split).
    Returns a (n_classes,) array of thresholds.
    """
    n_classes  = targets.shape[1]
    thresholds = np.full(n_classes, 0.5)
    for c in range(n_classes):
        if targets[:, c].sum() == 0:
            continue
        best_f1, best_t = 0., 0.5
        for t in sweep:
            preds_c = (predictions[:, c] > t).astype(int)
            f1 = f1_score(targets[:, c], preds_c, zero_division=0)
            if f1 > best_f1:
                best_f1, best_t = f1, t
        thresholds[c] = best_t
    return thresholds


# ---------------------------------------------------------------------------
# 8. Training Monitor
# ---------------------------------------------------------------------------

class TrainingMonitor:
    def compute_gradient_norm(self, model):
        return sum(p.grad.data.norm(2).item() ** 2
                   for p in model.parameters() if p.grad is not None) ** 0.5

    def compute_weight_update_ratio(self, model, optimizer):
        lr = optimizer.param_groups[0]['lr']
        ratios = [(lr * p.grad.data).norm(2).item() / p.data.norm(2).item()
                  for p in model.parameters()
                  if p.grad is not None and p.data.norm(2).item() > 1e-8]
        return float(np.mean(ratios)) if ratios else 0.0

    def compute_grad_min_max(self, model):
        grads = [p.grad.data.view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return 0.0, 0.0
        all_grads = torch.cat(grads)
        return all_grads.min().item(), all_grads.max().item()


# ---------------------------------------------------------------------------
# 9. Training Loop
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, device, train_dataset,
                num_epochs=100,
                save_path='results/ConvNeXt_SplitBandASL/best_model.pth',
                resume=False,
                checkpoint_path='results/ConvNeXt_SplitBandASL/checkpoint.pth',
                metrics_path='results/ConvNeXt_SplitBandASL/metrics.json',
                run_name=None):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)

    # ASL — no pos_weight needed
    criterion = AsymmetricLoss(gamma_neg=4, gamma_pos=0, margin=0.05)

    warmup_epochs = 5
    base_lr       = 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

    # Single smooth cosine decay — no restarts
    warmup_sched  = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=1e-6 / base_lr, end_factor=1.0, total_iters=warmup_epochs)
    cosine_sched  = optim.lr_scheduler.CosineAnnealingLR(
        optimizer, T_max=num_epochs - warmup_epochs, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, schedulers=[warmup_sched, cosine_sched], milestones=[warmup_epochs])

    monitor     = TrainingMonitor()
    start_epoch = 0
    best_map    = 0.0
    history     = {
        'run_metadata': {'run_name': run_name},
        'train_loss': [], 'val_loss': [], 'val_mAP': [],
        'grad_norm': [], 'update_ratio': [], 'grad_min': [], 'grad_max': [],
        'lse_beta_low': [], 'lse_beta_high': [],
    }

    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_map    = ckpt['best_map']
        history     = ckpt['history']
        print(f"Resumed at epoch {start_epoch}. Best mAP: {best_map:.4f}")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        gn, ur, gmin, gmax = [], [], [], []

        for mels, labels, _ in tqdm(train_loader, desc="Training"):
            mels, labels = mels.to(device), labels.to(device)
            optimizer.zero_grad()
            loss = criterion(model(mels), labels)
            loss.backward()

            gn.append(monitor.compute_gradient_norm(model))
            ur.append(monitor.compute_weight_update_ratio(model, optimizer))
            mn, mx = monitor.compute_grad_min_max(model)
            gmin.append(mn); gmax.append(mx)

            clip = 0.1 if epoch < 5 else 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            optimizer.step()
            running_loss += loss.item()

        model.eval()
        val_loss = 0.0
        all_logits, all_targets = [], []
        with torch.no_grad():
            for mels, labels, _ in tqdm(val_loader, desc="Validation"):
                mels, labels = mels.to(device), labels.to(device)
                out = model(mels)
                val_loss += criterion(out, labels).item()
                all_logits.append(out.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        logits  = np.concatenate(all_logits)
        targets = np.concatenate(all_targets)
        # Safe sigmoid: clip logits to avoid overflow
        preds   = 1.0 / (1.0 + np.exp(-np.clip(logits, -88, 88)))
        metrics = comprehensive_evaluation(preds, targets)
        val_map = metrics['mAP']

        # Log learnable LSE β values for monitoring
        beta_low  = float(model.pool.lse_low.beta.clamp(min=0.5).item())
        beta_high = float(model.pool.lse_high.beta.clamp(min=0.5).item())

        history['train_loss'].append(running_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_mAP'].append(val_map)
        history['grad_norm'].append(float(np.mean(gn)))
        history['update_ratio'].append(float(np.mean(ur)))
        history['grad_min'].append(float(np.mean(gmin)))
        history['grad_max'].append(float(np.mean(gmax)))
        history['lse_beta_low'].append(beta_low)
        history['lse_beta_high'].append(beta_high)

        print(f"  Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"  mAP: {val_map:.4f} | AUC: {metrics['AUC']:.4f} | F1-macro: {metrics['F1_macro']:.4f}")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f} | β_low: {beta_low:.3f} | β_high: {beta_high:.3f}")

        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), save_path)
            print(f"  >>> New best mAP: {best_map:.4f}. Saved.")

        scheduler.step()
        torch.save({
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_map': best_map,
            'history': history,
        }, checkpoint_path)

        run_str  = "run" + os.path.basename(save_path).split('run')[-1].split('.')[0] \
                   if 'run' in save_path else ""
        epoch_fn = f"model_{run_str}_epoch_{epoch+1}.pth" if run_str else f"model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), os.path.join(os.path.dirname(save_path), epoch_fn))

        if metrics_path:
            with open(metrics_path, 'w') as f:
                json.dump(history, f, indent=4)

    return history


# ---------------------------------------------------------------------------
# 10. Main
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(description="ConvNeXt SplitBand V2")
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--epochs',       type=int, default=100)
    parser.add_argument('--batch_size',   type=int, default=16)
    parser.add_argument('--limit_samples', type=int, default=None)

    script_dir         = os.path.dirname(os.path.abspath(__file__))
    default_results    = os.path.join(script_dir, 'results', 'ConvNeXt_SplitBandASL')
    os.makedirs(default_results, exist_ok=True)

    existing_runs = [f for f in os.listdir(default_results)
                     if f.startswith('metrics_run') and f.endswith('.json')]
    run_nums = [int(f.replace('metrics_run', '').replace('.json', ''))
                for f in existing_runs
                if f.replace('metrics_run', '').replace('.json', '').isdigit()]
    run_id = max(run_nums) + 1 if run_nums else 1
    print(f"Starting SplitBand V2 Run #{run_id}")

    parser.add_argument('--metrics_path',    type=str,
                        default=os.path.join(default_results, f'metrics_run{run_id}.json'))
    parser.add_argument('--resume',          action='store_true')
    parser.add_argument('--checkpoint_path', type=str,
                        default=os.path.join(default_results, f'checkpoint_run{run_id}.pth'))
    parser.add_argument('--save_path',       type=str,
                        default=os.path.join(default_results, f'best_model_run{run_id}.pth'))
    args = parser.parse_args()

    device   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")

    csv_path = os.path.join(args.dataset_path, 'annotations.csv')
    frontend = EnhancedAudioFrontend(n_mels=256)

    train_dataset = SONYCUSTDataset(csv_path, args.dataset_path, split='train',
                                    frontend=frontend, limit_samples=args.limit_samples)
    val_dataset   = SONYCUSTDataset(csv_path, args.dataset_path, split='validate',
                                    frontend=frontend, limit_samples=args.limit_samples)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=RareClassMixupCollate(train_dataset),
                              num_workers=4, pin_memory=True)
    val_loader   = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False,
                              collate_fn=RareClassMixupCollate(val_dataset),
                              num_workers=4, pin_memory=True)

    convnext_params = {
        'input_channels': 1,
        'depths': [2, 2, 6, 2],
        'dims':   [64, 128, 256, 512],
        'drop_path_rate': 0.2,
        'layer_scale_init_value': 1e-6
    }
    pooling_params = {
        'input_dim':  512,
        'output_dim': 512,
        'dropout':    0.1,
        'lse_beta':   5.0,
    }
    mlp_params = {
        'input_dim':    512,
        'num_classes':  23,
        'dropout_rate': 0.3
    }

    model = ConvNeXtTagger(convnext_params, pooling_params, mlp_params).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    pool_params  = sum(p.numel() for p in model.pool.parameters())
    print(f"Total params:           {total_params:,}")
    print(f"SplitBandPoolingV2:     {pool_params:,}")
    print(f"Rare classes in train:  {len(train_dataset.rare_class_ids)} / {len(LABEL_COLS)}")
    print(f"Mel resolution:         256 bins, fmax=16kHz")
    print(f"Loss:                   AsymmetricLoss (gamma_neg=4, margin=0.05)")

    history = train_model(
        model, train_loader, val_loader, device, train_dataset,
        num_epochs=args.epochs,
        resume=args.resume,
        checkpoint_path=args.checkpoint_path,
        save_path=args.save_path,
        metrics_path=args.metrics_path,
        run_name="ConvNeXt-SplitBandASL"
    )

    with open(args.metrics_path, 'w') as f:
        json.dump(history, f, indent=4)

    best_mAP = max(history['val_mAP']) if history['val_mAP'] else 0.0
    print(f"\nTraining Complete. Best mAP: {best_mAP:.4f}")


if __name__ == '__main__':
    main()
