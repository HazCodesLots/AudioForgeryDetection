import os
import json
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
import math
from collections import defaultdict

# =============================================================================
# ConvNeXt-GAPGMP: Fused Global Average + Global Max Pooling Baseline
# =============================================================================
# Motivation:
#   Pure GAP averages all spatial activations. This causes temporal sparsity:
#   brief events (<2-3s in a 10s clip) get diluted by silent frames and their
#   signal nearly vanishes. Rare, brief classes (small-engine, chainsaw,
#   rock-drill, hoe-ram) all suffer sub-0.21 AUPRC from this effect.
#
#   GMP takes the *peak* activation at each channel over the spatial grid.
#   Brief events — regardless of how short — produce a local peak that
#   survives max-reduction completely intact.
#
#   Fusion: concat([GAP, GMP]) doubles the feature dim fed to the MLP,
#   giving it both the average energy signature and the peak-event evidence.
# =============================================================================


class DropPath(nn.Module):
    """Stochastic Depth for ConvNeXt blocks."""
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
        output = x.div(keep_prob) * random_tensor
        return output


class SEBlock(nn.Module):
    """Squeeze-and-Excitation block for channel-wise attention."""
    def __init__(self, dim, reduction=16):
        super().__init__()
        self.fc1 = nn.Linear(dim, dim // reduction)
        self.act = nn.ReLU()
        self.fc2 = nn.Linear(dim // reduction, dim)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        y = x.mean(dim=(2, 3))
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.sigmoid(y).unsqueeze(-1).unsqueeze(-1)
        return x * y


class ConvNeXtBlock(nn.Module):
    """A ConvNeXt block for 2D data with SE attention."""
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-5):
        super().__init__()
        self.dwconv = nn.Conv2d(dim, dim, kernel_size=9, padding=4, groups=dim)
        self.norm = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1 = nn.Linear(dim, 4 * dim)
        self.act = nn.GELU()
        self.pwconv2 = nn.Linear(4 * dim, dim)
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim)), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.se = SEBlock(dim)

    def forward(self, x):
        shortcut = x
        x = self.dwconv(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.pwconv1(x)
        x = self.act(x)
        x = self.pwconv2(x)
        x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = self.se(x)
        x = shortcut + self.drop_path(x)
        return x


class ConvNeXt2D(nn.Module):
    """ConvNeXt backbone for spectrogram/audio features."""
    def __init__(self,
                 input_channels=1,
                 depths=[2, 2, 6, 2],
                 dims=[64, 128, 256, 512],
                 drop_path_rate=0.2,
                 layer_scale_init_value=1e-5):
        super().__init__()

        self.stem = nn.Sequential(
            nn.Conv2d(input_channels, dims[0], kernel_size=2, stride=2),
            nn.LayerNorm(dims[0])
        )
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(depths)):
            blocks = nn.Sequential(
                *[ConvNeXtBlock(dim=dims[i],
                                drop_path=dp_rates[cur + j],
                                layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(blocks)
            cur += depths[i]
            if i < len(depths) - 1:
                self.downsamples.append(
                    nn.Sequential(
                        nn.LayerNorm(dims[i], eps=1e-6),
                        nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2)
                    )
                )
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


class GAPGMPPooling(nn.Module):
    """
    Fused Global Average + Global Max Pooling.

    GAP captures the mean activation energy across the entire spectrogram,
    which is well-suited for continuous or sustained sounds.

    GMP captures the peak activation at each channel, preserving the signal
    from brief, sparse acoustic events that would otherwise be diluted by GAP.

    The two are concatenated, doubling the channel dimension, and then
    projected back to the original feature dimension.

    Args:
        input_dim  (int): Number of channels from the backbone (e.g. 512).
        output_dim (int): Projected output dimension. Defaults to input_dim.
        dropout    (float): Dropout after projection.
    """
    def __init__(self, input_dim=512, output_dim=512, dropout=0.1):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.proj = nn.Sequential(
            nn.Linear(input_dim * 2, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, C, F, T)
        gap_feat = self.gap(x).view(x.size(0), -1)   # (B, C)
        gmp_feat = self.gmp(x).view(x.size(0), -1)   # (B, C)
        fused = torch.cat([gap_feat, gmp_feat], dim=1)  # (B, 2C)
        return self.proj(fused)                          # (B, output_dim)


class MLPClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=23, dropout_rate=0.3):
        super().__init__()
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )

    def forward(self, x):
        return self.classifier(x)


class ConvNeXtTagger(nn.Module):
    """
    Full ConvNeXt + GAP/GMP fusion tagger.

    The backbone extracts a 2D feature map from the mel-spectrogram.
    GAPGMPPooling fuses average-energy and peak-event features.
    MLPClassifier produces per-class logits.
    """
    def __init__(self, convnext_params, pooling_params, mlp_params):
        super().__init__()
        self.convnext = ConvNeXt2D(**convnext_params)
        self.pool = GAPGMPPooling(**pooling_params)
        self.mlp = MLPClassifier(**mlp_params)

    def forward(self, x):
        features = self.convnext(x)
        pooled = self.pool(features)
        return self.mlp(pooled)


# =============================================================================
# Audio Frontend + Dataset (unchanged from ConvNeXt_GAP.py)
# =============================================================================

class MelSpectrogramTransform:
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256,
                 n_mels=128, fmin=50, fmax=8000, eps=1e-6):
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax
        self.eps = eps

    def __call__(self, waveform):
        waveform_np = waveform.numpy() if isinstance(waveform, torch.Tensor) else waveform
        if waveform_np.ndim > 1:
            waveform_np = waveform_np.squeeze()
        mel_spec = librosa.feature.melspectrogram(
            y=waveform_np, sr=self.sample_rate, n_fft=self.n_fft,
            hop_length=self.hop_length, n_mels=self.n_mels,
            fmin=self.fmin, fmax=self.fmax
        )
        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        normalized = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + self.eps)
        return torch.tensor(normalized).unsqueeze(0).float()


class SpecAugment(nn.Module):
    def __init__(self, time_mask_param=30, freq_mask_param=15):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

    def forward(self, x):
        if x.ndim == 3:
            C, F_dim, T_dim = x.shape
            f = int(np.random.uniform(0, self.freq_mask_param))
            f0 = int(np.random.uniform(0, F_dim - f))
            x[:, f0:f0+f, :] = 0
            t = int(np.random.uniform(0, self.time_mask_param))
            t0 = int(np.random.uniform(0, T_dim - t))
            x[:, :, t0:t0+t] = 0
        else:
            B, C, F_dim, T_dim = x.shape
            for i in range(B):
                f = int(np.random.uniform(0, self.freq_mask_param))
                f0 = int(np.random.uniform(0, F_dim - f))
                x[i, :, f0:f0+f, :] = 0
                t = int(np.random.uniform(0, self.time_mask_param))
                t0 = int(np.random.uniform(0, T_dim - t))
                x[i, :, :, t0:t0+t] = 0
        return x


class EnhancedAudioFrontend(nn.Module):
    def __init__(self, n_mels=128):
        super().__init__()
        self.mel_transform = MelSpectrogramTransform(
            sample_rate=16000, n_fft=1024, hop_length=256,
            n_mels=n_mels, fmin=50, fmax=8000
        )
        self.spec_augment = SpecAugment(time_mask_param=30, freq_mask_param=15)

    def forward(self, waveform, training=True):
        mel = self.mel_transform(waveform)
        if training:
            mel = self.spec_augment(mel)
        return mel


class SONYCUSTDataset(Dataset):
    def __init__(self, csv_path, audio_dir, split='train', frontend=None, limit_samples=None):
        self.label_cols = [
            '1-1_small-sounding-engine_presence', '1-2_medium-sounding-engine_presence', '1-3_large-sounding-engine_presence',
            '2-1_rock-drill_presence', '2-2_jackhammer_presence', '2-3_hoe-ram_presence', '2-4_pile-driver_presence',
            '3-1_non-machinery-impact_presence',
            '4-1_chainsaw_presence', '4-2_small-medium-rotating-saw_presence', '4-3_large-rotating-saw_presence',
            '5-1_car-horn_presence', '5-2_car-alarm_presence', '5-3_siren_presence', '5-4_reverse-beeper_presence',
            '6-1_stationary-music_presence', '6-2_mobile-music_presence', '6-3_ice-cream-truck_presence',
            '7-1_person-or-small-group-talking_presence', '7-2_person-or-small-group-shouting_presence',
            '7-3_large-crowd_presence', '7-4_amplified-speech_presence',
            '8-1_dog-barking-whining_presence'
        ]
        df_raw = pd.read_csv(csv_path)
        df_split = df_raw[df_raw['split'] == split].copy()
        df_split[self.label_cols] = df_split[self.label_cols].replace(-1, 0)
        row_sums = df_split[self.label_cols].sum(axis=1)
        df_split = df_split[row_sums <= 10]
        print(f"Aggregating labels for {split} split (Sanitized)...")
        self.df = df_split.groupby('audio_filename')[self.label_cols].max().reset_index()
        if limit_samples:
            self.df = self.df.head(limit_samples)
        self.audio_dir = audio_dir
        self.frontend = frontend
        self.split = split
        print(f"Building file index for {audio_dir}...")
        self.file_map = {}
        for root, _, files in os.walk(audio_dir):
            for f in files:
                if f.endswith('.wav'):
                    self.file_map[f] = os.path.join(root, f)
        print(f"Index built with {len(self.file_map)} unique audio files.")

        if split == 'train':
            labels = self.df[self.label_cols].values
            prev = labels.mean(axis=0)
            self.rare_classes = np.where(prev < 0.05)[0]
            self.rare_class_indices = {c: np.where(labels[:, c] > 0.5)[0]
                                       for c in self.rare_classes}
            print(f"Rare classes identified: {len(self.rare_classes)} / 23")

    def get_pos_weights(self):
        labels = self.df[self.label_cols].values
        binary_labels = np.where(labels > 0, 1.0, 0.0)
        pos_counts = binary_labels.sum(axis=0)
        neg_counts = len(binary_labels) - pos_counts
        pos_weights = neg_counts / (pos_counts + 1e-6)
        return torch.tensor(np.clip(pos_weights, None, 5.0), dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_name = row['audio_filename']
        if audio_name not in self.file_map:
            raise FileNotFoundError(f"Audio file {audio_name} not found in {self.audio_dir}.")
        audio_path = self.file_map[audio_name]
        waveform, sr = librosa.load(audio_path, sr=16000, duration=10.0)
        waveform = torch.from_numpy(waveform).float()
        if self.frontend:
            mel = self.frontend(waveform, training=(self.split == 'train'))
        else:
            mel = waveform
        labels = row[self.label_cols].values.astype(np.float32)
        labels = np.where(labels > 0, 1.0, 0.0).astype(np.float32)
        return mel, torch.tensor(labels), audio_name


class RareClassMixupCollate:
    """Picklable collate fn for Rare-Class Mixup (Windows-safe)."""
    def __init__(self, dataset, p_mixup=0.5, max_len=626):
        self.dataset, self.p_mixup, self.max_len = dataset, p_mixup, max_len

    def _resize(self, mel):
        if mel.shape[-1] > self.max_len:  return mel[..., :self.max_len]
        if mel.shape[-1] < self.max_len:  return F.pad(mel, (0, self.max_len - mel.shape[-1]))
        return mel

    def __call__(self, batch):
        mels, labels, names = [], [], []
        for mel, label, name in batch:
            mel = self._resize(mel)
            if self.dataset.split == 'train' and np.random.rand() < self.p_mixup:
                rc = np.random.choice(self.dataset.rare_classes)
                if len(self.dataset.rare_class_indices[rc]) > 0:
                    ridx  = np.random.choice(self.dataset.rare_class_indices[rc])
                    r_mel, r_label, _ = self.dataset[ridx]
                    r_mel = self._resize(r_mel)
                    lam   = np.random.beta(0.4, 0.4)
                    mel   = lam * mel + (1 - lam) * r_mel
                    label = torch.max(label, r_label)
            mels.append(mel); labels.append(label); names.append(name)
        return torch.stack(mels), torch.stack(labels), names


def pad_truncate_collate(batch, max_length=626):
    mels, labels, names = [], [], []
    for mel, label, name in batch:
        time_dim = mel.shape[-1]
        if time_dim > max_length:
            mel = mel[..., :max_length]
        elif time_dim < max_length:
            mel = F.pad(mel, (0, max_length - time_dim))
        mels.append(mel)
        labels.append(label)
        names.append(name)
    return torch.stack(mels), torch.stack(labels), names


def comprehensive_evaluation(predictions, targets, threshold=0.5):
    aps = []
    for i in range(targets.shape[1]):
        if targets[:, i].sum() > 0:
            aps.append(average_precision_score(targets[:, i], predictions[:, i]))
    mAP = np.mean(aps) if aps else 0.0

    auc_scores = []
    for i in range(targets.shape[1]):
        if len(np.unique(targets[:, i])) > 1:
            auc_scores.append(roc_auc_score(targets[:, i], predictions[:, i]))
    mean_auc = np.mean(auc_scores) if auc_scores else 0.0

    binary_preds = (predictions > threshold).astype(int)
    f1_micro = f1_score(targets, binary_preds, average='micro', zero_division=0)
    f1_macro = f1_score(targets, binary_preds, average='macro', zero_division=0)

    return {
        'mAP': mAP,
        'AUC': mean_auc,
        'F1_micro': f1_micro,
        'F1_macro': f1_macro,
    }


class TrainingMonitor:
    def __init__(self):
        self.history = {
            'grad_norm_total': [],
            'weight_update_ratio': [],
            'grad_min': [],
            'grad_max': []
        }

    def compute_gradient_norm(self, model):
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

    def compute_weight_update_ratio(self, model, optimizer):
        total_ratios = []
        lr = optimizer.param_groups[0]['lr']
        for param in model.parameters():
            if param.grad is not None:
                weight_norm = param.data.norm(2).item()
                update_norm = (lr * param.grad.data).norm(2).item()
                if weight_norm > 1e-8:
                    total_ratios.append(update_norm / weight_norm)
        return np.mean(total_ratios) if total_ratios else 0.0

    def compute_grad_min_max(self, model):
        grads = [p.grad.data.view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return 0.0, 0.0
        all_grads = torch.cat(grads)
        return all_grads.min().item(), all_grads.max().item()


def train_model(model, train_loader, val_loader, device, num_epochs=100,
                save_path='results/best_gapgmp_model.pth',
                resume=False, checkpoint_path='results/checkpoint_last.pth',
                metrics_path='results/metrics.json', run_name=None):

    results_dir = os.path.dirname(save_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)

    pos_weights = train_loader.dataset.get_pos_weights().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    warmup_epochs = 5
    base_lr = 1e-4
    warmup_start_lr = 1e-6

    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

    # Schedule: 5-epoch linear warmup -> CosineAnnealingWarmRestarts
    warmup_val = warmup_start_lr / base_lr
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_val, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=35, T_mult=1, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_epochs]
    )

    monitor = TrainingMonitor()
    start_epoch = 0
    best_map = 0.0
    history = {
        'run_metadata': {'run_name': run_name},
        'train_loss': [], 'val_loss': [], 'val_mAP': [],
        'grad_norm': [], 'update_ratio': [], 'grad_min': [], 'grad_max': []
    }

    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming training from checkpoint: {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
        start_epoch = checkpoint['epoch'] + 1
        best_map = checkpoint['best_map']
        history = checkpoint['history']
        print(f"Resumed from epoch {start_epoch}. Best mAP: {best_map:.4f}")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        train_logits, train_targets_list = [], []
        train_bar = tqdm(train_loader, desc="Training")
        epoch_grad_norms, epoch_update_ratios, epoch_grad_mins, epoch_grad_maxs = [], [], [], []

        for mels, labels, _ in train_bar:
            mels, labels = mels.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(mels)
            loss = criterion(outputs, labels)
            loss.backward()

            epoch_grad_norms.append(monitor.compute_gradient_norm(model))
            epoch_update_ratios.append(monitor.compute_weight_update_ratio(model, optimizer))
            g_min, g_max = monitor.compute_grad_min_max(model)
            epoch_grad_mins.append(g_min)
            epoch_grad_maxs.append(g_max)

            clip = 0.1 if epoch < 5 else 1.0
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=clip)
            optimizer.step()
            running_loss += loss.item()
            train_logits.append(outputs.detach().cpu().numpy())
            train_targets_list.append(labels.cpu().numpy())
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)

        model.eval()
        val_loss = 0.0
        all_logits, all_targets = [], []
        with torch.no_grad():
            for mels, labels, _ in tqdm(val_loader, desc="Validation"):
                mels, labels = mels.to(device), labels.to(device).float()
                outputs = model(mels)
                val_loss += criterion(outputs, labels).item()
                all_logits.append(outputs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        logits = np.concatenate(all_logits, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        predictions = 1 / (1 + np.exp(-logits))
        metrics = comprehensive_evaluation(predictions, targets)
        avg_val_loss = val_loss / len(val_loader)
        val_map = metrics['mAP']

        # Train accuracy (micro-F1 at 0.5)
        tr_preds   = 1 / (1 + np.exp(-np.concatenate(train_logits)))
        tr_targets = np.concatenate(train_targets_list)
        train_accuracy = float(f1_score(tr_targets, (tr_preds > 0.5).astype(int), average='micro', zero_division=0))

        val_binary = (predictions > 0.5).astype(int)
        val_accuracy = float(f1_score(targets, val_binary, average='micro', zero_division=0))

        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mAP'].append(val_map)
        history['grad_norm'].append(float(np.mean(epoch_grad_norms)))
        history['update_ratio'].append(float(np.mean(epoch_update_ratios)))
        history['grad_min'].append(float(np.mean(epoch_grad_mins)))
        history['grad_max'].append(float(np.mean(epoch_grad_maxs)))


        print(f"  Train Loss: {avg_train_loss:.4f} | Val Loss: {avg_val_loss:.4f}")
        print(f"  mAP: {val_map:.4f} | AUC: {metrics['AUC']:.4f} | F1-macro: {metrics['F1_macro']:.4f}")
        print(f"  LR:  {optimizer.param_groups[0]['lr']:.6f}")

        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), save_path)
            print(f"  >>> New best mAP: {best_map:.4f}. Model saved.")

        scheduler.step()
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_map': best_map,
            'history': history,
        }
        torch.save(checkpoint, checkpoint_path)

        run_str = "run" + os.path.basename(save_path).split('run')[-1].split('.')[0] if 'run' in save_path else ""
        epoch_filename = f"model_{run_str}_epoch_{epoch+1}.pth" if run_str else f"model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), os.path.join(os.path.dirname(save_path), epoch_filename))

        if metrics_path:
            with open(metrics_path, 'w') as f:
                json.dump(history, f, indent=4)

    return history


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=150)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--limit_samples', type=int, default=None)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_results_dir = os.path.join(script_dir, 'results', 'ConvNeXt_GAPGMP')
    os.makedirs(default_results_dir, exist_ok=True)

    existing_runs = [f for f in os.listdir(default_results_dir) if f.startswith('metrics_run') and f.endswith('.json')]
    run_nums = [int(f.replace('metrics_run', '').replace('.json', '')) for f in existing_runs if f.replace('metrics_run', '').replace('.json', '').isdigit()]
    run_id = max(run_nums) + 1 if run_nums else 1
    print(f"Starting GAPGMP Run #{run_id}")

    parser.add_argument('--metrics_path', type=str, default=os.path.join(default_results_dir, f'metrics_run{run_id}.json'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint_path', type=str, default=os.path.join(default_results_dir, f'checkpoint_run{run_id}.pth'))
    parser.add_argument('--save_path', type=str, default=os.path.join(default_results_dir, f'best_model_run{run_id}.pth'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    csv_path = os.path.join(args.dataset_path, 'annotations.csv')
    frontend = EnhancedAudioFrontend(n_mels=128)

    train_dataset = SONYCUSTDataset(csv_path, args.dataset_path, split='train',    frontend=frontend, limit_samples=args.limit_samples)
    val_dataset   = SONYCUSTDataset(csv_path, args.dataset_path, split='validate', frontend=frontend, limit_samples=args.limit_samples)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,  collate_fn=pad_truncate_collate)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False, collate_fn=pad_truncate_collate)

    convnext_params = {
        'input_channels': 1,
        'depths': [2, 2, 6, 2],
        'dims': [64, 128, 256, 512],
        'drop_path_rate': 0.2,
        'layer_scale_init_value': 1e-6
    }
    pooling_params = {
        'input_dim': 512,
        'output_dim': 512,
        'dropout': 0.1,
    }
    mlp_params = {
        'input_dim': 512,
        'num_classes': 23,
        'dropout_rate': 0.3
    }

    model = ConvNeXtTagger(convnext_params, pooling_params, mlp_params)
    model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    gap_params   = sum(p.numel() for p in model.pool.parameters())
    print(f"Total params:        {total_params:,}")
    print(f"GAPGMPPooling params: {gap_params:,}  (proj layer adds minimal overhead)")

    history = train_model(
        model, train_loader, val_loader, device,
        num_epochs=args.epochs, resume=args.resume,
        checkpoint_path=args.checkpoint_path,
        save_path=args.save_path,
        metrics_path=args.metrics_path,
        run_name="ConvNeXt-GAPGMP"
    )

    with open(args.metrics_path, 'w') as f:
        json.dump(history, f, indent=4)

    best_mAP = max(history['val_mAP']) if history['val_mAP'] else 0.0
    print(f"\nTraining Complete. Best mAP: {best_mAP:.4f}")


if __name__ == '__main__':
    main()
