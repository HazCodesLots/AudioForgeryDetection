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
from collections import defaultdict

# =============================================================================
# ConvNeXt-SplitBand: Separate Low- and High-Frequency Band Pooling
# =============================================================================
# Motivation:
#   The Pure GAP baseline shows consistent low-frequency dominance: F0 (the
#   lowest mel band) achieves the highest activation in every sample. This is
#   acoustically expected for urban sounds (engines/machinery dominate 50–500Hz)
#   but it means the model underweights high-frequency discriminative content
#   (1–8kHz) that separates subcategories: chainsaw buzz, car-horn harmonics,
#   siren sweeps, dog barking formants.
#
#   Fix: after the backbone produces (B, 512, F, T), split the frequency
#   dimension at the midpoint and pool each half independently with both
#   GAP and GMP. The MLP then sees:
#       [GAP_low | GMP_low | GAP_high | GMP_high]  ->  4 * 512 = 2048-D
#   projected down to 512-D before the classifier.
#
#   This explicitly forces the model to read high-frequency content even when
#   low-frequency activations are much larger in absolute magnitude.
# =============================================================================


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
    def __init__(self, input_channels=1, depths=[2, 2, 6, 2], dims=[64, 128, 256, 512],
                 drop_path_rate=0.2, layer_scale_init_value=1e-5):
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
                *[ConvNeXtBlock(dim=dims[i], drop_path=dp_rates[cur + j],
                                layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(blocks)
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


class SplitBandPooling(nn.Module):
    """
    Split the frequency dimension of the backbone feature map at the midpoint
    and apply both GAP and GMP to each half independently.

    Given backbone output (B, C, F, T):
      - low_feats  = x[:, :, :F//2, :]   -> (B, C, F/2, T)  [0 – ~4kHz approx]
      - high_feats = x[:, :, F//2:, :]   -> (B, C, F/2, T)  [~4kHz – 8kHz]

    For each half:
      - gap -> (B, C)
      - gmp -> (B, C)

    Concatenated: [gap_low | gmp_low | gap_high | gmp_high] -> (B, 4C)
    Projected back to output_dim via a small MLP.

    Args:
        input_dim  (int): Channel count from backbone (e.g. 512).
        output_dim (int): Dimension fed to the classifier. Defaults to input_dim.
        dropout    (float): Dropout after projection.
    """
    def __init__(self, input_dim=512, output_dim=512, dropout=0.1):
        super().__init__()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        # 4 streams × input_dim -> output_dim
        self.proj = nn.Sequential(
            nn.Linear(input_dim * 4, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        # x: (B, C, F, T)
        F_dim = x.shape[2]
        mid = F_dim // 2

        low  = x[:, :, :mid, :]   # (B, C, F/2, T)
        high = x[:, :, mid:, :]   # (B, C, F/2, T)

        gap_low  = self.gap(low).view(x.size(0), -1)    # (B, C)
        gmp_low  = self.gmp(low).view(x.size(0), -1)    # (B, C)
        gap_high = self.gap(high).view(x.size(0), -1)   # (B, C)
        gmp_high = self.gmp(high).view(x.size(0), -1)   # (B, C)

        fused = torch.cat([gap_low, gmp_low, gap_high, gmp_high], dim=1)  # (B, 4C)
        return self.proj(fused)  # (B, output_dim)


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
    ConvNeXt backbone + SplitBandPooling + MLPClassifier.

    The backbone is identical to ConvNeXt_GAP.py and ConvNeXt_GAPGMP.py.
    Only the pooling layer differs — results from this model are directly
    comparable to the other baselines.
    """
    def __init__(self, convnext_params, pooling_params, mlp_params):
        super().__init__()
        self.convnext = ConvNeXt2D(**convnext_params)
        self.pool = SplitBandPooling(**pooling_params)
        self.mlp = MLPClassifier(**mlp_params)

    def forward(self, x):
        features = self.convnext(x)
        pooled = self.pool(features)
        return self.mlp(pooled)


# =============================================================================
# Audio Frontend + Dataset (identical to ConvNeXt_GAP.py)
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

        # Precompute rare class mapping for Mixup
        if split == 'train':
            labels = self.df[self.label_cols].values
            prevalence = labels.mean(axis=0)
            # Threshold for rare: < 5% prevalence
            self.rare_classes = np.where(prevalence < 0.05)[0]
            self.rare_class_indices = {c: np.where(labels[:, c] > 0.5)[0] for c in self.rare_classes}
            print(f"Rare classes identified for Mixup: {len(self.rare_classes)} / 23")

    def get_pos_weights(self):
        labels = self.df[self.label_cols].values
        binary_labels = np.where(labels > 0, 1.0, 0.0)
        pos_counts = binary_labels.sum(axis=0)
        neg_counts = len(binary_labels) - pos_counts
        pos_weights = neg_counts / pos_counts
        # Exact champion clipping (5.0) from the 0.33 run
        return torch.tensor(np.clip(pos_weights, None, 5.0), dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_name = row['audio_filename']
        if audio_name not in self.file_map:
            raise FileNotFoundError(f"Audio file {audio_name} not found.")
        waveform, _ = librosa.load(self.file_map[audio_name], sr=16000, duration=10.0)
        waveform = torch.from_numpy(waveform).float()
        mel = self.frontend(waveform, training=(self.split == 'train')) if self.frontend else waveform
        labels = np.where(row[self.label_cols].values.astype(np.float32) > 0, 1.0, 0.0).astype(np.float32)
        return mel, torch.tensor(labels), audio_name

def pad_truncate_collate(batch, max_length=626):
    mels, labels, names = [], [], []
    for mel, label, name in batch:
        t = mel.shape[-1]
        if t > max_length:
            mel = mel[..., :max_length]
        elif t < max_length:
            mel = F.pad(mel, (0, max_length - t))
        mels.append(mel); labels.append(label); names.append(name)
    return torch.stack(mels), torch.stack(labels), names


def comprehensive_evaluation(predictions, targets, threshold=0.5):
    aps = [average_precision_score(targets[:, i], predictions[:, i])
           for i in range(targets.shape[1]) if targets[:, i].sum() > 0]
    mAP = np.mean(aps) if aps else 0.0
    aucs = [roc_auc_score(targets[:, i], predictions[:, i])
            for i in range(targets.shape[1]) if len(np.unique(targets[:, i])) > 1]
    binary_preds = (predictions > threshold).astype(int)
    return {
        'mAP': mAP,
        'AUC': np.mean(aucs) if aucs else 0.0,
        'F1_micro': f1_score(targets, binary_preds, average='micro', zero_division=0),
        'F1_macro': f1_score(targets, binary_preds, average='macro', zero_division=0),
    }


class TrainingMonitor:
    def compute_gradient_norm(self, model):
        return sum(p.grad.data.norm(2).item() ** 2 for p in model.parameters() if p.grad is not None) ** 0.5

    def compute_weight_update_ratio(self, model, optimizer):
        lr = optimizer.param_groups[0]['lr']
        ratios = [(lr * p.grad.data).norm(2).item() / p.data.norm(2).item()
                  for p in model.parameters()
                  if p.grad is not None and p.data.norm(2).item() > 1e-8]
        return np.mean(ratios) if ratios else 0.0

    def compute_grad_min_max(self, model):
        grads = [p.grad.data.view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return 0.0, 0.0
        all_grads = torch.cat(grads)
        return all_grads.min().item(), all_grads.max().item()


def train_model(model, train_loader, val_loader, device, num_epochs=150,
                save_path='results/best_splitband_model.pth',
                resume=False, checkpoint_path='results/checkpoint_last.pth',
                metrics_path='results/metrics.json', run_name=None):

    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pos_weights = train_loader.dataset.get_pos_weights().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    warmup_epochs = 5
    base_lr = 1e-4
    optimizer = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)

    # Schedule: 5-epoch linear warmup -> CosineAnnealingWarmRestarts
    # Champion Truth: The 0.33 run used restarts and peaked VERY early (Epoch 15).
    warmup_val = 1e-6 / base_lr
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
        print(f"Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_map = ckpt['best_map']
        history = ckpt['history']
        print(f"Resumed from epoch {start_epoch}. Best mAP: {best_map:.4f}")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc="Training")
        gn, ur, gmin, gmax = [], [], [], []

        for mels, labels, _ in train_bar:
            mels, labels = mels.to(device), labels.to(device).float()
            optimizer.zero_grad()
            loss = criterion(model(mels), labels)
            loss.backward()

            gn.append(monitor.compute_gradient_norm(model))
            ur.append(monitor.compute_weight_update_ratio(model, optimizer))
            mn, mx = monitor.compute_grad_min_max(model)
            gmin.append(mn); gmax.append(mx)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1 if epoch < 5 else 1.0)
            optimizer.step()
            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0.0
        all_logits, all_targets = [], []
        with torch.no_grad():
            for mels, labels, _ in tqdm(val_loader, desc="Validation"):
                mels, labels = mels.to(device), labels.to(device).float()
                out = model(mels)
                val_loss += criterion(out, labels).item()
                all_logits.append(out.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        logits  = np.concatenate(all_logits)
        targets = np.concatenate(all_targets)
        preds   = 1 / (1 + np.exp(-logits))
        metrics = comprehensive_evaluation(preds, targets)
        val_map = metrics['mAP']

        history['train_loss'].append(running_loss / len(train_loader))
        history['val_loss'].append(val_loss / len(val_loader))
        history['val_mAP'].append(val_map)
        history['grad_norm'].append(float(np.mean(gn)))
        history['update_ratio'].append(float(np.mean(ur)))
        history['grad_min'].append(float(np.mean(gmin)))
        history['grad_max'].append(float(np.mean(gmax)))

        print(f"  Train Loss: {running_loss/len(train_loader):.4f} | Val Loss: {val_loss/len(val_loader):.4f}")
        print(f"  mAP: {val_map:.4f} | AUC: {metrics['AUC']:.4f} | F1-macro: {metrics['F1_macro']:.4f}")
        print(f"  LR:  {optimizer.param_groups[0]['lr']:.6f}")

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

        run_str = "run" + os.path.basename(save_path).split('run')[-1].split('.')[0] if 'run' in save_path else ""
        epoch_fn = f"model_{run_str}_epoch_{epoch+1}.pth" if run_str else f"model_epoch_{epoch+1}.pth"
        torch.save(model.state_dict(), os.path.join(os.path.dirname(save_path), epoch_fn))

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
    # Isolated directory to prevent collision with other SplitBand variants
    default_results_dir = os.path.join(script_dir, 'results', 'ConvNeXt_SplitBand_GAPGMP')
    os.makedirs(default_results_dir, exist_ok=True)

    existing_runs = [f for f in os.listdir(default_results_dir)
                     if f.startswith('metrics_run') and f.endswith('.json')]
    run_nums = [int(f.replace('metrics_run', '').replace('.json', ''))
                for f in existing_runs if f.replace('metrics_run', '').replace('.json', '').isdigit()]
    run_id = max(run_nums) + 1 if run_nums else 1
    print(f"Starting SplitBand Run #{run_id}")

    parser.add_argument('--metrics_path', type=str,
                        default=os.path.join(default_results_dir, f'metrics_run{run_id}.json'))
    parser.add_argument('--resume', action='store_true')
    parser.add_argument('--checkpoint_path', type=str,
                        default=os.path.join(default_results_dir, f'checkpoint_run{run_id}.pth'))
    parser.add_argument('--save_path', type=str,
                        default=os.path.join(default_results_dir, f'best_model_run{run_id}.pth'))
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    csv_path = os.path.join(args.dataset_path, 'annotations.csv')
    frontend = EnhancedAudioFrontend(n_mels=128)

    train_dataset = SONYCUSTDataset(csv_path, args.dataset_path, split='train',
                                    frontend=frontend, limit_samples=args.limit_samples)
    val_dataset   = SONYCUSTDataset(csv_path, args.dataset_path, split='validate',
                                    frontend=frontend, limit_samples=args.limit_samples)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True,
                              collate_fn=pad_truncate_collate)
    val_loader   = DataLoader(val_dataset,   batch_size=args.batch_size, shuffle=False,
                              collate_fn=pad_truncate_collate)

    convnext_params = {
        'input_channels': 1,
        'depths': [2, 2, 6, 2],
        'dims': [64, 128, 256, 512],
        'drop_path_rate': 0.2,
        'layer_scale_init_value': 1e-6
    }
    # 4 streams (gap_low, gmp_low, gap_high, gmp_high) -> proj to 512
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
    pool_params  = sum(p.numel() for p in model.pool.parameters())
    print(f"Total params:          {total_params:,}")
    print(f"SplitBandPooling params: {pool_params:,}  (4C->C projection)")
    print(f"Split point: F//2 = ~4kHz boundary in 128-mel / 8kHz range")

    history = train_model(
        model, train_loader, val_loader, device,
        num_epochs=args.epochs, resume=args.resume,
        checkpoint_path=args.checkpoint_path,
        save_path=args.save_path,
        metrics_path=args.metrics_path,
        run_name="ConvNeXt-SplitBand"
    )

    with open(args.metrics_path, 'w') as f:
        json.dump(history, f, indent=4)

    best_mAP = max(history['val_mAP']) if history['val_mAP'] else 0.0
    print(f"\nTraining Complete. Best mAP: {best_mAP:.4f}")


if __name__ == '__main__':
    main()