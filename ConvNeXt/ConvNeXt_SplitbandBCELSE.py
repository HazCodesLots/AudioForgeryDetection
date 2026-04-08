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
# ConvNeXt-SplitBand BCE-LSE: Ablation — Weighted BCE + LSE Pooling
# =============================================================================
# Purpose (vs other variants):
#   SplitBand V1 : GAP + GMP      + Weighted BCE          (0.3326 mAP)
#   SplitBand M-BCE: LSE pooling  + Margin BCE             (0.2367 mAP)
#   SplitBand BCE-LSE: LSE pooling + Weighted BCE (no margin)  ← THIS FILE
#
# Key question: does LSE pooling ALONE help, independent of the loss margin?
# =============================================================================


# ---------------------------------------------------------------------------
# 1. Backbone Blocks
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
        y = self.fc1(y)
        y = self.act(y)
        y = self.fc2(y)
        y = self.sigmoid(y).unsqueeze(-1).unsqueeze(-1)
        return x * y


class ConvNeXtBlock(nn.Module):
    def __init__(self, dim, drop_path=0.0, layer_scale_init_value=1e-5):
        super().__init__()
        self.dwconv   = nn.Conv2d(dim, dim, kernel_size=9, padding=4, groups=dim)
        self.norm     = nn.LayerNorm(dim, eps=1e-6)
        self.pwconv1  = nn.Linear(dim, 4 * dim)
        self.act      = nn.GELU()
        self.pwconv2  = nn.Linear(4 * dim, dim)
        self.gamma    = nn.Parameter(layer_scale_init_value * torch.ones(dim), requires_grad=True)
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.se       = SEBlock(dim)

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
        return shortcut + self.drop_path(x)


class ConvNeXt2D(nn.Module):
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
        self.stages      = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(len(depths)):
            self.stages.append(nn.Sequential(
                *[ConvNeXtBlock(dims[i], dp_rates[cur+j], layer_scale_init_value)
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
# 2. LSE Pooling + SplitBand Head
#    Key difference vs V1: LSE replaces hard GMP.
#    Key difference vs M-BCE: no negative probability margin in the loss.
# ---------------------------------------------------------------------------

class LSEPool2d(nn.Module):
    """Learnable-beta LogSumExp pooling over spatial dims (soft maximum)."""
    def __init__(self, init_beta=10.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))

    def forward(self, x):
        B, C = x.shape[:2]
        flat = x.view(B, C, -1)
        r = self.beta.clamp(min=0.5)
        return (1.0 / r) * torch.logsumexp(r * flat, dim=-1)


class SplitBandLSEPooling(nn.Module):
    """
    Splits the 128 Mel bins into Low (0-64) and High (64-128).
    Applies Log-Sum-Exp (LSE) pooling to both, concatenates, and projects to 512.
    """
    def __init__(self, input_dim=512, output_dim=512, dropout=0.3, initial_beta=1.0):
        super().__init__()
        self.lse_pool = LSEPool2d(init_beta=initial_beta)
        self.proj = nn.Sequential(
            nn.Linear(input_dim * 4, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, C, F_dim, T = x.shape
        mid  = F_dim // 2
        low  = x[:, :, :mid, :]
        high = x[:, :, mid:, :]

        l_gap = F.adaptive_avg_pool2d(low,  (1, 1)).view(B, C)
        l_lse = self.lse_pool(low)
        h_gap = F.adaptive_avg_pool2d(high, (1, 1)).view(B, C)
        h_lse = self.lse_pool(high)

        return self.proj(torch.cat([l_gap, l_lse, h_gap, h_lse], dim=-1))


# ---------------------------------------------------------------------------
# 3. Classifier + Full Model
# ---------------------------------------------------------------------------

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
    ConvNeXt + Frequency-Split GAP/LSE Pooling + Standardized MLP.
    """
    def __init__(self, convnext_params, pooling_params, mlp_params):
        super().__init__()
        self.convnext = ConvNeXt2D(**convnext_params)
        self.pool = SplitBandLSEPooling(**pooling_params)
        self.mlp = MLPClassifier(**mlp_params)

    def forward(self, x):
        return self.mlp(self.pool(self.convnext(x)))


# ---------------------------------------------------------------------------
# 4. Frontend: 16kHz / 128 Mel (same as V1 and M-BCE)
# ---------------------------------------------------------------------------

class MelSpectrogramTransform:
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256,
                 n_mels=128, fmin=50, fmax=8000):
        self.sr, self.n_fft = sample_rate, n_fft
        self.hop, self.n_mels = hop_length, n_mels
        self.fmin, self.fmax = fmin, fmax

    def __call__(self, waveform):
        mel = librosa.feature.melspectrogram(
            y=waveform.numpy(), sr=self.sr, n_fft=self.n_fft,
            hop_length=self.hop, n_mels=self.n_mels,
            fmin=self.fmin, fmax=self.fmax
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        norm = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
        return torch.tensor(norm).unsqueeze(0).float()


class SpecAugment(nn.Module):
    def __init__(self, t_mask=40, f_mask=20):
        super().__init__()
        self.tm, self.fm = t_mask, f_mask

    def forward(self, x):
        F_dim, T_dim = x.shape[-2:]
        if self.training:
            f  = np.random.randint(0, self.fm)
            f0 = np.random.randint(0, F_dim - f) if F_dim > f else 0
            x[..., f0:f0+f, :] = 0
            t  = np.random.randint(0, self.tm)
            t0 = np.random.randint(0, T_dim - t) if T_dim > t else 0
            x[..., :, t0:t0+t] = 0
        return x


class EnhancedAudioFrontend(nn.Module):
    def __init__(self, n_mels=128):
        super().__init__()
        self.mel_transform = MelSpectrogramTransform(sample_rate=16000, n_mels=n_mels)
        self.spec_augment  = SpecAugment(t_mask=30, f_mask=15)

    def forward(self, waveform, training=True):
        mel = self.mel_transform(waveform)
        if training:
            mel = self.spec_augment(mel)
        return mel


# ---------------------------------------------------------------------------
# 5. Dataset + Collation (Rare-Class Mixup)
# ---------------------------------------------------------------------------

class SONYCUSTDataset(Dataset):
    LABEL_COLS = [
        '1-1_small-sounding-engine_presence',  '1-2_medium-sounding-engine_presence',
        '1-3_large-sounding-engine_presence',  '2-1_rock-drill_presence',
        '2-2_jackhammer_presence',             '2-3_hoe-ram_presence',
        '2-4_pile-driver_presence',            '3-1_non-machinery-impact_presence',
        '4-1_chainsaw_presence',               '4-2_small-medium-rotating-saw_presence',
        '4-3_large-rotating-saw_presence',     '5-1_car-horn_presence',
        '5-2_car-alarm_presence',              '5-3_siren_presence',
        '5-4_reverse-beeper_presence',         '6-1_stationary-music_presence',
        '6-2_mobile-music_presence',           '6-3_ice-cream-truck_presence',
        '7-1_person-or-small-group-talking_presence',
        '7-2_person-or-small-group-shouting_presence',
        '7-3_large-crowd_presence',            '7-4_amplified-speech_presence',
        '8-1_dog-barking-whining_presence',
    ]

    def __init__(self, csv_path, audio_dir, split='train', frontend=None):
        self.label_cols = self.LABEL_COLS
        df_raw = pd.read_csv(csv_path)
        self.df = df_raw[df_raw['split'] == split].copy()
        self.df[self.label_cols] = self.df[self.label_cols].replace(-1, 0)
        
        row_sums = self.df[self.label_cols].sum(axis=1)
        self.df = self.df[row_sums <= 10]
        
        print(f"Aggregating labels for {split} split (Sanitized)...")
        self.df = self.df.groupby('audio_filename')[self.label_cols].max().reset_index()
        self.audio_dir = audio_dir
        self.frontend  = frontend
        self.split     = split

        # Build file index
        self.file_map = {}
        for root, _, files in os.walk(audio_dir):
            for f in files:
                if f.endswith('.wav'):
                    self.file_map[f] = os.path.join(root, f)

        if split == 'train':
            labels = self.df[self.label_cols].values
            prev   = labels.mean(axis=0)
            self.rare_classes      = np.where(prev < 0.05)[0]
            self.rare_class_indices = {c: np.where(labels[:, c] > 0.5)[0]
                                       for c in self.rare_classes}
            print(f"Rare classes identified: {len(self.rare_classes)} / 23")

    def __len__(self):
        return len(self.df)

    def get_pos_weights(self):
        labels = self.df[self.label_cols].values
        pos    = labels.sum(axis=0)
        neg    = len(labels) - pos
        return torch.tensor(np.clip(neg / (pos + 1e-6), 1.0, 5.0)).float()

    def __getitem__(self, idx):
        row  = self.df.iloc[idx]
        path = self.file_map[row['audio_filename']]
        wav, _ = librosa.load(path, sr=16000, duration=10.0)
        wav = torch.from_numpy(wav).float()
        mel = self.frontend(wav, training=(self.split == 'train')) if self.frontend else wav
        return mel, torch.tensor(row[self.label_cols].values.astype(np.float32)), row['audio_filename']


class RareClassMixupCollate:
    """Picklable collate fn for Rare-Class Mixup (Windows-safe)."""
    def __init__(self, dataset, p_mixup=0.5, max_len=626):
        self.dataset, self.p_mixup, self.max_len = dataset, p_mixup, max_len

    def _resize(self, mel):
        if mel.shape[-1] > self.max_len:
            return mel[..., :self.max_len]
        if mel.shape[-1] < self.max_len:
            return F.pad(mel, (0, self.max_len - mel.shape[-1]))
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


def pad_truncate_collate(batch, max_len=626):
    mels, labels, names = [], [], []
    for mel, label, name in batch:
        if mel.shape[-1] > max_len:   mel = mel[..., :max_len]
        elif mel.shape[-1] < max_len: mel = F.pad(mel, (0, max_len - mel.shape[-1]))
        mels.append(mel); labels.append(label); names.append(name)
    return torch.stack(mels), torch.stack(labels), names


# ---------------------------------------------------------------------------
# 6. Training Monitor
# ---------------------------------------------------------------------------

class TrainingMonitor:
    def compute_gradient_norm(self, model):
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        return total_norm ** 0.5

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
# 7. Training Loop
# ---------------------------------------------------------------------------

def train_model(model, train_loader, val_loader, device, num_epochs=100,
                exp_name='ConvNeXt_Splitband_BCELSE',
                resume=False, no_save=False):
    """
    Standardized training pipeline — identical hyperparameters across all 3 models.
    JSON: list of per-epoch dicts with train_loss, train_accuracy, val_loss, val_accuracy,
          val_mAP, val_AUC, val_F1_macro, gradients {grad_norm, grad_min, grad_max, update_ratio},
          and lse_beta (learnable pooling beta — unique to this ablation model).
    """

    script_dir      = os.path.dirname(os.path.abspath(__file__))
    save_dir        = os.path.join(script_dir, 'results', exp_name)
    save_path       = os.path.join(save_dir, f'{exp_name}.pth')
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    metrics_path    = os.path.join(save_dir, 'metrics.json')

    if not no_save:
        os.makedirs(save_dir, exist_ok=True)

    # ----- Loss: standard Weighted BCE (no negative margin) -----
    pos_weights = train_loader.dataset.get_pos_weights().to(device)
    criterion   = nn.BCEWithLogitsLoss(
        pos_weight=pos_weights,
        reduction='mean'
    )

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)

    # ----- Optimizer + Cosine Schedule with Warmup -----
    # Schedule: 5-epoch linear warmup -> CosineAnnealingWarmRestarts
    warmup    = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-2,
                                            end_factor=1.0, total_iters=5)
    cosine    = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=35, T_mult=1, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine],
                                                milestones=[5])

    monitor  = TrainingMonitor()
    history  = []   # list of per-epoch dicts
    best_map, start_epoch = 0.0, 0

    if resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch = ckpt['epoch'] + 1
        best_map    = ckpt['best_map']
        history     = ckpt['history']
        print(f"Resumed from epoch {start_epoch}, best mAP so far: {best_map:.4f}")

    for epoch in range(start_epoch, num_epochs):
        # --- Train ---
        model.train()
        running_loss = 0.0
        train_logits, train_targets_list = [], []
        gn, ur, gmin, gmax = [], [], [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for mels, labels, _ in pbar:
            mels, labels = mels.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(mels)
            loss   = criterion(outputs, labels)
            loss.backward()

            gn.append(monitor.compute_gradient_norm(model))
            ur.append(monitor.compute_weight_update_ratio(model, optimizer))
            mn, mx = monitor.compute_grad_min_max(model)
            gmin.append(mn); gmax.append(mx)

            torch.nn.utils.clip_grad_norm_(model.parameters(), 0.1 if epoch < 5 else 1.0)
            optimizer.step()
            running_loss += loss.item()
            train_logits.append(outputs.detach().cpu().numpy())
            train_targets_list.append(labels.cpu().numpy())
            pbar.set_postfix(loss=f"{loss.item():.4f}")

        # Train accuracy (micro-F1 at 0.5 threshold)
        tr_preds   = 1 / (1 + np.exp(-np.concatenate(train_logits)))
        tr_targets = np.concatenate(train_targets_list)
        train_acc  = float(f1_score(tr_targets, (tr_preds > 0.5).astype(int),
                                    average='micro', zero_division=0))

        # --- Validate ---
        val_loss, all_preds, all_targets = 0.0, [], []
        with torch.no_grad():
            for mels, labels, _ in val_loader:
                mels, labels = mels.to(device), labels.to(device)
                logits = model(mels)
                val_loss += criterion(logits, labels).item()
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        preds   = np.concatenate(all_preds)
        targets = np.concatenate(all_targets)

        # Per-class AUPRC
        aps, class_names = [], val_loader.dataset.label_cols
        per_class = {}
        for i in range(23):
            if targets[:, i].sum() > 0:
                ap = average_precision_score(targets[:, i], preds[:, i])
                aps.append(ap)
                per_class[class_names[i]] = round(float(ap), 6)
            else:
                per_class[class_names[i]] = 0.0
        cur_map = float(np.mean(aps)) if aps else 0.0

        aucs    = [roc_auc_score(targets[:, i], preds[:, i])
                   for i in range(23) if len(np.unique(targets[:, i])) > 1]
        mean_auc  = float(np.mean(aucs)) if aucs else 0.0
        f1_macro  = float(f1_score(targets, (preds > 0.5).astype(int),
                                   average='macro', zero_division=0))
        f1_micro  = float(f1_score(targets, (preds > 0.5).astype(int),
                                   average='micro', zero_division=0))
        lse_beta  = model.pooling.lse_pool.beta.item()

        # --- Gradients (computed on current model state after step) ---
        gn  = monitor.gradient_norm(model) if any(p.grad is not None for p in model.parameters()) else 0.0
        ur  = monitor.update_ratio(model, optimizer)

        history.append({
            'epoch': epoch + 1,
            'train_loss': float(avg_train_loss),
            'val_loss': float(avg_val_loss),
            'val_mAP': float(cur_map),
            'grad_norm': float(gn),
            'update_ratio': float(ur),
            'grad_min': float(np.mean(gmin)),
            'grad_max': float(np.mean(gmax))
        })

        best_tag = ''
        if cur_map > best_map:
            best_map = cur_map
            best_tag = ' ★ NEW BEST'
            if not no_save:
                torch.save(model.state_dict(), save_path)

        print(f"  Train Loss: {epoch_record['train_loss']:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {epoch_record['val_loss']:.4f}   | Val   Acc: {f1_micro:.4f}")
        print(f"  mAP: {cur_map:.4f} (Best: {best_map:.4f}) | "
              f"AUC: {mean_auc:.4f} | F1: {f1_macro:.4f} | "
              f"β: {lse_beta:.3f} | LR: {optimizer.param_groups[0]['lr']:.2e}"
              + best_tag)

        scheduler.step()

        if not no_save:
            torch.save({
                'epoch': epoch, 'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'scheduler': scheduler.state_dict(),
                'best_map': best_map, 'history': history
            }, checkpoint_path)
            with open(metrics_path, 'w') as f:
                json.dump(history, f, indent=2)

    return history


# ---------------------------------------------------------------------------
# 8. Entry Point
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="ConvNeXt SplitBand BCE-LSE (Ablation A) — Fair Retraining")
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--epochs',       type=int, default=150)
    parser.add_argument('--batch_size',   type=int, default=16)
    parser.add_argument('--seed',         type=int, default=42)
    parser.add_argument('--exp_name',     type=str, default='ConvNeXt_Splitband_BCELSE')
    parser.add_argument('--resume',       action='store_true')
    parser.add_argument('--no_save',      action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Seed: {args.seed}")

    frontend = EnhancedAudioFrontend(n_mels=128)

    csv_path  = os.path.join(args.dataset_path, 'annotations.csv')
    train_ds  = SONYCUSTDataset(csv_path, args.dataset_path, split='train',    frontend=frontend)
    val_ds    = SONYCUSTDataset(csv_path, args.dataset_path, split='validate', frontend=frontend)

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, shuffle=True,
        collate_fn=pad_truncate_collate,
        num_workers=0, pin_memory=True
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, shuffle=False,
        collate_fn=pad_truncate_collate,
        num_workers=0, pin_memory=True
    )

    model = ConvNeXtTagger(
        convnext_params = {'depths': [2, 2, 6, 2], 'dims': [64, 128, 256, 512],
                           'drop_path_rate': 0.2, 'layer_scale_init_value': 1e-5},
        pooling_params  = {'input_dim': 512, 'output_dim': 512,
                           'dropout': 0.1, 'initial_beta': 10.0},
        mlp_params      = {'input_dim': 512, 'num_classes': 23, 'dropout_rate': 0.3}
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    train_model(
        model, train_loader, val_loader, device,
        num_epochs=args.epochs,
        exp_name=args.exp_name,
        resume=args.resume,
        no_save=args.no_save
    )


if __name__ == '__main__':
    main()
