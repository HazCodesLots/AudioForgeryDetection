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
from sklearn.metrics import f1_score, precision_score, recall_score, average_precision_score, roc_auc_score
from tqdm import tqdm
import argparse
import math
from collections import defaultdict

# Pure GAP Baseline model for SONYC-UST


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
    """A ConvNeXt block for 1D/2D data with SE attention."""
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
    """ConvNeXt backbone — standardized to [64,128,256,512] / [2,2,6,2] for fair comparison."""
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
        """
        x: (B, 1, F, T) -> input mel-spectrogram, shape: batch x channel x freq x time
        """
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

class GAPPooling(nn.Module):
    def __init__(self):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

    def forward(self, x):
        # x: (B, C, F, T)
        x = self.pool(x) # (B, C, 1, 1)
        return x.view(x.size(0), -1)


class SpecAugment(nn.Module):
    """Manual implementation of Time and Frequency Masking for SpecAugment."""
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

class MLPClassifier(nn.Module):
    """Standardized 2-layer MLP: 512 → 256 (GELU+LN) → 23."""
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
    def __init__(self, convnext_params, mlp_params):
        super().__init__()
        self.convnext = ConvNeXt2D(**convnext_params)
        self.pool = GAPPooling()
        self.mlp = MLPClassifier(**mlp_params)

    def forward(self, x):
        features = self.convnext(x)
        pooled = self.pool(features)
        return self.mlp(pooled)


class MelSpectrogramTransform:
    def __init__(self,
                 sample_rate=16000,
                 n_fft=1024,
                 hop_length=256,
                 n_mels=128,
                 fmin=50,
                 fmax=8000,
                 eps=1e-6):
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
            y=waveform_np,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.fmin,
            fmax=self.fmax
        )

        log_mel_spec = librosa.power_to_db(mel_spec, ref=np.max)
        normalized = (log_mel_spec - np.mean(log_mel_spec)) / (np.std(log_mel_spec) + self.eps)
        tensor = torch.tensor(normalized).unsqueeze(0).float()
        return tensor

class EnhancedAudioFrontend(nn.Module):
    def __init__(self, n_mels=128):
        super().__init__()
        self.mel_transform = MelSpectrogramTransform(
            sample_rate=16000,
            n_fft=1024,
            hop_length=256,
            n_mels=n_mels,
            fmin=50,
            fmax=8000
        )
        self.spec_augment = SpecAugment(time_mask_param=30, freq_mask_param=15)
        
    def forward(self, waveform, training=True):
        mel = self.mel_transform(waveform)
        if training:
            mel = self.spec_augment(mel)
        return mel

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

        print(f"Building file index for {audio_dir}...")
        self.file_map = {}
        for root, _, files in os.walk(audio_dir):
            for f in files:
                if f.endswith('.wav'):
                    self.file_map[f] = os.path.join(root, f)
        print(f"Index built with {len(self.file_map)} unique audio files.")

        if split == 'train':
            labels = self.df[self.label_cols].values
            prev   = labels.mean(axis=0)
            self.rare_classes       = np.where(prev < 0.05)[0]
            self.rare_class_indices = {c: np.where(labels[:, c] > 0.5)[0]
                                       for c in self.rare_classes}
            print(f"Rare classes identified: {len(self.rare_classes)} / 23")

    def get_pos_weights(self):
        labels    = self.df[self.label_cols].values
        pos       = labels.sum(axis=0)
        neg       = len(labels) - pos
        return torch.tensor(np.clip(neg / (pos + 1e-6), 1.0, 5.0)).float()

    def __len__(self):
        return len(self.df)

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


def pad_truncate_collate(batch, max_len=626):
    mels, labels, names = [], [], []
    for mel, label, name in batch:
        if mel.shape[-1] > max_len:   mel = mel[..., :max_len]
        elif mel.shape[-1] < max_len: mel = F.pad(mel, (0, max_len - mel.shape[-1]))
        mels.append(mel); labels.append(label); names.append(name)
    return torch.stack(mels), torch.stack(labels), names

def comprehensive_evaluation(predictions, targets, threshold=0.5):
    aps = [average_precision_score(targets[:, i], predictions[:, i])
           for i in range(targets.shape[1]) if targets[:, i].sum() > 0]
    aucs = [roc_auc_score(targets[:, i], predictions[:, i])
            for i in range(targets.shape[1]) if len(np.unique(targets[:, i])) > 1]
    binary_preds = (predictions > threshold).astype(int)
    return {
        'mAP':      np.mean(aps) if aps else 0.0,
        'AUC':      np.mean(aucs) if aucs else 0.0,
        'F1_micro': float(f1_score(targets, binary_preds, average='micro', zero_division=0)),
        'F1_macro': float(f1_score(targets, binary_preds, average='macro', zero_division=0)),
    }

class TrainingMonitor:
    """
    Clean monitoring of gradients and weight updates.
    Aggregates statistics per epoch.
    """
    def __init__(self):
        self.history = {
            'grad_norm_total': [],
            'weight_update_ratio': [],
            'grad_min': [],
            'grad_max': []
        }
    
    def compute_gradient_norm(self, model):
        """Calculate total gradient norm."""
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
        
        return total_norm ** 0.5
    
    def compute_weight_update_ratio(self, model, optimizer):
        """
        Calculate weight_update / weight_magnitude ratio.
        """
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
        """Calculate min and max gradient values across all parameters."""
        grads = [p.grad.data.view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return 0.0, 0.0
        all_grads = torch.cat(grads)
        return all_grads.min().item(), all_grads.max().item()


def train_model(model, train_loader, val_loader, device, num_epochs=100,
                save_path='results/ConvNeXt_GAP/ConvNeXt_GAP.pth',
                resume=False,
                checkpoint_path='results/ConvNeXt_GAP/checkpoint.pth',
                metrics_path='results/ConvNeXt_GAP/metrics.json',
                run_name=None):
    """
    Standardized training pipeline — identical hyperparameters across all 3 models.
    JSON: list of per-epoch dicts with train_loss, train_accuracy, val_loss, val_accuracy,
          val_mAP, val_AUC, val_F1_macro, gradients {grad_norm, grad_min, grad_max, update_ratio}.
    """

    results_dir = os.path.dirname(save_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    pos_weights = train_loader.dataset.get_pos_weights().to(device)
    criterion   = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    warmup_epochs = 5
    base_lr       = 1e-4
    optimizer     = optim.AdamW(model.parameters(), lr=base_lr, weight_decay=1e-4)
    # Schedule: 5-epoch linear warmup -> CosineAnnealingWarmRestarts
    warmup_val = 1e-6 / base_lr
    warmup_sched = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_val, end_factor=1.0, total_iters=warmup_epochs
    )
    cosine_sched = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=35, T_mult=1, eta_min=1e-6
    )
    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_sched, cosine_sched], 
        milestones=[warmup_epochs]
    )

    monitor     = TrainingMonitor()
    start_epoch = 0
    best_map    = 0.0
    history     = []   # list of per-epoch dicts
    
    if resume and os.path.exists(checkpoint_path):
        print(f"Resuming from {checkpoint_path}")
        ckpt = torch.load(checkpoint_path, map_location=device, weights_only=False)
        model.load_state_dict(ckpt['model_state_dict'])
        optimizer.load_state_dict(ckpt['optimizer_state_dict'])
        scheduler.load_state_dict(ckpt['scheduler_state_dict'])
        start_epoch = ckpt['epoch'] + 1
        best_map    = ckpt['best_map']
        history     = ckpt['history']
        print(f"Resumed from epoch {start_epoch}. Best mAP: {best_map:.4f}")

    for epoch in range(start_epoch, num_epochs):
        # --- Train ---
        model.train()
        running_loss  = 0.0
        train_logits, train_targets_list = [], []
        gn, ur, gmin, gmax = [], [], [], []
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs} [Train]")

        for mels, labels, _ in pbar:
            mels, labels = mels.to(device), labels.to(device).float()
            optimizer.zero_grad()
            outputs = model(mels)
            loss    = criterion(outputs, labels)
            loss.backward()

            gn.append(monitor.compute_gradient_norm(model))
            ur.append(monitor.compute_weight_update_ratio(model, optimizer))
            mn, mx = monitor.compute_grad_min_max(model)
            gmin.append(mn); gmax.append(mx)

            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=0.1 if epoch < 5 else 1.0)
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
        model.eval()
        val_loss, all_logits, all_targets = 0.0, [], []
        with torch.no_grad():
            for mels, labels, _ in tqdm(val_loader, desc="Validation"):
                mels, labels = mels.to(device), labels.to(device).float()
                outputs = model(mels)
                val_loss += criterion(outputs, labels).item()
                all_logits.append(outputs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        logits  = np.concatenate(all_logits)
        targets = np.concatenate(all_targets)
        preds   = 1 / (1 + np.exp(-logits))
        metrics = comprehensive_evaluation(preds, targets)
        val_map = metrics['mAP']
        val_acc = metrics['F1_micro']   # micro-F1 = val accuracy proxy

        epoch_record = {
            'epoch':          epoch + 1,
            'train_loss':     round(running_loss / len(train_loader), 6),
            'train_accuracy': round(train_acc, 6),
            'val_loss':       round(val_loss / len(val_loader), 6),
            'val_accuracy':   round(val_acc, 6),
            'val_mAP':        round(val_map, 6),
            'val_AUC':        round(metrics['AUC'], 6),
            'val_F1_macro':   round(metrics['F1_macro'], 6),
            'lr':             optimizer.param_groups[0]['lr'],
            'gradients': {
                'grad_norm':    round(float(np.mean(gn)), 6),
                'grad_min':     round(float(np.mean(gmin)), 6),
                'grad_max':     round(float(np.mean(gmax)), 6),
                'update_ratio': round(float(np.mean(ur)), 8),
            }
        }
        history.append(epoch_record)

        best_tag = ''
        if val_map > best_map:
            best_map = val_map
            best_tag = ' ★ NEW BEST'
            torch.save(model.state_dict(), save_path)

        print(f"  Train Loss: {epoch_record['train_loss']:.4f} | Train Acc: {train_acc:.4f}")
        print(f"  Val   Loss: {epoch_record['val_loss']:.4f}   | Val   Acc: {val_acc:.4f}")
        print(f"  mAP: {val_map:.4f} (Best: {best_map:.4f}) | AUC: {metrics['AUC']:.4f} | F1: {metrics['F1_macro']:.4f}{best_tag}")
        print(f"  Grad Norm: {epoch_record['gradients']['grad_norm']:.4f} | LR: {optimizer.param_groups[0]['lr']:.2e}")

        scheduler.step()
        torch.save({
            'epoch': epoch, 'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_map': best_map, 'history': history
        }, checkpoint_path)
        with open(metrics_path, 'w') as f:
            json.dump(history, f, indent=2)

    return history

def main():
    parser = argparse.ArgumentParser(description='ConvNeXt_GAP Baseline — Fair Retraining')
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--epochs',       type=int, default=150)
    parser.add_argument('--batch_size',   type=int, default=16)
    parser.add_argument('--seed',         type=int, default=42)
    parser.add_argument('--resume',       action='store_true')
    args = parser.parse_args()

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    script_dir  = os.path.dirname(os.path.abspath(__file__))
    results_dir = os.path.join(script_dir, 'results', 'ConvNeXt_GAP')
    os.makedirs(results_dir, exist_ok=True)

    save_path       = os.path.join(results_dir, 'ConvNeXt_GAP.pth')
    checkpoint_path = os.path.join(results_dir, 'checkpoint.pth')
    metrics_path    = os.path.join(results_dir, 'metrics.json')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Seed: {args.seed}")

    csv_path = os.path.join(args.dataset_path, 'annotations.csv')
    frontend = EnhancedAudioFrontend(n_mels=128)

    train_ds = SONYCUSTDataset(csv_path, args.dataset_path, split='train',    frontend=frontend)
    val_ds   = SONYCUSTDataset(csv_path, args.dataset_path, split='validate', frontend=frontend)

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True,
                              collate_fn=pad_truncate_collate,
                              num_workers=0, pin_memory=True)
    val_loader   = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False,
                              collate_fn=pad_truncate_collate,
                              num_workers=0, pin_memory=True)

    model = ConvNeXtTagger(
        convnext_params={'depths': [2, 2, 6, 2], 'dims': [64, 128, 256, 512],
                         'drop_path_rate': 0.2, 'layer_scale_init_value': 1e-5},
        mlp_params      ={'input_dim': 512, 'num_classes': 23, 'dropout_rate': 0.3}
    ).to(device)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Model parameters: {total_params:,}")

    train_model(model, train_loader, val_loader, device,
                num_epochs=args.epochs, save_path=save_path,
                resume=args.resume, checkpoint_path=checkpoint_path,
                metrics_path=metrics_path, run_name='ConvNeXt_GAP')


if __name__ == '__main__':
    main()
