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
from collections import defaultdict

# =============================================================================
# ConvNeXt-SplitBand M-BCE: Hybrid Margin BCE + 16kHz Revert
# =============================================================================
# Changes vs V2 (ASL):
#   1. MarginBCEWithLogitsLoss replaces AsymmetricLoss
#      - Restores aggressive `pos_weight` for rare classes.
#      - Keeps a small 0.02 negative margin for noise suppression.
#   2. Frontend Revert: 16kHz / 128 Mels / 8kHz Fmax
#      - Higher resolution in V2 proved to be noisier for this backbone.
#   3. LSE pooling beta increased to 10.0 (starts closer to GMP).
# =============================================================================

# ---------------------------------------------------------------------------
# 1. Hybrid Margin BCE Loss
# ---------------------------------------------------------------------------

class MarginBCEWithLogitsLoss(nn.Module):
    def __init__(self, pos_weight=1.0, margin=0.02, reduction='mean'):
        """
        Hybrid loss that uses aggressive positive weights for rare classes
        but suppresses confident negatives using a probability margin.
        """
        super().__init__()
        self.pos_weight = pos_weight
        self.margin = margin
        self.reduction = reduction

    def forward(self, logits, targets):
        p = torch.sigmoid(logits)                   # (B, C)

        # Positive branch: standard weighted BCE 
        # (No margin here to keep gradients aggressive for rare classes)
        loss_pos = targets * (-torch.log(p.clamp(min=1e-8))) * self.pos_weight

        # Negative branch: shifted probability (borrowed from ASL)
        p_neg = (p - self.margin).clamp(min=0)      # shift then clamp
        loss_neg = (1 - targets) * (-torch.log((1 - p_neg).clamp(min=1e-8)))

        loss = loss_pos + loss_neg
        return loss.mean() if self.reduction == 'mean' else loss.sum()


# ---------------------------------------------------------------------------
# 2. Backbone: ConvNeXt2D
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
# 3. LSE Pooling + SplitBandPooling_MBCE
# ---------------------------------------------------------------------------

class LSEPool2d(nn.Module):
    """Learnable-beta LogSumExp pooling over spatial dimensions."""
    def __init__(self, init_beta=10.0):
        super().__init__()
        self.beta = nn.Parameter(torch.tensor(float(init_beta)))

    def forward(self, x):
        B, C = x.shape[:2]
        flat = x.view(B, C, -1)
        r = self.beta.clamp(min=0.5)
        return (1/r) * torch.logsumexp(r * flat, dim=-1)

class SplitBandPooling_MBCE(nn.Module):
    def __init__(self, input_dim=512, output_dim=512, dropout=0.1, lse_beta=10.0):
        super().__init__()
        self.lse_pool = LSEPool2d(init_beta=lse_beta)
        self.proj = nn.Sequential(
            nn.Linear(input_dim * 4, output_dim),
            nn.LayerNorm(output_dim),
            nn.GELU(),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        B, C, F_dim, T = x.shape
        # Split low (0-4kHz) vs High (4-8kHz) at 16kHz SR
        mid = F_dim // 2
        low  = x[:, :, :mid, :]
        high = x[:, :, mid:, :]

        l_gap = F.adaptive_avg_pool2d(low, (1, 1)).view(B, C)
        l_lse = self.lse_pool(low)
        h_gap = F.adaptive_avg_pool2d(high, (1, 1)).view(B, C)
        h_lse = self.lse_pool(high)

        fused = torch.cat([l_gap, l_lse, h_gap, h_lse], dim=-1)
        return self.proj(fused)


# ---------------------------------------------------------------------------
# 4. Mel Tagger Model
# ---------------------------------------------------------------------------

class MLPClassifier(nn.Module):
    def __init__(self, input_dim=512, num_classes=23, dropout_rate=0.3):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 256),
            nn.LayerNorm(256),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(256, num_classes)
        )
    def forward(self, x):
        return self.net(x)

class ConvNeXtTagger(nn.Module):
    def __init__(self, convnext_params, pooling_params, mlp_params):
        super().__init__()
        self.backbone = ConvNeXt2D(**convnext_params)
        self.pooling  = SplitBandPooling_MBCE(**pooling_params)
        self.mlp      = MLPClassifier(**mlp_params)

    def forward(self, x):
        feat = self.backbone(x)
        pool = self.pooling(feat)
        return self.mlp(pool)


# ---------------------------------------------------------------------------
# 5. Frontend & Augmented SONYC Dataset
# ---------------------------------------------------------------------------

class MelSpectrogramTransform:
    def __init__(self, sample_rate=16000, n_fft=1024, hop_length=256, n_mels=128, fmin=50, fmax=8000):
        self.sr = sample_rate
        self.n_fft = n_fft
        self.hop = hop_length
        self.n_mels = n_mels
        self.fmin = fmin
        self.fmax = fmax

    def __call__(self, waveform):
        mel = librosa.feature.melspectrogram(
            y=waveform.numpy(), sr=self.sr, n_fft=self.n_fft,
            hop_length=self.hop, n_mels=self.n_mels, fmin=self.fmin, fmax=self.fmax
        )
        log_mel = librosa.power_to_db(mel, ref=np.max)
        norm = (log_mel - np.mean(log_mel)) / (np.std(log_mel) + 1e-6)
        return torch.tensor(norm).unsqueeze(0).float()

class SpecAugment(nn.Module):
    def __init__(self, t_mask=40, f_mask=20):
        super().__init__()
        self.tm = t_mask
        self.fm = f_mask

    def forward(self, x):
        # x: (1, F, T)
        F_dim, T_dim = x.shape[-2:]
        if self.training:
            f = np.random.randint(0, self.fm)
            f0 = np.random.randint(0, F_dim - f)
            x[..., f0:f0+f, :] = 0
            t = np.random.randint(0, self.tm)
            t0 = np.random.randint(0, T_dim - t)
            x[..., :, t0:t0+t] = 0
        return x

class EnhancedAudioFrontend(nn.Module):
    def __init__(self, n_mels=128):
        super().__init__()
        self.mel_transform = MelSpectrogramTransform(sample_rate=16000, n_mels=n_mels)
        self.spec_augment  = SpecAugment()

    def forward(self, waveform, training=True):
        mel = self.mel_transform(waveform)
        if training:
            mel = self.spec_augment(mel)
        return mel

class SONYCUSTDataset(Dataset):
    def __init__(self, csv_path, audio_dir, split='train', frontend=None):
        self.label_cols = [
            '1-1_small-sounding-engine_presence', '1-2_medium-sounding-engine_presence', '1-3_large-sounding-engine_presence',
            '2-1_rock-drill_presence', '2-2_jackhammer_presence', '2-3_hoe-ram_presence', '2-4_pile-driver_presence',
            '3-1_non-machinery-impact_presence', '4-1_chainsaw_presence', '4-2_small-medium-rotating-saw_presence',
            '4-3_large-rotating-saw_presence', '5-1_car-horn_presence', '5-2_car-alarm_presence', '5-3_siren_presence',
            '5-4_reverse-beeper_presence', '6-1_stationary-music_presence', '6-2_mobile-music_presence',
            '6-3_ice-cream-truck_presence', '7-1_person-or-small-group-talking_presence',
            '7-2_person-or-small-group-shouting_presence', '7-3_large-crowd_presence', '7-4_amplified-speech_presence',
            '8-1_dog-barking-whining_presence'
        ]
        df_raw = pd.read_csv(csv_path)
        self.df = df_raw[df_raw['split'] == split].copy()
        self.df[self.label_cols] = self.df[self.label_cols].replace(-1, 0)
        self.df = self.df.groupby('audio_filename')[self.label_cols].max().reset_index()
        
        self.audio_dir = audio_dir
        self.frontend  = frontend
        self.split     = split
        
        # Build index
        self.file_map = {}
        for root, _, files in os.walk(audio_dir):
            for f in files:
                if f.endswith('.wav'): self.file_map[f] = os.path.join(root, f)

        # Precompute rare class mapping for Mixup
        if split == 'train':
            labels = self.df[self.label_cols].values
            prevalence = labels.mean(axis=0)
            self.rare_classes = np.where(prevalence < 0.05)[0]
            self.rare_class_indices = {c: np.where(labels[:, c] > 0.5)[0] for c in self.rare_classes}
            print(f"Rare classes identified: {len(self.rare_classes)} / 23")

    def __len__(self): return len(self.df)

    def get_pos_weights(self):
        labels = self.df[self.label_cols].values
        pos = labels.sum(axis=0)
        neg = len(labels) - pos
        weights = neg / (pos + 1e-6)
        return torch.tensor(np.clip(weights, 1.0, 20.0)).float()

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_name = row['audio_filename']
        audio_path = self.file_map[audio_name]
        waveform, _ = librosa.load(audio_path, sr=16000, duration=10.0)
        waveform = torch.from_numpy(waveform).float()

        if self.frontend:
            mel = self.frontend(waveform, training=(self.split == 'train'))
        else:
            mel = waveform
        
        labels = row[self.label_cols].values.astype(np.float32)
        return mel, torch.tensor(labels), audio_name

class RareClassMixupCollate:
    """Picklable collate class for Rare-Class Mixup on Windows."""
    def __init__(self, dataset, p_mixup=0.5, max_len=626):
        self.dataset = dataset
        self.p_mixup = p_mixup
        self.max_len = max_len

    def __call__(self, batch):
        mels, labels, names = [], [], []
        for mel, label, name in batch:
            # Resize
            if mel.shape[-1] > self.max_len: mel = mel[..., :self.max_len]
            elif mel.shape[-1] < self.max_len: mel = F.pad(mel, (0, self.max_len - mel.shape[-1]))

            # Rare-class mixup
            if self.dataset.split == 'train' and np.random.rand() < self.p_mixup:
                rc = np.random.choice(self.dataset.rare_classes)
                if len(self.dataset.rare_class_indices[rc]) > 0:
                    ridx = np.random.choice(self.dataset.rare_class_indices[rc])
                    r_mel, r_label, _ = self.dataset[ridx]
                    
                    if r_mel.shape[-1] > self.max_len: r_mel = r_mel[..., :self.max_len]
                    elif r_mel.shape[-1] < self.max_len: r_mel = F.pad(r_mel, (0, self.max_len - r_mel.shape[-1]))
                    
                    lam = np.random.beta(0.4, 0.4)
                    mel = lam * mel + (1 - lam) * r_mel
                    label = torch.max(label, r_label)  # Max-label fusion

            mels.append(mel)
            labels.append(label)
            names.append(name)
            
        return torch.stack(mels), torch.stack(labels), names


# ---------------------------------------------------------------------------
# 6. Training Logic
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
        if not grads: return 0.0, 0.0
        all_grads = torch.cat(grads)
        return all_grads.min().item(), all_grads.max().item()

def train_model(model, train_loader, val_loader, device, num_epochs=100,
                exp_name='ConvNeXt_SplitBand_M-BCE',
                resume=False, no_save=False):

    save_dir = os.path.join('results', exp_name)
    save_path = os.path.join(save_dir, 'best_model.pth')
    checkpoint_path = os.path.join(save_dir, 'checkpoint.pth')
    metrics_path = os.path.join(save_dir, 'metrics.json')

    if not no_save:
        os.makedirs(save_dir, exist_ok=True)
    pos_weights = train_loader.dataset.get_pos_weights().to(device)
    # MarginBCE restores pos_weight and adds negative margin 0.02
    criterion = MarginBCEWithLogitsLoss(pos_weight=pos_weights, margin=0.02)

    optimizer = optim.AdamW(model.parameters(), lr=1e-4, weight_decay=1e-4)
    # Smooth Cosine Schedule
    warmup = optim.lr_scheduler.LinearLR(optimizer, start_factor=1e-3, end_factor=1.0, total_iters=5)
    cosine = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs-5, eta_min=1e-6)
    scheduler = optim.lr_scheduler.SequentialLR(optimizer, [warmup, cosine], milestones=[5])

    monitor = TrainingMonitor()
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'val_mAP': [], 
        'val_AUC': [],
        'val_F1': [],
        'per_class_ap': defaultdict(list),
        'grad_norm': [], 
        'update_ratio': [], 
        'lse_beta': []
    }
    best_map, start_epoch = 0.0, 0

    if resume and os.path.exists(checkpoint_path):
        ckpt = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(ckpt['model'])
        optimizer.load_state_dict(ckpt['optimizer'])
        scheduler.load_state_dict(ckpt['scheduler'])
        start_epoch, best_map, history = ckpt['epoch']+1, ckpt['best_map'], ckpt['history']
        # Convert history back to defaultdict if needed
        if not isinstance(history.get('per_class_ap'), defaultdict):
            history['per_class_ap'] = defaultdict(list, history.get('per_class_ap', {}))

    for epoch in range(start_epoch, num_epochs):
        model.train()
        running_loss = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")
        
        for mels, labels, _ in pbar:
            mels, labels = mels.to(device), labels.to(device)
            optimizer.zero_grad()
            logits = model(mels)
            loss = criterion(logits, labels)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            running_loss += loss.item()
            pbar.set_postfix(loss=loss.item())

        # Eval
        model.eval()
        val_loss, all_preds, all_targets = 0.0, [], []
        with torch.no_grad():
            for mels, labels, _ in val_loader:
                mels, labels = mels.to(device), labels.to(device)
                logits = model(mels)
                val_loss += criterion(logits, labels).item()
                all_preds.append(torch.sigmoid(logits).cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        preds, targets = np.concatenate(all_preds), np.concatenate(all_targets)
        
        # Comprehensive Evaluation
        aps = []
        class_names = val_loader.dataset.label_cols
        for i in range(23):
            if targets[:, i].sum() > 0:
                ap = average_precision_score(targets[:, i], preds[:, i])
                aps.append(ap)
                history['per_class_ap'][class_names[i]].append(float(ap))
            else:
                history['per_class_ap'][class_names[i]].append(0.0)
        
        cur_map = np.mean(aps) if aps else 0.0
        
        # AUC and F1 (macro)
        aucs = []
        for i in range(23):
            if len(np.unique(targets[:, i])) > 1:
                aucs.append(roc_auc_score(targets[:, i], preds[:, i]))
        mean_auc = np.mean(aucs) if aucs else 0.0
        
        binary_preds = (preds > 0.5).astype(int)
        f1_macro = f1_score(targets, binary_preds, average='macro', zero_division=0)
        
        history['train_loss'].append(running_loss/len(train_loader))
        history['val_loss'].append(val_loss/len(val_loader))
        history['val_mAP'].append(cur_map)
        history['val_AUC'].append(float(mean_auc))
        history['val_F1'].append(float(f1_macro))
        history['grad_norm'].append(monitor.compute_gradient_norm(model))
        history['update_ratio'].append(monitor.compute_weight_update_ratio(model, optimizer))
        history['lse_beta'].append(model.pooling.lse_pool.beta.item())

        print(f" mAP: {cur_map:.4f} (Best: {max(best_map, cur_map):.4f}) | AUC: {mean_auc:.4f} | F1: {f1_macro:.4f} | β: {history['lse_beta'][-1]:.3f}")
        
        if cur_map > best_map:
            best_map = cur_map
            if not no_save:
                torch.save(model.state_dict(), save_path)
                print(f"  >>> New Best mAP: {best_map:.4f}! Model saved.")
            else:
                print(f"  >>> New Best mAP: {best_map:.4f}! (Saving skipped due to --no_save)")
        
        scheduler.step()
        if not no_save:
            torch.save({'epoch':epoch, 'model':model.state_dict(), 'optimizer':optimizer.state_dict(),
                        'scheduler':scheduler.state_dict(), 'best_map':best_map, 'history':dict(history)}, checkpoint_path)
            
            with open(metrics_path, 'w') as f: 
                json.dump(dict(history), f, indent=4)

    return history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--exp_name', type=str, default='ConvNeXt_SplitBand_M-BCE')
    parser.add_argument('--no_save', action='store_true', help='Skip saving checkpoints/metrics')
    parser.add_argument('--resume', action='store_true', help='Resume from checkpoint in exp_name folder')
    args = parser.parse_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    frontend = EnhancedAudioFrontend(n_mels=128)
    train_ds = SONYCUSTDataset(os.path.join(args.dataset_path, 'annotations.csv'), args.dataset_path, split='train', frontend=frontend)
    val_ds   = SONYCUSTDataset(os.path.join(args.dataset_path, 'annotations.csv'), args.dataset_path, split='validate', frontend=frontend)
    
    collate = RareClassMixupCollate(train_ds)
    # Using num_workers=0 for Windows stability with custom classes
    train_loader = DataLoader(train_ds, batch_size=16, shuffle=True, collate_fn=collate, num_workers=0)
    val_loader   = DataLoader(val_ds, batch_size=16, shuffle=False, collate_fn=pad_truncate_collate_simple, num_workers=0)

    model = ConvNeXtTagger(
        convnext_params = {'depths':[2,2,6,2], 'dims':[64,128,256,512], 'drop_path_rate':0.2},
        pooling_params  = {'input_dim':512, 'output_dim':512, 'lse_beta':10.0},
        mlp_params      = {'input_dim':512, 'num_classes':23}
    ).to(device)

    train_model(model, train_loader, val_loader, device, 
                num_epochs=args.epochs, exp_name=args.exp_name, 
                resume=args.resume, no_save=args.no_save)

def pad_truncate_collate_simple(batch, max_len=626):
    mels, labels, names = [], [], []
    for mel, label, name in batch:
        if mel.shape[-1] > max_len: mel = mel[..., :max_len]
        elif mel.shape[-1] < max_len: mel = F.pad(mel, (0, max_len - mel.shape[-1]))
        mels.append(mel); labels.append(label); names.append(name)
    return torch.stack(mels), torch.stack(labels), names

if __name__ == '__main__': main()
