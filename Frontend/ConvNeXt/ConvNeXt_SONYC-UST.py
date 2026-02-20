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

def entropy_regularisation(attn_weights, scale=0.01):
    pool = attn_weights.mean(dim=1)
    pool = F.softmax(pool, dim=-1)
    entropy = -(pool * (pool + 1e-9).log()).sum(dim=-1).mean()
    return scale * entropy


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
    """Upgraded ConvNeXt backbone for spectrogram/audio features."""
    def __init__(self,
                 input_channels=1,
                 depths=[5, 5, 15, 5],
                 dims=[128, 256, 512, 1024],
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

class FactorizedAttentionPooling(nn.Module):
    def __init__(self, input_dim=512, num_heads=4):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True, dropout=0.0)
        self.freq_attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True, dropout=0.0)
        self.time_norm = nn.LayerNorm(input_dim)
        self.freq_norm = nn.LayerNorm(input_dim)
        self.time_dropout = nn.Dropout(0.1)
        self.freq_dropout = nn.Dropout(0.1)
        self.out_proj = nn.Linear(input_dim, input_dim)

    def forward(self, x):
        B, C, F, T = x.shape

        # Stage 1: Time-first (per frequency bin independently)
        x_t = x.permute(0, 2, 3, 1).contiguous().view(B * F, T, C)
        attn_out, time_w = self.time_attn(x_t, x_t, x_t,
                                           need_weights=True,
                                           average_attn_weights=True)
        x_t = self.time_norm(attn_out + x_t)
        x_t = self.time_dropout(x_t)
        x_t = x_t.mean(dim=1).view(B, F, C)       # (B, F, C)

        # Stage 2: Frequency
        attn_out, freq_w = self.freq_attn(x_t, x_t, x_t,
                                           need_weights=True,
                                           average_attn_weights=True)
        x_f = self.freq_norm(attn_out + x_t)
        x_f = self.freq_dropout(x_f)
        pooled = x_f.mean(dim=1)                   # (B, C)

        return self.out_proj(pooled), time_w, freq_w


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
    """
    Lightweight classifier for multi-label audio tagging.
    """
    def __init__(self, input_dim=256, num_classes=23, dropout_rate=0.3):
        super().__init__()
        
        self.classifier = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(128, num_classes)
        )
    
    def forward(self, x):
        return self.classifier(x)

class ConvNeXtTagger(nn.Module):
    def __init__(self, convnext_params, attention_params, mlp_params):
        super().__init__()
        self.convnext = ConvNeXt2D(**convnext_params)
        self.att_pool = FactorizedAttentionPooling(**attention_params)
        self.mlp = MLPClassifier(**mlp_params)

    def forward(self, x):
        features = self.convnext(x)
        pooled, time_w, freq_w = self.att_pool(features)
        logits = self.mlp(pooled)
        if self.training:
            return logits, time_w, freq_w
        return logits


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
    def __init__(self, csv_path, audio_dir, split='train', frontend=None, limit_samples=None):
        self.label_cols = [
            '1-1_small-sounding-engine_presence', '1-2_medium-sounding-engine_presence', '1-3_large-sounding-engine_presence',
            '2-1_rock-drill_presence', '2-2_jackhammer_presence', '2-3_hoe-ram_presence', '2-4_pile-driver_presence',
            '3-1_non-machinery-impact_presence',
            '4-1_chainsaw_presence', '4-2_small-medium-rotating-saw_presence', '4-3_large-rotating-saw_presence',
            '5-1_car-horn_presence', '5-2_car-alarm_presence', '5-3_siren_presence', '5-4_reverse-beeper_presence',
            '6-1_stationary-music_presence', '6-2_mobile-music_presence', '6-3_ice-cream-truck_presence',
            '7-1_person-or-small-group-talking_presence', '7-2_person-or-small-group-shouting_presence', '7-3_large-crowd_presence', '7-4_amplified-speech_presence',
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

    def get_pos_weights(self):
        """Compute the positive weights for imbalance handling."""
        labels = self.df[self.label_cols].values
        binary_labels = np.where(labels > 0, 1.0, 0.0)
        
        pos_counts = binary_labels.sum(axis=0)
        total_counts = len(binary_labels)
        neg_counts = total_counts - pos_counts
        
        pos_weights = neg_counts / (pos_counts + 1e-6)
        
        # Cap weights to prevent gradient explosion in downsample layers
        pos_weights = np.clip(pos_weights, a_min=None, a_max=5.0)
        
        return torch.tensor(pos_weights, dtype=torch.float32)

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_name = row['audio_filename']
        
        if audio_name not in self.file_map:
            raise FileNotFoundError(f"Audio file {audio_name} not found in {self.audio_dir} or subdirectories.")
            
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

def pad_truncate_collate(batch, max_length=626):
    mels = []
    labels = []
    names = []

    for mel, label, name in batch:
        time_dim = mel.shape[-1]
        
        if time_dim > max_length:
            mel = mel[..., :max_length]
        elif time_dim < max_length:
            pad_amount = max_length - time_dim
            mel = F.pad(mel, (0, pad_amount))

        mels.append(mel)
        labels.append(label)
        names.append(name)

    batch_mels = torch.stack(mels)
    batch_labels = torch.stack(labels)

    return batch_mels, batch_labels, names

def comprehensive_evaluation(predictions, targets, threshold=0.5):
    """
    Complete evaluation suite for multi-label classification.
    predictions: probabilities (after sigmoid)
    targets: binary labels
    """
    aps = []
    for i in range(targets.shape[1]):
        if targets[:, i].sum() > 0:
            ap = average_precision_score(targets[:, i], predictions[:, i])
            aps.append(ap)
    mAP = np.mean(aps) if aps else 0.0
    
    auc_scores = []
    for i in range(targets.shape[1]):
        if len(np.unique(targets[:, i])) > 1:
            auc = roc_auc_score(targets[:, i], predictions[:, i])
            auc_scores.append(auc)
    mean_auc = np.mean(auc_scores) if auc_scores else 0.0
    
    binary_preds = (predictions > threshold).astype(int)
    f1_micro = f1_score(targets, binary_preds, average='micro', zero_division=0)
    f1_macro = f1_score(targets, binary_preds, average='macro', zero_division=0)
    
    precision_samples = []
    recall_samples = []
    for i in range(len(targets)):
        true_positives = np.sum(binary_preds[i] & targets[i].astype(int))
        pred_positives = np.sum(binary_preds[i])
        actual_positives = np.sum(targets[i])
        
        prec = true_positives / pred_positives if pred_positives > 0 else 0
        rec = true_positives / actual_positives if actual_positives > 0 else 0
        
        precision_samples.append(prec)
        recall_samples.append(rec)
    
    return {
        'mAP': mAP,
        'AUC': mean_auc,
        'F1_micro': f1_micro,
        'F1_macro': f1_macro,
        'Precision_sample': np.mean(precision_samples),
        'Recall_sample': np.mean(recall_samples)
    }

class EarlyStopping:
    def __init__(self, patience=15, min_delta=0.001):
        self.patience = patience
        self.min_delta = min_delta
        self.counter = 0
        self.best_score = None
        self.early_stop = False
        
    def __call__(self, val_map):
        if self.best_score is None:
            self.best_score = val_map
        elif val_map < self.best_score + self.min_delta:
            self.counter += 1
            if self.counter >= self.patience:
                self.early_stop = True
        else:
            self.best_score = val_map
            self.counter = 0

    def get_state(self):
        return {
            'counter': self.counter,
            'best_score': self.best_score,
            'early_stop': self.early_stop
        }
    
    def load_state(self, state):
        self.counter = state.get('counter', 0)
        self.best_score = state.get('best_score', None)
        self.early_stop = state.get('early_stop', False)

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


def train_model(model, train_loader, val_loader, device, num_epochs=100, save_path='results/best_sonyc_model.pth', resume=False, checkpoint_path='results/checkpoint_last.pth', metrics_path='results/metrics.json'):
    """
    Upgraded training pipeline for SONYC-UST with imbalance handling, continuous saving, and result organization.
    """

    results_dir = os.path.dirname(save_path)
    if results_dir:
        os.makedirs(results_dir, exist_ok=True)
    
    pos_weights = train_loader.dataset.get_pos_weights().to(device)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weights)

    warmup_epochs = 5
    base_lr = 1e-4
    warmup_start_lr = 1e-6

    optimizer = optim.AdamW(
        model.parameters(),
        lr=base_lr,
        weight_decay=1e-4
    )

    warmup_val = warmup_start_lr / base_lr
    warmup_scheduler = optim.lr_scheduler.LinearLR(
        optimizer, start_factor=warmup_val, end_factor=1.0, total_iters=warmup_epochs
    )

    cosine_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(
        optimizer, T_0=15, T_mult=2, eta_min=1e-6
    )

    scheduler = optim.lr_scheduler.SequentialLR(
        optimizer, 
        schedulers=[warmup_scheduler, cosine_scheduler], 
        milestones=[warmup_epochs]
    )
    
    early_stopping = EarlyStopping(patience=15)
    monitor = TrainingMonitor()
    
    start_epoch = 0
    best_map = 0.0
    history = {
        'train_loss': [], 
        'val_loss': [], 
        'val_mAP': [],
        'grad_norm': [],
        'update_ratio': [],
        'grad_min': [],
        'grad_max': []
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
        early_stopping.load_state(checkpoint['early_stopping_state'])
        print(f"Resumed from epoch {start_epoch}. Previous best mAP: {best_map:.4f}")

    for epoch in range(start_epoch, num_epochs):
        print(f"\nEpoch {epoch+1}/{num_epochs}")
        
        model.train()
        running_loss = 0.0
        train_bar = tqdm(train_loader, desc="Training")
        
        epoch_grad_norms = []
        epoch_update_ratios = []
        epoch_grad_mins = []
        epoch_grad_maxs = []

        for mels, labels, _ in train_bar:
            mels, labels = mels.to(device), labels.to(device).float()
            
            optimizer.zero_grad()
            outputs, time_w, freq_w = model(mels)
            loss = (criterion(outputs, labels)
                  + entropy_regularisation(time_w, scale=0.01)
                  + entropy_regularisation(freq_w, scale=0.01))
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
            train_bar.set_postfix(loss=loss.item())

        avg_train_loss = running_loss / len(train_loader)
        avg_grad_norm = np.mean(epoch_grad_norms)
        avg_update_ratio = np.mean(epoch_update_ratios)
        avg_grad_min = np.mean(epoch_grad_mins)
        avg_grad_max = np.mean(epoch_grad_maxs)

        model.eval()
        val_loss = 0.0
        all_logits = []
        all_targets = []
        
        with torch.no_grad():
            for mels, labels, _ in tqdm(val_loader, desc="Validation"):
                mels, labels = mels.to(device), labels.to(device).float()
                outputs = model(mels)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                all_logits.append(outputs.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        print("\n--- Prediction Spot Check ---")
        try:
            mels_check, labels_check, _ = next(iter(val_loader))
            mels_check = mels_check.to(device)
            outputs_check = model(mels_check)
            probs_check = torch.sigmoid(outputs_check)

            for i in range(min(3, len(probs_check))):
                print(f"Sample {i}:")
                print(f"  Label sum:  {labels_check[i].sum().item()}")
                print(f"  Prob max:   {probs_check[i].max().item():.4f}")
            
            print(f"=== Overall Batch Stats ===")
            print(f"Pred mean:  {probs_check.mean():.4f}")
            print(f"Pred max:   {probs_check.max():.4f}")
            print(f"Preds>0.5:  {(probs_check > 0.5).sum().item()} / {probs_check.numel()}")
            print(f"Labels sum: {labels_check.sum().item()} / {labels_check.numel()}")
        except Exception as e:
            print(f"Spot Check Failed: {e}")

        logits = np.concatenate(all_logits, axis=0)
        targets = np.concatenate(all_targets, axis=0)
        
        predictions = 1 / (1 + np.exp(-logits))
        
        metrics = comprehensive_evaluation(predictions, targets)
        avg_val_loss = val_loss / len(val_loader)
        val_map = metrics['mAP']
        
        history['train_loss'].append(avg_train_loss)
        history['val_loss'].append(avg_val_loss)
        history['val_mAP'].append(val_map)
        history['grad_norm'].append(float(avg_grad_norm))
        history['update_ratio'].append(float(avg_update_ratio))
        history['grad_min'].append(float(avg_grad_min))
        history['grad_max'].append(float(avg_grad_max))
        
        print(f"Epoch {epoch+1} Results:")
        print(f"  Train Loss:   {avg_train_loss:.4f}")
        print(f"  Val Loss:     {avg_val_loss:.4f}")
        print(f"  mAP:          {val_map:.4f} (Primary)")
        print(f"  AUC-ROC:      {metrics['AUC']:.4f}")
        print(f"  Macro F1:     {metrics['F1_macro']:.4f}")
        print(f"  Grad Norm:    {avg_grad_norm:.4f}")
        print(f"  Grad Min/Max: {avg_grad_min:.2e} / {avg_grad_max:.2e}")
        print(f"  Update Ratio: {avg_update_ratio:.2e}")
        print(f"  LR:           {optimizer.param_groups[0]['lr']:.6f}")

        if val_map > best_map:
            best_map = val_map
            torch.save(model.state_dict(), save_path)
            print(f"  New best mAP: {best_map:.4f}. Model saved.")
        
        scheduler.step()
        
        checkpoint = {
            'epoch': epoch,
            'model_state_dict': model.state_dict(),
            'optimizer_state_dict': optimizer.state_dict(),
            'scheduler_state_dict': scheduler.state_dict(),
            'best_map': best_map,
            'history': history,
            'early_stopping_state': early_stopping.get_state()
        }
        torch.save(checkpoint, checkpoint_path)
        
        run_str = "run" + os.path.basename(save_path).split('run')[-1].split('.')[0] if 'run' in save_path else ""
        epoch_filename = f"model_{run_str}_epoch_{epoch+1}.pth" if run_str else f"model_epoch_{epoch+1}.pth"
        
        epoch_model_path = os.path.join(os.path.dirname(save_path), epoch_filename)
        torch.save(model.state_dict(), epoch_model_path)
        
        if metrics_path:
            with open(metrics_path, 'w') as f:
                json.dump(history, f, indent=4)
            print(f"  Metrics updated: {os.path.basename(metrics_path)}")
        
        early_stopping(val_map)
        if early_stopping.early_stop:
            print(f"\nEarly stopping triggered at epoch {epoch+1}")
            break
            
    return history

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to SONYC-UST dataset')
    parser.add_argument('--epochs', type=int, default=100)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--limit_samples', type=int, default=None)
    script_dir = os.path.dirname(os.path.abspath(__file__))
    default_results_dir = os.path.join(script_dir, 'results')
    os.makedirs(default_results_dir, exist_ok=True)

    existing_runs = [f for f in os.listdir(default_results_dir) if f.startswith('metrics_run') and f.endswith('.json')]
    run_nums = [int(f.replace('metrics_run', '').replace('.json', '')) for f in existing_runs if f.replace('metrics_run', '').replace('.json', '').isdigit()]
    run_id = max(run_nums) + 1 if run_nums else 1
    print(f"Starting Run #{run_id}")

    parser.add_argument('--metrics_path', type=str, default=os.path.join(default_results_dir, f'metrics_run{run_id}.json'), help='Path to save training metrics')
    parser.add_argument('--resume', action='store_true', help='Resume training from last checkpoint')
    parser.add_argument('--checkpoint_path', type=str, default=os.path.join(default_results_dir, f'checkpoint_run{run_id}.pth'), help='Path to the last checkpoint')
    parser.add_argument('--save_path', type=str, default=os.path.join(default_results_dir, f'best_model_run{run_id}.pth'), help='Path to save best model weights')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    audio_dir = args.dataset_path
    csv_path = os.path.join(args.dataset_path, 'annotations.csv')

    frontend = EnhancedAudioFrontend(n_mels=128)
    
    train_dataset = SONYCUSTDataset(csv_path, audio_dir, split='train', frontend=frontend, limit_samples=args.limit_samples)
    val_dataset = SONYCUSTDataset(csv_path, audio_dir, split='validate', frontend=frontend, limit_samples=args.limit_samples)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_truncate_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_truncate_collate)

    convnext_params = {
        'input_channels': 1,
        'depths': [2, 2, 6, 2],
        'dims': [64, 128, 256, 512],
        'drop_path_rate': 0.2,
        'layer_scale_init_value': 1e-6
    }
    attention_params = {
        'input_dim': 512,
        'num_heads': 4,
    }
    mlp_params = {
        'input_dim': 512,
        'num_classes': 23,
        'dropout_rate': 0.3
    }

    model = ConvNeXtTagger(convnext_params, attention_params, mlp_params)
    model.to(device)

    history = train_model(model, train_loader, val_loader, device, 
                          num_epochs=args.epochs, resume=args.resume, 
                          checkpoint_path=args.checkpoint_path,
                          save_path=args.save_path,
                          metrics_path=args.metrics_path)
    
    with open(args.metrics_path, 'w') as f:
        json.dump(history, f, indent=4)
        
    print(f"\nTraining Complete. Metrics saved to {args.metrics_path}")
    best_mAP = max(history['val_mAP']) if history['val_mAP'] else 0.0
    print(f"Best mAP achieved: {best_mAP:.4f}")

if __name__ == '__main__':
    main()
