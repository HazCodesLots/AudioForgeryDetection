import os
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np
import librosa
from sklearn.metrics import f1_score, precision_score, recall_score
from tqdm import tqdm
import argparse
import math
# import torchaudio.transforms as T  # Removed due to environment issues

# --- Model Components from Notebook ---

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

class SelfAttentivePooling(nn.Module):
    def __init__(self, input_dim=384, attention_dim=256, num_heads=4, dropout_rate=0.15):
        super().__init__()
        self.attention_dim = attention_dim
        self.num_heads = num_heads
        self.head_dim = attention_dim // num_heads  # 64 per head
        self.dropout_rate = dropout_rate

        # Q, K, V projections
        self.query = nn.Linear(input_dim, attention_dim)
        self.key = nn.Linear(input_dim, attention_dim)
        self.value = nn.Linear(input_dim, attention_dim)

        # Dropouts
        self.attn_dropout = nn.Dropout(dropout_rate)
        self.out_proj = nn.Linear(attention_dim, attention_dim)
        self.out_dropout = nn.Dropout(dropout_rate)
        
        # Residual connection
        self.layer_norm = nn.LayerNorm(attention_dim)
        self.input_proj = nn.Linear(input_dim, attention_dim)

    def forward(self, x):
        batch_size, channels, freq, time = x.shape
        # Flatten freq×time into sequence
        x_reshaped = x.permute(0, 2, 3, 1).contiguous().view(batch_size, freq * time, channels)

        Q = self.query(x_reshaped)
        K = self.key(x_reshaped)
        V = self.value(x_reshaped)

        # Multi-head split
        Q = Q.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        K = K.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)
        V = V.view(batch_size, -1, self.num_heads, self.head_dim).transpose(1, 2)

        # Scaled dot-product attention
        scores = torch.matmul(Q, K.transpose(-2, -1)) / (self.head_dim ** 0.5)
        attention_weights = F.softmax(scores, dim=-1)
        attention_weights = self.attn_dropout(attention_weights)

        attended = torch.matmul(attention_weights, V)
        attended = attended.transpose(1, 2).contiguous().view(batch_size, -1, self.attention_dim)

        # Output projection + residual
        out = self.out_proj(attended)
        out = self.out_dropout(out)
        
        x_proj = self.input_proj(x_reshaped)
        out = self.layer_norm(out + x_proj)

        # Final pooling across all positions
        pooled = out.mean(dim=1)  # (B, attention_dim)
        return pooled


class SpecAugment(nn.Module):
    """Manual implementation of Time and Frequency Masking for SpecAugment."""
    def __init__(self, time_mask_param=30, freq_mask_param=15):
        super().__init__()
        self.time_mask_param = time_mask_param
        self.freq_mask_param = freq_mask_param

    def forward(self, x):
        # x shape: (B, C, F, T) or (C, F, T)
        if x.ndim == 3:
            C, F_dim, T_dim = x.shape
            # Frequency masking
            f = int(np.random.uniform(0, self.freq_mask_param))
            f0 = int(np.random.uniform(0, F_dim - f))
            x[:, f0:f0+f, :] = 0
            # Time masking
            t = int(np.random.uniform(0, self.time_mask_param))
            t0 = int(np.random.uniform(0, T_dim - t))
            x[:, :, t0:t0+t] = 0
        else:
            B, C, F_dim, T_dim = x.shape
            for i in range(B):
                # Frequency masking
                f = int(np.random.uniform(0, self.freq_mask_param))
                f0 = int(np.random.uniform(0, F_dim - f))
                x[i, :, f0:f0+f, :] = 0
                # Time masking
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
        return self.classifier(x)  # Raw logits for BCEWithLogitsLoss

class DeepFakeDetectionModel(nn.Module):
    def __init__(self,
                 convnext_params,
                 attention_params,
                 mlp_params):
        super().__init__()

        self.convnext = ConvNeXt2D(**convnext_params)
        self.att_pool = SelfAttentivePooling(**attention_params)
        self.mlp = MLPClassifier(**mlp_params)

    def forward(self, x):
        features = self.convnext(x)
        pooled_features = self.att_pool(features)
        logits = self.mlp(pooled_features)
        return logits

# --- Enhanced Frontend and Data Layer ---

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
        self.df = pd.read_csv(csv_path)
        self.df = self.df[self.df['split'] == split].reset_index(drop=True)
        
        if limit_samples:
            self.df = self.df.head(limit_samples)
            
        self.audio_dir = audio_dir
        self.frontend = frontend
        
        # Build file index to handle subdirectories (audio-0, audio-1, etc.)
        print(f"Building file index for {audio_dir}...")
        self.file_map = {}
        for root, _, files in os.walk(audio_dir):
            for f in files:
                if f.endswith('.wav'):
                    self.file_map[f] = os.path.join(root, f)
        print(f"Index built with {len(self.file_map)} files.")
        
        # SONYC-UST 23-class fine-grained presence columns
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

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        audio_name = row['audio_filename']
        
        if audio_name not in self.file_map:
            # Fallback or error
            raise FileNotFoundError(f"Audio file {audio_name} not found in {self.audio_dir} or subdirectories.")
            
        audio_path = self.file_map[audio_name]
        
        waveform, sr = librosa.load(audio_path, sr=16000, duration=10.0) # 10s audio
        waveform = torch.from_numpy(waveform).float()
        
        if self.frontend:
            mel = self.frontend(waveform, training=self.frontend.training)
        else:
            mel = waveform
            
        labels = row[self.label_cols].values.astype(np.float32)
        # SONYC-UST convention: 1 = present, 0 = absent, others might exist but typically binary for urban sound tagging
        # We ensure it's 0 or 1 for multi-label.
        labels = np.where(labels > 0, 1.0, 0.0).astype(np.float32)
        
        return mel, torch.tensor(labels), audio_name

def pad_truncate_collate(batch, max_length=626): # ~10s at sr=16000, hop=256 => 160000/256 = 625
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

# --- Training / Loop ---

def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=20, save_path='best_sonyc_model.pth'):
    best_val_f1 = 0.0
    
    for epoch in range(num_epochs):
        print(f"Epoch {epoch+1}/{num_epochs}")
        model.train()
        running_loss = 0.0
        
        train_bar = tqdm(train_loader, desc="Training")
        for mels, labels, _ in train_bar:
            mels, labels = mels.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(mels)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            train_bar.set_postfix(loss=loss.item())

        model.eval()
        val_loss = 0.0
        all_preds = []
        all_targets = []
        
        with torch.no_grad():
            for mels, labels, _ in tqdm(val_loader, desc="Validation"):
                mels, labels = mels.to(device), labels.to(device)
                outputs = model(mels)
                loss = criterion(outputs, labels)
                val_loss += loss.item()
                
                preds = (torch.sigmoid(outputs) > 0.5).float()
                all_preds.append(preds.cpu().numpy())
                all_targets.append(labels.cpu().numpy())

        all_preds = np.concatenate(all_preds, axis=0)
        all_targets = np.concatenate(all_targets, axis=0)
        
        # Macro F1 for urban sound tagging
        f1 = f1_score(all_targets, all_preds, average='macro', zero_division=0)
        print(f"Epoch {epoch+1} - Val Loss: {val_loss/len(val_loader):.4f}, Macro F1: {f1:.4f}")
        
        if scheduler:
            scheduler.step()

        if f1 > best_val_f1:
            best_val_f1 = f1
            torch.save(model.state_dict(), save_path)
            print(f"Saved best model with F1: {best_val_f1:.4f}")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True, help='Path to SONYC-UST dataset')
    parser.add_argument('--epochs', type=int, default=30)
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--limit_samples', type=int, default=None)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Audio Directory - check if it's top level or 'audio' subfolder
    audio_dir = args.dataset_path # SONYC-UST usually unpacks to many wavs. Needs a path.
    csv_path = os.path.join(args.dataset_path, 'annotations.csv')

    frontend = EnhancedAudioFrontend(n_mels=128)
    
    train_dataset = SONYCUSTDataset(csv_path, audio_dir, split='train', frontend=frontend, limit_samples=args.limit_samples)
    val_dataset = SONYCUSTDataset(csv_path, audio_dir, split='validate', frontend=frontend, limit_samples=args.limit_samples)

    train_loader = DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True, collate_fn=pad_truncate_collate)
    val_loader = DataLoader(val_dataset, batch_size=args.batch_size, shuffle=False, collate_fn=pad_truncate_collate)

    # Model Params (adapted for 128 mels and 8 classes)
    convnext_params = {
        'input_channels': 1,
        'depths': [2, 2, 4, 2],
        'dims': [48, 96, 192, 384],
        'drop_path_rate': 0.2,
        'layer_scale_init_value': 1e-6
    }
    attention_params = {
        'input_dim': 384,       # From new lightweight ConvNeXt
        'attention_dim': 256,   # Down from 512 (lighter)
        'num_heads': 4,         # Down from 8 (still multi-head!)
        'dropout_rate': 0.15    # Up from 0.1 (more regularization)
    }
    mlp_params = {
        'input_dim': 256,
        'num_classes': 23,
        'dropout_rate': 0.3
    }

    model = DeepFakeDetectionModel(convnext_params, attention_params, mlp_params)
    model.to(device)

    criterion = nn.BCEWithLogitsLoss()
    optimizer = optim.AdamW(model.parameters(), lr=args.lr)
    
    # Simple scheduler
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=args.epochs)

    train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, device, num_epochs=args.epochs)

if __name__ == '__main__':
    main()
