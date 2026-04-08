import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import scipy.signal
import scipy.fftpack
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt

from calculate_tDCF import load_asv_metrics, compute_min_tDCF

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")


def delta(feat, N=2):
    """Compute delta features from a feature matrix."""
    if N < 1:
        raise ValueError("N must be at least 1")
    denom = 2 * sum([i**2 for i in range(1, N + 1)])
    padded = np.pad(feat, ((N, N), (0, 0)), mode='edge')
    delta_feat = np.zeros_like(feat)
    for t in range(feat.shape[0]):
        for n in range(1, N + 1):
            delta_feat[t] += n * (padded[t + N + n] - padded[t + N - n])
    return delta_feat / denom

def extract_lfcc(signal, samplerate=16000, n_fft=512, n_filters=70, n_ceps=70,
                 winlen=0.025, winstep=0.01, preemph=0.97, with_delta=True):

    signal = scipy.signal.lfilter([1, -preemph], 1, signal)

    frame_len = int(winlen * samplerate)
    frame_step = int(winstep * samplerate)
    signal_length = len(signal)
    num_frames = max(1, 1 + int((signal_length - frame_len) / frame_step))

    pad_length = (num_frames - 1) * frame_step + frame_len
    if pad_length > signal_length:
        signal = np.pad(signal, (0, pad_length - signal_length), mode='constant')
    frames = np.zeros((num_frames, frame_len))
    for i in range(num_frames):
        start = i * frame_step
        frames[i] = signal[start:start + frame_len] * np.hamming(frame_len)

    mag_frames = np.absolute(np.fft.rfft(frames, n=n_fft))
    pow_frames = (1.0 / n_fft) * (mag_frames ** 2)

    low_freq = 0
    high_freq = samplerate / 2
    hz_points = np.linspace(low_freq, high_freq, n_filters + 2)
    bins = np.floor((n_fft + 1) * hz_points / samplerate).astype(int)

    filterbank = np.zeros((n_filters, int(n_fft / 2 + 1)))
    for m in range(1, n_filters + 1):
        f_m_minus, f_m, f_m_plus = bins[m - 1], bins[m], bins[m + 1]
        for k in range(f_m_minus, f_m):
            filterbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus + 1e-8)
        for k in range(f_m, f_m_plus):
            filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m + 1e-8)

    feat = np.dot(pow_frames, filterbank.T)
    feat = np.where(feat == 0, np.finfo(float).eps, feat)
    log_feat = np.log(feat)
    static = scipy.fftpack.dct(log_feat, type=2, axis=1, norm='ortho')[:, :n_ceps]
    
    if not with_delta:
        return static
        
    d1 = delta(static)
    d2 = delta(d1)
    
    # Stack to [3, time, n_ceps]
    feat_3ch = np.stack([static, d1, d2], axis=0) # [3, T, F]
    return feat_3ch


def extract_phase_derivative(signal, sample_rate=16000, n_fft=512, win_len=0.025, win_step=0.01):
    """Extract phase-based features for A17 detection"""
    frame_len = int(win_len * sample_rate)
    frame_step = int(win_step * sample_rate)
    
    # STFT for phase extraction
    f, t, Zxx = scipy.signal.stft(signal, fs=sample_rate, 
                                   nperseg=frame_len, 
                                   noverlap=frame_len-frame_step)
    
    # Instantaneous phase
    phase = np.angle(Zxx)
    
    # Phase derivative (group delay)
    phase_delta = np.diff(phase, axis=1, prepend=phase[:, :1])
    
    # Unwrap and normalize
    phase_delta = np.unwrap(phase_delta, axis=1)
    phase_delta = (phase_delta - phase_delta.mean()) / (phase_delta.std() + 1e-8)
    
    return phase_delta.T  # [time, freq]


class ASVspoof2019LA(Dataset):

    def __init__(self, audio_dir, protocol_path, partition='train',
                 max_len=400, n_ceps=70, sample_rate=16000):

        self.audio_dir = Path(audio_dir)
        self.max_len = max_len
        self.n_ceps = n_ceps
        self.sample_rate = sample_rate
        self.samples = []

        with open(protocol_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    speaker_id = parts[0]
                    audio_id = parts[1]
                    system_id = parts[3]  
                    key = parts[4]  

                    label = 1 if key == 'bonafide' else 0
                    audio_path = self.audio_dir / f"{audio_id}.flac"

                    if audio_path.exists():
                        self.samples.append({
                            'path': audio_path,
                            'label': label,
                            'speaker_id': speaker_id,
                            'system_id': system_id,
                            'audio_id': audio_id
                        })

        bonafide_count = sum(1 for s in self.samples if s['label'] == 1)
        spoof_count = len(self.samples) - bonafide_count
        print(f"[{partition.upper()}] Loaded {len(self.samples)} samples: "
              f"{bonafide_count} bonafide, {spoof_count} spoof")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample = self.samples[idx]

        try:
            signal, sr = sf.read(str(sample['path']))

            if sr != self.sample_rate:
                import librosa
                signal = librosa.resample(signal, orig_sr=sr, target_sr=self.sample_rate)

            lfcc = extract_lfcc(signal, samplerate=self.sample_rate, n_ceps=self.n_ceps) # [3, time, n_ceps]
            
            # --- NEW: Phase Derivative Features ---
            phase_feats = extract_phase_derivative(signal, sample_rate=self.sample_rate)
            
            # Resize phase to match LFCC temporal dimension
            if phase_feats.shape[0] != lfcc.shape[1]:
                from scipy.interpolate import interp1d
                old_time = np.linspace(0, 1, phase_feats.shape[0])
                new_time = np.linspace(0, 1, lfcc.shape[1])
                # Check for empty phase features to avoid errors
                if phase_feats.shape[0] > 1:
                     interp = interp1d(old_time, phase_feats, axis=0, kind='linear')
                     phase_feats = interp(new_time)
                else:
                     # Edge case fallback
                     phase_feats = np.resize(phase_feats, (lfcc.shape[1], phase_feats.shape[1]))

            # --- NEW: Phase Feature Representation V2 ---
            # Bin phase features to match n_ceps dimension
            if phase_feats.shape[1] != self.n_ceps:
                from scipy.ndimage import zoom
                zoom_factor = self.n_ceps / phase_feats.shape[1]
                # zoom takes (zoom_axis0, zoom_axis1)
                phase_binned = zoom(phase_feats, (1, zoom_factor), order=1)  # [time, 70]
            else:
                phase_binned = phase_feats

            # Normalize per-bin
            phase_binned = (phase_binned - phase_binned.mean(axis=0, keepdims=True)) / \
                           (phase_binned.std(axis=0, keepdims=True) + 1e-8)

            phase_channel = phase_binned[np.newaxis, :, :]  # [1, time, 70]
            
            # Concatenate
            lfcc_4ch = np.concatenate([lfcc, phase_channel], axis=0) # [4, time, n_ceps]
            
            # Pad/Crop
            if lfcc_4ch.shape[1] < self.max_len:
                pad_width = self.max_len - lfcc_4ch.shape[1]
                lfcc_4ch = np.pad(lfcc_4ch, ((0, 0), (0, pad_width), (0, 0)), mode='constant')
            else:
                lfcc_4ch = lfcc_4ch[:, :self.max_len, :]

            lfcc = torch.tensor(lfcc_4ch, dtype=torch.float32) # [4, 400, 70]
            label = torch.tensor(sample['label'], dtype=torch.long)

            return lfcc, label, sample['audio_id']

        except Exception as e:
            print(f"[ERROR] Loading {sample['audio_id']}: {e}")
            print(f"[ERROR] Loading {sample['audio_id']}: {e}")
            lfcc = torch.zeros(4, self.max_len, self.n_ceps)
            label = torch.tensor(sample['label'], dtype=torch.long)
            return lfcc, label, sample['audio_id']


class SEBlock(nn.Module):

    def __init__(self, channels, reduction=16):

        super(SEBlock, self).__init__()

        self.squeeze = nn.AdaptiveAvgPool2d(1)
        self.excitation = nn.Sequential(
            nn.Linear(channels, channels // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channels // reduction, channels, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        batch, channels, _, _ = x.size()

        y = self.squeeze(x).view(batch, channels)

        y = self.excitation(y).view(batch, channels, 1, 1)

        return x * y.expand_as(x)


class SEBasicBlock(nn.Module):
    """
    ResNet BasicBlock with SE attention.
    """
    expansion = 1

    def __init__(self, in_planes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.se = SEBlock(planes, reduction)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class SEBottleneck(nn.Module):
    """
    ResNet Bottleneck with SE attention (for ResNet-50/101/152).
    """
    expansion = 4

    def __init__(self, in_planes, planes, stride=1, downsample=None, reduction=16):
        super(SEBottleneck, self).__init__()

        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)

        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride,
                               padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)

        self.conv3 = nn.Conv2d(planes, planes * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * self.expansion)

        self.relu = nn.ReLU(inplace=True)
        self.se = SEBlock(planes * self.expansion, reduction)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out = self.se(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

class TemporalAttention(nn.Module):
    """Captures temporal artifacts (A17 clicks, A18 waveform edits)"""
    def __init__(self, in_channels, reduction=4):
        super().__init__()
        self.temporal_squeeze = nn.AdaptiveAvgPool2d((None, 1))  # Pool freq dim
        
        # 1D temporal convolution with large receptive field
        self.temporal_conv = nn.Sequential(
            nn.Conv1d(in_channels, in_channels // reduction, 
                     kernel_size=15, padding=7, groups=1),
            nn.ReLU(inplace=True),
            nn.Conv1d(in_channels // reduction, in_channels, 
                     kernel_size=15, padding=7, groups=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [B, C, T, F]
        B, C, T, F = x.size()
        
        # Squeeze frequency dimension
        y = self.temporal_squeeze(x).squeeze(-1)  # [B, C, T]
        
        # Temporal attention
        att = self.temporal_conv(y)  # [B, C, T]
        att = att.unsqueeze(-1)  # [B, C, T, 1]
        
        return x * att.expand_as(x)

class SpectralAttention(nn.Module):
    def __init__(self, in_channels):
        super().__init__()
        self.conv_squeeze = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // 4, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // 4, in_channels, kernel_size=1),
            nn.Sigmoid()
        )
    
    def forward(self, x):
        # x: [B, C, T, F]
        # Average pool across time dimension
        freq_att = x.mean(dim=2, keepdim=True)  # [B, C, 1, F]
        freq_att = self.conv_squeeze(freq_att)  # [B, C, 1, F]
        return x * freq_att  # [B, C, T, F] * [B, C, 1, F]



class ResNetSE(nn.Module):


    def __init__(self, block, layers, num_classes=2, in_channels=1,
                 reduction=16, base_channels=64):

        super(ResNetSE, self).__init__()

        self.inplanes = base_channels
        self.reduction = reduction

        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7,
                               stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # NEW: Add temporal attention after conv1
        self.temporal_attn = TemporalAttention(base_channels, reduction=4)
        self.spectral_attn = SpectralAttention(base_channels)  # No freq_dim argument
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, base_channels, layers[0])
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.dropout = nn.Dropout(0.2)  # reduce overfitting to dev, help A17 / eval generalization
        self.fc = nn.Linear(base_channels * 8 * block.expansion, num_classes)

        self._init_weights()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None

        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion,
                          kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.reduction))
        self.inplanes = planes * block.expansion

        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, reduction=self.reduction))

        return nn.Sequential(*layers)

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        # NEW: Apply temporal attention
        x = self.temporal_attn(x)
        x = self.spectral_attn(x)
        
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.dropout(x)
        x = self.fc(x)

        return x


def resnet18_se(num_classes=2, **kwargs):
    """ResNet-18 with SE blocks."""
    return ResNetSE(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

def resnet34_se(num_classes=2, **kwargs):
    """ResNet-34 with SE blocks."""
    return ResNetSE(SEBasicBlock, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def resnet50_se(num_classes=2, **kwargs):
    """ResNet-50 with SE blocks."""
    return ResNetSE(SEBottleneck, [3, 4, 6, 3], num_classes=num_classes, **kwargs)

def resnet101_se(num_classes=2, **kwargs):
    """ResNet-101 with SE blocks."""
    return ResNetSE(SEBottleneck, [3, 4, 23, 3], num_classes=num_classes, **kwargs)


DATA_ROOT = Path(r"N:\ASVspoof2019\LA")

TRAIN_AUDIO_DIR = DATA_ROOT / "ASVspoof2019_LA_train" / "flac"
DEV_AUDIO_DIR = DATA_ROOT / "ASVspoof2019_LA_dev" / "flac"
EVAL_AUDIO_DIR = DATA_ROOT / "ASVspoof2019_LA_eval" / "flac"

PROTOCOL_DIR = DATA_ROOT / "ASVspoof2019_LA_cm_protocols"
TRAIN_PROTOCOL = PROTOCOL_DIR / "ASVspoof2019.LA.cm.train.trn.txt"
DEV_PROTOCOL = PROTOCOL_DIR / "ASVspoof2019.LA.cm.dev.trl.txt"
EVAL_PROTOCOL = PROTOCOL_DIR / "ASVspoof2019.LA.cm.eval.trl.txt"

BATCH_SIZE = 32
NUM_EPOCHS = 30
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-4
MAX_LEN = 400
N_CEPS = 70
CHANNELS = 4
USE_RESNET34 = False  # set True for more capacity (may help A17 / eval); if True, use resnet34_se in Eval script too


WEIGHTS_DIR = Path(r"n:\AudioForgeryDetection\Frontend\ResNet\ResNet-SEWeights_v4")
WEIGHTS_DIR.mkdir(parents=True, exist_ok=True)
SAVE_PATH = WEIGHTS_DIR / "resnet_se_asvspoof2019_v4.pth"
SAVE_PATH_BEST_TDCF = WEIGHTS_DIR / "resnet_se_best_tdcf.pth"
HISTORY_PATH = WEIGHTS_DIR / "history_v4.json"


class FocalLoss(nn.Module):
    def __init__(self, alpha=0.25, gamma=2.0, weight=None):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.weight = weight
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction='none', weight=self.weight)
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean()


def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    eer = fpr[eer_idx] * 100
    eer_threshold = thresholds[eer_idx]
    return eer, eer_threshold


def train_epoch(model, loader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    pbar = tqdm(loader, desc="Training")
    
    # SpecAugment: stronger augmentation for better eval / A17 generalization
    freq_masking_max_width = 14
    time_masking_max_width = 40
    num_freq_masks = 2
    num_time_masks = 2
    
    for inputs, labels, _ in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        
        # Apply SpecAugment (on the fly) - multiple masks
        if model.training:
            for _ in range(num_freq_masks):
                f = np.random.randint(1, freq_masking_max_width + 1)
                f0 = np.random.randint(0, max(1, N_CEPS - f))
                inputs[:, :, :, f0:f0+f] = 0
            for _ in range(num_time_masks):
                t = np.random.randint(1, time_masking_max_width + 1)
                t0 = np.random.randint(0, max(1, MAX_LEN - t))
                inputs[:, :, t0:t0+t, :] = 0

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        pbar.set_postfix({'loss': loss.item(), 'acc': 100. * correct / total})
    return running_loss / total, 100. * correct / total


@torch.no_grad()
def evaluate(model, loader, criterion, device, asv_metrics=None, epoch=-1, final_eval=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    all_labels = []
    all_scores = []
    all_audio_ids = []
    
    pbar = tqdm(loader, desc="Evaluating")
    for inputs, labels, audio_ids in pbar:
        inputs, labels = inputs.to(device), labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        probs = torch.softmax(outputs, dim=1)[:, 0]
        all_scores.extend(probs.cpu().numpy())
        all_labels.extend(labels.cpu().numpy())
        all_audio_ids.extend(audio_ids)
    
    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    
    if epoch >= 0 or final_eval:
        print(f"\nEvaluation Debug - Epoch {epoch+1}" if not final_eval else "\nFinal Evaluation Debug")
        print(f"Total samples: {len(all_scores)}")
        print(f"Score Min: {all_scores.min():.8f}, Max: {all_scores.max():.8f}, Mean: {all_scores.mean():.8f}, Std: {all_scores.std():.8f}")
        print(f"Unique scores: {len(np.unique(all_scores))}")
        
        if all_scores.std() < 1e-8:
            print(f"CRITICAL: All scores identical = {all_scores[0]:.8f}")
        elif len(np.unique(all_scores)) < 10:
            print(f"WARNING: Very few unique scores ({len(np.unique(all_scores))})")
        
        bonafide_mask = all_labels == 1
        spoof_mask = all_labels == 0
        bonafide_scores = all_scores[bonafide_mask]
        spoof_scores = all_scores[spoof_mask]
        
        print(f"Bonafide: Mean={bonafide_scores.mean():.6f}, Std={bonafide_scores.std():.6f}, N={len(bonafide_scores)}")
        print(f"Spoof: Mean={spoof_scores.mean():.6f}, Std={spoof_scores.std():.6f}, N={len(spoof_scores)}")
        print(f"Separation: {abs(bonafide_scores.mean() - spoof_scores.mean()):.6f}")
        
        print(f"\nSample predictions:")
        for i in range(min(10, len(all_scores))):
            label_str = "Bonafide" if all_labels[i] == 1 else "Spoof"
            predicted_spoof = all_scores[i] > 0.5
            actual_spoof = all_labels[i] == 0
            correct_mark = "OK" if predicted_spoof == actual_spoof else "WRONG"
            print(f"  {i}: Score={all_scores[i]:.6f}, True={label_str}, {correct_mark}")
        
        bonafide_near_zero = np.sum(bonafide_scores < 0.1)
        bonafide_near_one = np.sum(bonafide_scores > 0.9)
        spoof_near_zero = np.sum(spoof_scores < 0.1)
        spoof_near_one = np.sum(spoof_scores > 0.9)
        
        print(f"Bonafide <0.1: {bonafide_near_zero}/{len(bonafide_scores)} ({100*bonafide_near_zero/len(bonafide_scores):.1f}%)")
        print(f"Bonafide >0.9: {bonafide_near_one}/{len(bonafide_scores)} ({100*bonafide_near_one/len(bonafide_scores):.1f}%)")
        print(f"Spoof <0.1: {spoof_near_zero}/{len(spoof_scores)} ({100*spoof_near_zero/len(spoof_scores):.1f}%)")
        print(f"Spoof >0.9: {spoof_near_one}/{len(spoof_scores)} ({100*spoof_near_one/len(spoof_scores):.1f}%)\n")

    eer, _ = compute_eer(all_labels, all_scores)
    
    min_tdcf = 1.0
    if asv_metrics and all_audio_ids:
        try:
            print(f"\n{'='*60}")
            print(f"DEBUG t-minDCF Calculation:")
            print(f"{'='*60}")
            print(f"Total CM scores: {len(all_scores)}")
            print(f"Total audio_ids: {len(all_audio_ids)}")
            print(f"Total ASV metrics: {len(asv_metrics)}")
            print(f"Unique audio_ids: {len(set(all_audio_ids))}")
            print(f"\nSample audio_ids from CM: {all_audio_ids[:5]}")
            print(f"Sample ASV keys: {list(asv_metrics.keys())[:5]}")
        
            matching = sum(1 for aid in all_audio_ids if aid in asv_metrics)
            print(f"\n✓ Matching IDs: {matching}/{len(all_audio_ids)} ({100*matching/len(all_audio_ids):.1f}%)")
            print(f"\n{'='*60}")
            print(f"ASV Score Distribution:")
            print(f"{'='*60}")
            sample_asv = []
            sample_keys = []
            for aid in all_audio_ids[:1000]:
                if aid in asv_metrics:
                    sample_asv.append(asv_metrics[aid]['score'])
                    sample_keys.append(asv_metrics[aid]['key'])
            if sample_asv:
                sample_asv = np.array(sample_asv)
                print(f"ASV scores (n={len(sample_asv)}):")
                print(f"  Min: {sample_asv.min():.6f}")
                print(f"  Max: {sample_asv.max():.6f}")
                print(f"  Mean: {sample_asv.mean():.6f}")
                print(f"  Std: {sample_asv.std():.6f}")
                print(f"  Unique values: {len(np.unique(sample_asv))}")
                from collections import Counter
                key_counts = Counter(sample_keys)
                print(f"\nKey distribution:")
                for key, count in key_counts.items():
                    print(f"  {key}: {count}")
                print(f"\nCM Score Stats:")
                print(f"  Min: {min(all_scores):.6f}")
                print(f"  Max: {max(all_scores):.6f}")
                print(f"  Mean: {np.mean(all_scores):.6f}")
            if matching == 0:
                print("\n❌ ZERO MATCHES! Check audio_id format mismatch!")
                print(f"  CM format: {type(all_audio_ids[0])}, example: '{all_audio_ids[0]}'")
                print(f"  ASV format: {type(list(asv_metrics.keys())[0])}, example: '{list(asv_metrics.keys())[0]}'")
            print(f"{'='*60}\n")
            min_tdcf = compute_min_tDCF(all_scores, all_audio_ids, asv_metrics)
        except Exception as e:
            print(f"Warning: t-minDCF calculation failed: {e}")
            min_tdcf = 1.0

    avg_loss = running_loss / total
    accuracy = 100. * correct / total

    return avg_loss, accuracy, eer, min_tdcf, all_scores, all_labels



if __name__ == '__main__':
    train_dataset = ASVspoof2019LA(audio_dir=TRAIN_AUDIO_DIR, protocol_path=TRAIN_PROTOCOL, partition='train', max_len=MAX_LEN, n_ceps=N_CEPS)
    dev_dataset = ASVspoof2019LA(audio_dir=DEV_AUDIO_DIR, protocol_path=DEV_PROTOCOL, partition='dev', max_len=MAX_LEN, n_ceps=N_CEPS)
    
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    dev_loader = DataLoader(dev_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    
    model = (resnet34_se if USE_RESNET34 else resnet18_se)(num_classes=2, in_channels=CHANNELS).to(device)
    
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    ASV_SCORES_FILE = DATA_ROOT / "ASVspoof2019_LA_asv_scores" / "ASVspoof2019.LA.asv.dev.gi.trl.scores.txt"
    ASV_PROTOCOL_FILE = DATA_ROOT / "ASVspoof2019_LA_asv_protocols" / "ASVspoof2019.LA.asv.dev.gi.trl.txt"
    print("Loading ASV metrics for Dev set...")
    asv_metrics = load_asv_metrics(ASV_SCORES_FILE, ASV_PROTOCOL_FILE)
    
    bonafide_count = sum(1 for s in train_dataset.samples if s['label'] == 1)
    spoof_count = len(train_dataset.samples) - bonafide_count
    total = len(train_dataset.samples)
    weight_spoof = total / (2 * spoof_count)
    weight_bonafide = total / (2 * bonafide_count)
    class_weights = torch.tensor([weight_spoof, weight_bonafide], dtype=torch.float32).to(device)
    print(f"Class weights - Spoof: {weight_spoof:.4f}, Bonafide: {weight_bonafide:.4f}")
    
    criterion = nn.CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
    optimizer = optim.AdamW(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
    scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=NUM_EPOCHS)
    
    print("\nPre-training sanity check:")
    model.eval()
    test_batch = next(iter(train_loader))
    test_inputs, test_labels = test_batch[0][:10].to(device), test_batch[1][:10]
    
    with torch.no_grad():
        outputs = model(test_inputs)
        probs = torch.softmax(outputs, dim=1)
    
    print(f"Inputs: {test_inputs.shape}, Outputs: {outputs.shape}")
    for i in range(10):
        print(f"Sample {i}: logits=[{outputs[i,0].item():7.4f}, {outputs[i,1].item():7.4f}], probs=[{probs[i,0].item():.4f}, {probs[i,1].item():.4f}], label={test_labels[i].item()}")
    
    spoof_scores = probs[:, 1].cpu().numpy()
    print(f"Spoof scores: {spoof_scores}")
    print(f"Unique: {len(np.unique(spoof_scores))}, Std: {spoof_scores.std():.8f}")
    
    if spoof_scores.std() < 1e-6:
        print("CRITICAL: Model outputs are constant before training")
    else:
        print("Model initialization OK\n")
    
    best_eer = 100.0
    best_loss = float('inf')
    best_tdcf = float('inf')  # lower t-minDCF is better; baseline = 1.0
    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': [], 'val_eer': [], 'val_tdcf': []}
    
    for epoch in range(NUM_EPOCHS):
        print(f"\nEpoch {epoch + 1}/{NUM_EPOCHS}")
        print("-" * 40)
        
        train_loss, train_acc = train_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_acc, val_eer, val_tdcf, val_scores, val_labels = evaluate(model, dev_loader, criterion, device, asv_metrics, epoch=epoch)
        
        scheduler.step()
        
        print(f"Train Loss: {train_loss:.4f} | Acc: {train_acc:.2f}%")
        print(f"Val   Loss: {val_loss:.4f} | Acc: {val_acc:.2f}% | EER: {val_eer:.6f}% | t-minDCF: {val_tdcf:.6f}")
        if val_tdcf < 1.0:
            print(f"  -> t-minDCF beat baseline (1.0)")
        if val_tdcf < best_tdcf:
            best_tdcf = val_tdcf
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'val_tdcf': best_tdcf, 'val_eer': val_eer}, SAVE_PATH_BEST_TDCF)
            print(f"  -> New best t-minDCF: {best_tdcf:.6f} (saved to {SAVE_PATH_BEST_TDCF.name})")
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        history['val_eer'].append(val_eer)
        history['val_tdcf'].append(val_tdcf)
        
        if val_eer < best_eer or (abs(val_eer - best_eer) < 1e-7 and val_loss < best_loss):
            best_eer = val_eer
            best_loss = val_loss
            torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'best_eer': best_eer, 'best_loss': best_loss}, SAVE_PATH)
            print(f"Saved best model at Epoch {epoch+1} (EER: {best_eer:.6f}%, Loss: {best_loss:.4f})")
        
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_eer': val_eer, 'val_tdcf': val_tdcf}, WEIGHTS_DIR / f"Epoch{epoch+1}.pth")
        torch.save({'epoch': epoch, 'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict(), 'val_eer': val_eer}, WEIGHTS_DIR / "resnet_se_latest.pth")
    
    print(f"\nTraining complete. Best EER: {best_eer:.6f}% | Best t-minDCF: {best_tdcf:.6f}")
    print(f"  LA baselines (ASVspoof2019): LFCC-GMM=0.0663, CQCC-GMM=0.0123 (lower=better)")
    
    fig, axes = plt.subplots(1, 4, figsize=(20, 4))
    axes[0].plot(history['train_loss'], label='Train')
    axes[0].plot(history['val_loss'], label='Val')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Loss Curve')
    axes[0].legend()
    axes[0].grid(True)
    
    axes[1].plot(history['train_acc'], label='Train')
    axes[1].plot(history['val_acc'], label='Val')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Accuracy Curve')
    axes[1].legend()
    axes[1].grid(True)
    
    axes[2].plot(history['val_eer'])
    axes[2].set_xlabel('Epoch')
    axes[2].set_ylabel('EER (%)')
    axes[2].set_title('Validation EER')
    axes[2].grid(True)
    
    axes[3].plot(history['val_tdcf'])
    axes[3].set_xlabel('Epoch')
    axes[3].set_ylabel('t-minDCF')
    axes[3].set_title('Validation t-minDCF')
    axes[3].grid(True)
    
    plt.tight_layout()
    plt.show()
    
    checkpoint = torch.load(SAVE_PATH, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    
    val_loss, val_acc, val_eer, val_tdcf, scores, labels = evaluate(model, dev_loader, criterion, device, asv_metrics, epoch=-1, final_eval=True)
    
    print(f"\nFINAL RESULTS (Best Model)")
    print(f"Validation Loss: {val_loss:.4f}")
    print(f"Validation Accuracy: {val_acc:.2f}%")
    print(f"Validation EER: {val_eer:.6f}%")
    print(f"Validation t-minDCF: {val_tdcf:.6f}")
    
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, label=f'ROC curve (AUC = {roc_auc:.3f})')
    plt.plot([0, 1], [0, 1], 'k--')
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve - ResNet-SE ASVspoof 2019 LA')
    plt.legend(loc='lower right')
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    plt.figure(figsize=(10, 5))
    plt.hist(scores[labels == 1], bins=100, alpha=0.6, label='Bonafide', color='green')
    plt.hist(scores[labels == 0], bins=100, alpha=0.6, label='Spoof', color='red')
    plt.xlabel('Score')
    plt.ylabel('Count')
    plt.title('Score Distribution - Bonafide vs Spoof')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()
    
    from scipy.stats import norm
    fnr = 1 - tpr
    
    plt.figure(figsize=(8, 6))
    plt.plot(norm.ppf(fpr), norm.ppf(fnr), 'b-', linewidth=2)
    plt.xlabel('False Positive Rate (%)')
    plt.ylabel('False Negative Rate (%)')
    plt.title('DET Curve - ResNet-SE ASVspoof 2019 LA')
    
    ticks = [0.001, 0.01, 0.05, 0.1, 0.2, 0.4]
    tick_labels = [f'{t*100:.1f}' for t in ticks]
    tick_locs = [norm.ppf(t) for t in ticks]
    plt.xticks(tick_locs, tick_labels)
    plt.yticks(tick_locs, tick_labels)
    
    plt.grid(True)
    plt.tight_layout()
    plt.show()
