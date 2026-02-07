import os
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
import scipy.signal
import scipy.fftpack
from pathlib import Path
from tqdm import tqdm
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
import json
import pandas as pd
from calculate_tDCF import load_asv_metrics, compute_min_tDCF

# --- Enhanced Feature Extraction (v2) ---

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
    hz_points = np.linspace(0, samplerate / 2, n_filters + 2)
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
    if not with_delta: return static
    d1 = delta(static)
    d2 = delta(d1)
    return np.stack([static, d1, d2], axis=0)


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

# --- Enhanced Model Architecture (v2) ---

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
    expansion = 1
    def __init__(self, in_planes, planes, stride=1, downsample=None, reduction=16):
        super(SEBasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
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
        if self.downsample is not None: identity = self.downsample(x)
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
    def __init__(self, block, layers, num_classes=2, in_channels=1, reduction=16, base_channels=64):
        super(ResNetSE, self).__init__()
        self.inplanes = base_channels
        self.reduction = reduction
        self.conv1 = nn.Conv2d(in_channels, base_channels, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(base_channels)
        self.relu = nn.ReLU(inplace=True)
        
        # NEW: Add temporal attention after conv1
        self.temporal_attn = TemporalAttention(base_channels, reduction=4)
        self.spectral_attn = SpectralAttention(base_channels)
        
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, base_channels, layers[0])
        self.layer2 = self._make_layer(block, base_channels * 2, layers[1], stride=2)
        self.layer3 = self._make_layer(block, base_channels * 4, layers[2], stride=2)
        self.layer4 = self._make_layer(block, base_channels * 8, layers[3], stride=2)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(base_channels * 8 * block.expansion, num_classes)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )
        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.reduction))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks): layers.append(block(self.inplanes, planes, reduction=self.reduction))
        return nn.Sequential(*layers)
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        
        x = self.temporal_attn(x)
        x = self.spectral_attn(x)
        
        x = self.maxpool(x)
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

def resnet18_se(num_classes=2, **kwargs):
    return ResNetSE(SEBasicBlock, [2, 2, 2, 2], num_classes=num_classes, **kwargs)

class ASVspoof2019Eval(Dataset):
    def __init__(self, audio_dir, protocol_path, max_len=400, n_ceps=70, sample_rate=16000):
        self.audio_dir = Path(audio_dir)
        self.max_len = max_len
        self.n_ceps = n_ceps
        self.sample_rate = sample_rate
        self.samples = []
        with open(protocol_path, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    audio_id, attack_type, key = parts[1], parts[3], parts[4]
                    label = 1 if key == 'bonafide' else 0
                    audio_path = self.audio_dir / f"{audio_id}.flac"
                    if audio_path.exists():
                        self.samples.append({'path': audio_path, 'label': label, 'audio_id': audio_id, 'attack_type': attack_type})
        print(f"[EVAL] Loaded {len(self.samples)} samples.")

    def __len__(self): return len(self.samples)

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
                if phase_feats.shape[0] > 1:
                     interp = interp1d(old_time, phase_feats, axis=0, kind='linear')
                     phase_feats = interp(new_time)
                else:
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

            return torch.tensor(lfcc_4ch, dtype=torch.float32), torch.tensor(sample['label']), sample['attack_type'], sample['audio_id']
        except Exception as e:
            print(f"[ERROR] Loading {sample['audio_id']}: {e}")
            return torch.zeros(4, self.max_len, self.n_ceps), torch.tensor(sample['label']), sample['attack_type'], sample['audio_id']

def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=0)
    fnr = 1 - tpr
    eer_idx = np.nanargmin(np.absolute(fnr - fpr))
    return fpr[eer_idx] * 100

@torch.no_grad()
def run_evaluation():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def evaluate_epoch(epoch_path, model, device, eval_loader):
    if not os.path.exists(epoch_path):
        print(f"Skipping {epoch_path} (not found)")
        return None

    checkpoint = torch.load(epoch_path, map_location=device, weights_only=False)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.eval()
    eer_val = checkpoint.get('best_eer', checkpoint.get('val_eer', 0.0))
    print(f"\nLoaded model (Epoch {checkpoint['epoch']+1}, Dev EER: {eer_val:.2f}%)")

    results = []
    # Use standard loop without tqdm for cleaner output during multi-epoch eval
    with torch.no_grad():
        for inputs, labels, attack_types, audio_ids in tqdm(eval_loader, desc=f"Eval {epoch_path}"):
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)[:, 0].cpu().numpy()
            for i in range(len(probs)):
                results.append({'audio_id': audio_ids[i], 'attack_type': attack_types[i], 'label': labels[i].item(), 'score': probs[i]})
    
    return pd.DataFrame(results)

def run_evaluation():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    DATA_ROOT = Path(r"N:\ASVspoof2019\LA")
    EVAL_AUDIO_DIR = DATA_ROOT / "ASVspoof2019_LA_eval" / "flac"
    EVAL_PROTOCOL = DATA_ROOT / "ASVspoof2019_LA_cm_protocols" / "ASVspoof2019.LA.cm.eval.trl.txt"
    
    BATCH_SIZE, MAX_LEN, N_CEPS, CHANNELS = 64, 400, 70, 4
    
    WEIGHTS_DIR = Path(r"n:\AudioForgeryDetection\Frontend\ResNet\ResNet-SEWeights_v4")

    model = resnet18_se(num_classes=2, in_channels=CHANNELS).to(device)
    eval_dataset = ASVspoof2019Eval(EVAL_AUDIO_DIR, EVAL_PROTOCOL, max_len=MAX_LEN, n_ceps=N_CEPS)
    eval_loader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    # Load ASV Metrics for Eval set
    ASV_SCORES_FILE = DATA_ROOT / "ASVspoof2019_LA_asv_scores" / "ASVspoof2019.LA.asv.eval.gi.trl.scores.txt"
    ASV_PROTOCOL_FILE = DATA_ROOT / "ASVspoof2019_LA_asv_protocols" / "ASVspoof2019.LA.asv.eval.gi.trl.txt"
    print("Loading ASV metrics for Eval set...")
    asv_metrics = load_asv_metrics(ASV_SCORES_FILE, ASV_PROTOCOL_FILE)


    # Evaluate all epochs 1-30 in new weights directory
    epochs_to_test = [WEIGHTS_DIR / f"Epoch{i}.pth" for i in range(1, 31)]
    
    final_results = {}
    
    for epoch_file in epochs_to_test:
        df = evaluate_epoch(epoch_file, model, device, eval_loader)
        if df is None: continue

        global_eer = compute_eer(df['label'], df['score'])
        min_tdcf = compute_min_tDCF(df['score'].values, df['audio_id'].values, asv_metrics)
        print(f"GLOBAL EVAL EER ({epoch_file}): {global_eer:.4f}% | t-minDCF: {min_tdcf:.4f}")

        attack_groups = df.groupby('attack_type')
        bonafide_scores = df[df['label'] == 1]['score'].values
        
        attack_eers = {}
        for attack, group in attack_groups:
            if attack == '-': continue
            attack_scores = group['score'].values
            y_true = np.concatenate([np.ones(len(bonafide_scores)), np.zeros(len(attack_scores))])
            y_score = np.concatenate([bonafide_scores, attack_scores])
            attack_eer = compute_eer(y_true, y_score)
            attack_eers[attack] = attack_eer
            print(f"  {attack:<5} EER: {attack_eer:.4f}%")
        
        final_results[str(epoch_file)] = {
            "Global_EER": global_eer,
            "min_tDCF": min_tdcf,
            "Attack_EERs": attack_eers
        }
        print("-" * 40)
    
    # Best-by-eval-minDCF and best A17
    if final_results:
        best_tdcf_epoch = min(final_results.items(), key=lambda x: x[1]["min_tDCF"])
        best_a17 = None
        for path, data in final_results.items():
            ae = data.get("Attack_EERs", {})
            if "A17" in ae:
                if best_a17 is None or ae["A17"] < best_a17[1]:
                    best_a17 = (path, ae["A17"])
        print("\n" + "=" * 60)
        print("EVAL SUMMARY")
        print("=" * 60)
        print(f"Best eval t-minDCF: {best_tdcf_epoch[1]['min_tDCF']:.4f} @ {best_tdcf_epoch[0]}")
        print(f"  -> Use this checkpoint for best tandem performance.")
        if best_a17 is not None:
            print(f"Best A17 EER:       {best_a17[1]:.4f}% @ {best_a17[0]}")
        print("=" * 60)
        
    # Save to JSON
    with open("eval_results_all_epochs.json", "w") as f:
        json.dump(final_results, f, indent=4)
    print("Saved results to eval_results_all_epochs.json")

if __name__ == "__main__":
    run_evaluation()
