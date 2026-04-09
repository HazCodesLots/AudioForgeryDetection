import os
import numpy as np
import soundfile as sf
import librosa
from pathlib import Path
import torch
from torch.utils.data import Dataset

def extract_mfcc(signal, samplerate=16000, n_mfcc=20, winlen=0.025, winstep=0.01):
    hop_length = int(samplerate * winstep)
    win_length = int(samplerate * winlen)

    mfcc = librosa.feature.mfcc(
        y=signal,
        sr=samplerate,
        n_mfcc=n_mfcc,
        n_fft=512,
        hop_length=hop_length,
        win_length=win_length,
        window="hamming",
        center=False
    )
    mfcc = mfcc.T
    return mfcc

def process_odss_mfcc_style(dataset_path, output_path, sr=16000):
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    class_map = {
        "natural": "bonafide",
        "fastpitch-hifigan": "spoof",
        "vits": "spoof"
    }

    total = 0
    skipped = 0

    for subfolder, label in class_map.items():
        audio_dir = dataset_path / subfolder
        out_dir_base = output_path / label / subfolder

        for wav_file in audio_dir.rglob("*.wav"):
            try:
                signal, rate = sf.read(str(wav_file))
                if rate != sr:
                    print(f"[!] Skipping {wav_file.name}: Sample rate {rate} ≠ {sr}")
                    skipped += 1
                    continue

                mfcc_feat = extract_mfcc(signal, samplerate=sr)
                rel_path = wav_file.relative_to(audio_dir).with_suffix(".npy")
                out_file = out_dir_base / rel_path
                out_file.parent.mkdir(parents=True, exist_ok=True)

                np.save(out_file, mfcc_feat)
                total += 1
                print(f"[✓] Saved MFCC: {out_file}")
            except Exception as e:
                print(f"[X] Failed {wav_file.name}: {e}")
                skipped += 1

process_odss_mfcc_style(
    dataset_path=r"C:\Users\DaysPC\Documents\Datasets\odss",
    output_path=r"C:\Users\DaysPC\Documents\Datasets\odss_mfcc"
)


class ODSSFeatureDataset(Dataset):
    def __init__(self, root_dir, transform=None, max_len=400, expected_dim=20):
        self.root_dir = Path(root_dir)
        self.transform = transform
        self.max_len = max_len
        self.expected_dim = expected_dim
        self.samples = []

        for label_str in ['bonafide', 'spoof']:
            label = 1 if label_str == 'bonafide' else 0
            for npy_file in (self.root_dir / label_str).rglob("*.npy"):
                self.samples.append((npy_file, label))

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        path, label = self.samples[idx]

        try:
            feat = np.load(path)
        except Exception as e:
            print(f"[ERROR] Could not load {path.name}: {e}")
            return self.__getitem__((idx + 1) % len(self))

        if feat.ndim != 2 or feat.shape[1] != self.expected_dim:
            print(f"[WARN] Invalid feature shape: {path.name} -> {feat.shape}")
            return self.__getitem__((idx + 1) % len(self))

        if feat.shape[0] < self.max_len:
            feat = np.pad(feat, ((0, self.max_len - feat.shape[0]), (0, 0)), mode='constant')
        else:
            feat = feat[:self.max_len, :]

        tensor = torch.tensor(feat, dtype=torch.float32).unsqueeze(0)
        return tensor, torch.tensor(label, dtype=torch.long)
from torch.utils.data import DataLoader, random_split

dataset_path = r"C:\Users\DaysPC\Documents\Datasets\odss_mfcc"

dataset = ODSSFeatureDataset(
    root_dir=dataset_path,
    expected_dim=20
)

train_size = int(0.8 * len(dataset))
val_size = len(dataset) - train_size
train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

train_loader = DataLoader(train_dataset, batch_size=16, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)