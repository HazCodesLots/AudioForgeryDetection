"""
Modified loader that uses pre-defined splits from JSON
Includes Peak Normalization and Minimal Gaussian Noise safeguards
"""

import torch
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, DataLoader
import json
import soundfile as sf
import librosa
from pathlib import Path

class WaveFakeDatasetFixed(Dataset):
    """
    WaveFake dataset using fixed pre-defined splits
    """
    def __init__(
        self,
        splits_json: str,
        split_type: str = 'train',
        vocoders_to_include: list = None,
        include_real: bool = True,
        sample_rate: int = 16000,
        max_length: int = 64000,
        lfcc_extractor = None,
        noise_std: float = 0.001,
        root_dir: str = None
    ):
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.lfcc_extractor = lfcc_extractor
        self.split_type = split_type
        self.noise_std = noise_std
        self.root_dir = root_dir
        self.consecutive_errors = 0
        self.max_consecutive_errors = 50
        self.error_warned = False
        
        with open(splits_json, 'r') as f:
            self.splits = json.load(f)

        self.samples = []
        
        vocoder_names = vocoders_to_include if vocoders_to_include else list(self.splits['vocoders'].keys())
        
        for vocoder_name in vocoder_names:
            if vocoder_name not in self.splits['vocoders']:
                print(f"Warning: {vocoder_name} not in splits")
                continue
            
            if split_type == 'all':
                files = self.splits['vocoders'][vocoder_name]['train'] + self.splits['vocoders'][vocoder_name]['test']
            else:
                files = self.splits['vocoders'][vocoder_name][split_type]
            label = self.splits['vocoders'][vocoder_name]['label']
            
            for f in files:
                self.samples.append({
                    'path': f,
                    'label': label,
                    'vocoder': vocoder_name
                })
        
        if include_real and 'real' in self.splits:
            if split_type == 'all':
                files = self.splits['real']['train'] + self.splits['real']['test']
            else:
                files = self.splits['real'][split_type]
            label = self.splits['real']['label']
            
            for f in files:
                self.samples.append({
                    'path': f,
                    'label': label,
                    'vocoder': 'real'
                })
        
        print(f"LOADED {split_type.upper()} set: {len(self.samples)} samples")
        self.real_count = sum(1 for s in self.samples if s['label'] == 1)
        self.fake_count = sum(1 for s in self.samples if s['label'] == 0)
        print(f"  Real: {self.real_count}, Fake: {self.fake_count}")
    
    def __len__(self):
        return len(self.samples)
    
    def __getitem__(self, idx):
        sample = self.samples[idx]
        audio_path = sample['path']
        label = sample['label']
        
        try:
            if self.root_dir:
                if 'generated_audio' in audio_path:
                    relative_path = audio_path.split('generated_audio')[-1].lstrip('\\/')
                    audio_path = os.path.join(self.root_dir, relative_path)
            
            waveform, sr = sf.read(audio_path)
            
            if waveform.ndim > 1:
                waveform = np.mean(waveform, axis=1)
            
            if sr != self.sample_rate:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
            
            waveform = torch.from_numpy(waveform).float().unsqueeze(0)
            self.consecutive_errors = 0
            
        except Exception as e:
            self.consecutive_errors += 1
            if not self.error_warned:
                print(f"\n[WaveFakeLoader] WARNING: Error loading {audio_path}: {e}")
                print(f"[WaveFakeLoader] (Further errors will be silenced until {self.max_consecutive_errors} failures)")
                self.error_warned = True
            
            if self.consecutive_errors >= self.max_consecutive_errors:
                raise RuntimeError(f"FATAL: Dataset appears missing or inaccessible ({self.consecutive_errors} consecutive failures). last path: {audio_path}")
                
            waveform = torch.zeros(1, self.max_length)
        
        peak = torch.max(torch.abs(waveform))
        if peak > 0:
            waveform = waveform / peak
            
        if self.noise_std > 0:
            noise = torch.randn_like(waveform) * self.noise_std
            waveform = waveform + noise

        if waveform.shape[1] < self.max_length:
            waveform = F.pad(waveform, (0, self.max_length - waveform.shape[1]))
        else:
            waveform = waveform[:, :self.max_length]
        
        if self.lfcc_extractor is not None:
            with torch.no_grad():
                lfcc = self.lfcc_extractor(waveform)
                lfcc = lfcc.squeeze(0)
        else:
            lfcc = waveform
        
        return lfcc, label

def create_loaders_from_splits(
    splits_json: str,
    vocoders_train: list = None,
    vocoders_test: list = None,
    batch_size: int = 64,
    num_workers: int = 4,
    noise_std: float = 0.001,
    root_dir: str = None
):
    """
    Create train/test loaders from fixed splits
    """
    from feature_extraction import LFCCExtractor
    
    lfcc_extractor = LFCCExtractor(
        sample_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_lfcc=60,
        n_filter=60
    )
    
    train_dataset = WaveFakeDatasetFixed(
        splits_json=splits_json,
        split_type='train',
        vocoders_to_include=vocoders_train,
        include_real=True,
        lfcc_extractor=lfcc_extractor,
        noise_std=noise_std,
        root_dir=root_dir
    )
    
    test_dataset = WaveFakeDatasetFixed(
        splits_json=splits_json,
        split_type='test',
        vocoders_to_include=vocoders_test,
        include_real=True,
        lfcc_extractor=lfcc_extractor,
        noise_std=0.0,
        root_dir=root_dir
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True
    )
    
    return train_loader, test_loader
