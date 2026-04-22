import torch
import torch.nn.functional as F
from feature_extraction import LFCCExtractor
from LCNN import LCNN
from torch.utils.data import DataLoader, Dataset
import numpy as np
from pathlib import Path
from tqdm import tqdm
import soundfile as sf
import librosa
from sklearn.metrics import accuracy_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d
import os
import pandas as pd

OFFICIAL_FOLDS = [
    "Fold1_MelGAN", "Fold2_MelGAN_Large", "Fold3_FB_MelGAN",
    "Fold4_MB_MelGAN", "Fold5_HiFiGAN", "Fold6_ParallelWaveGAN"
]

class InTheWildDataset(Dataset):
    def __init__(self, meta_file, data_dir, sample_rate=16000, max_length=64000):
        self.data_dir, self.sample_rate, self.max_length = Path(data_dir), sample_rate, max_length
        df = pd.read_csv(meta_file)
        self.samples = [{'path': str(self.data_dir / r['file']), 'label': 1 if r['label'] == 'bona-fide' else 0} 
                        for _, r in df.iterrows() if (self.data_dir / r['file']).exists()]

    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        sample = self.samples[idx]
        try:
            w, sr = sf.read(sample['path'])
            if sr != self.sample_rate: w = librosa.resample(w, orig_sr=sr, target_sr=self.sample_rate)
            w = torch.from_numpy(w).float()
            if w.ndim == 1: w = w.unsqueeze(0)
            p = torch.max(torch.abs(w))
            if p > 0: w /= p
            if w.shape[1] < self.max_length: w = F.pad(w, (0, self.max_length - w.shape[1]))
            else: w = w[:, :self.max_length]
        except: w = torch.zeros(1, self.max_length)
        return w, sample['label']

def calculate_eer(labels, scores):
    fpr, tpr, _ = roc_curve(labels, scores, pos_label=1)
    return brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.) * 100

def evaluate_on_itw(meta, data, device='cuda'):
    script_dir = os.path.dirname(os.path.abspath(__file__))
    dataset = InTheWildDataset(meta, data)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    eers = []
    # 1. Evaluate ID Baseline (The control group)
    id_weight = os.path.join(script_dir, 'weights', 'protocol', 'id_baseline', 'id_baseline_best.pt')
    if os.path.exists(id_weight):
        print("  Testing ID Baseline (Control)...")
        ckpt = torch.load(id_weight, map_location=device, weights_only=False)
        model = LCNN(n_lfcc=60, num_classes=2).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        ext = LFCCExtractor(n_lfcc=60, n_filter=60)
        ext.load_state_dict(ckpt['lfcc_extractor_state_dict'])
        ext.eval()
        y_true, y_score = [], []
        with torch.no_grad():
            for w, l in tqdm(loader, desc="ID_Baseline", leave=False):
                # Run extraction on CPU since extractor is on CPU
                lfcc = ext(w.squeeze(1)).unsqueeze(1).to(device)
                s = F.softmax(model(lfcc), dim=1)[:, 1].cpu().numpy()
                y_true.extend(l); y_score.extend(s)
        eer = calculate_eer(y_true, y_score)
        print(f"    ID Baseline EER: {eer:.4f}%")

    # 2. Evaluate LOO Folds
    for fold in OFFICIAL_FOLDS:
        weight = os.path.join(script_dir, 'weights', 'protocol', fold, f'{fold}_best.pt')
        if not os.path.exists(weight): continue
        print(f"  Testing {fold}...")
        ckpt = torch.load(weight, map_location=device, weights_only=False)
        model = LCNN(n_lfcc=60, num_classes=2).to(device)
        model.load_state_dict(ckpt['model_state_dict'])
        model.eval()
        ext = LFCCExtractor(n_lfcc=60, n_filter=60)
        ext.load_state_dict(ckpt['lfcc_extractor_state_dict'])
        ext.eval()
        y_true, y_score = [], []
        with torch.no_grad():
            for w, l in tqdm(loader, desc=fold, leave=False):
                # Run extraction on CPU since extractor is on CPU
                lfcc = ext(w.squeeze(1)).unsqueeze(1).to(device)
                s = F.softmax(model(lfcc), dim=1)[:, 1].cpu().numpy()
                y_true.extend(l); y_score.extend(s)
        eer = calculate_eer(y_true, y_score)
        eers.append(eer)
        print(f"    EER: {eer:.4f}%")
    if eers: print(f"Average ITW EER: {np.mean(eers):.4f}%")

if __name__ == "__main__":
    evaluate_on_itw(r"N:\release_in_the_wild\meta.csv", r"N:\release_in_the_wild")
