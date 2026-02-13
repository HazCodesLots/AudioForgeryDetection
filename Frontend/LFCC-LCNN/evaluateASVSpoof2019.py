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
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d

class ASVspoof2019Dataset(Dataset):
    """
    Dataset loader for ASVspoof 2019 Logical Access (LA)
    """
    def __init__(self, protocol_file, data_dir, sample_rate=16000, max_length=64000):
        self.data_dir = Path(data_dir)
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.samples = []
        
        print(f"Loading protocol: {protocol_file}")
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    file_name = parts[1]
                    label_str = parts[4]
                    
                    # bonafide = 1 (Real), spoof = 0 (Fake)
                    label = 1 if label_str == 'bonafide' else 0
                    file_path = self.data_dir / f"{file_name}.flac"
                    
                    if file_path.exists():
                        self.samples.append({
                            'path': str(file_path),
                            'label': label
                        })
        
        print(f"Loaded {len(self.samples)} samples from {protocol_file}")
        self.real_count = sum(1 for s in self.samples if s['label'] == 1)
        self.fake_count = sum(1 for s in self.samples if s['label'] == 0)
        print(f"  Real: {self.real_count}, Fake: {self.fake_count}")

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        sample_info = self.samples[idx]
        audio_path = sample_info['path']
        label = sample_info['label']
        
        try:
            waveform, sr = sf.read(audio_path)
            if sr != self.sample_rate:
                waveform = librosa.resample(waveform, orig_sr=sr, target_sr=self.sample_rate)
            
            waveform = torch.from_numpy(waveform).float()
            if waveform.ndim == 1:
                waveform = waveform.unsqueeze(0)
            
            # Peak normalization to [-1, 1] — matches WaveFake training preprocessing
            peak = torch.max(torch.abs(waveform))
            if peak > 0:
                waveform = waveform / peak
            
            # Pad or truncate
            if waveform.shape[1] < self.max_length:
                waveform = F.pad(waveform, (0, self.max_length - waveform.shape[1]))
            else:
                waveform = waveform[:, :self.max_length]
                
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            waveform = torch.zeros(1, self.max_length)
            
        return waveform, label

def calculate_eer(labels, scores):
    """Calculate Equal Error Rate"""
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100, fpr, tpr, thresholds

def evaluate_asvspoof(model_path, protocol_path, data_path, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}\n")
    
    # Load checkpoint
    print(f"Loading model: {model_path}")
    checkpoint = torch.load(model_path, map_location=device)
    
    # Initialize components
    lfcc_extractor = LFCCExtractor(n_lfcc=60, n_filter=60)
    lfcc_extractor.load_state_dict(checkpoint['lfcc_extractor_state_dict'])
    lfcc_extractor.to(device)
    lfcc_extractor.eval()
    
    model = LCNN(n_lfcc=60, num_classes=2)
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    # Dataset
    dataset = ASVspoof2019Dataset(protocol_path, data_path)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4)
    
    all_labels = []
    all_scores = []
    
    print("Evaluating...")
    with torch.no_grad():
        for waveforms, labels in tqdm(loader):
            waveforms = waveforms.to(device)
            
            # Extract features (squeeze added to handle batch-channel dim)
            lfcc = lfcc_extractor(waveforms.squeeze(1)) # (batch, n_lfcc, time)
            lfcc = lfcc.unsqueeze(1) # Add channel dim: (batch, 1, n_lfcc, time)
            
            outputs = model(lfcc)
            probs = F.softmax(outputs, dim=1)
            
            all_labels.extend(labels.numpy())
            all_scores.extend(probs[:, 1].cpu().numpy()) # Probability of Real
            
    all_labels = np.array(all_labels)
    all_scores = np.array(all_scores)
    
    # Accuracy at 0.5 threshold
    all_preds = (all_scores > 0.5).astype(int)
    acc = accuracy_score(all_labels, all_preds)
    
    # AUC and EER
    auc_score = roc_auc_score(all_labels, all_scores)
    eer, _, _, _ = calculate_eer(all_labels, all_scores)
    
    print("\n" + "="*40)
    print("ASVSPOOF 2019 EVALUATION RESULTS")
    print("="*40)
    print(f"Accuracy: {acc*100:.2f}%")
    print(f"AUC:      {auc_score:.4f}")
    print(f"EER:      {eer:.4f}%")
    print("="*40)
    
    if acc > 0.95 and eer < 5.0:
        print("\nWARNING: Model generalizes perfectly. This is extremely rare for cross-dataset.")
    elif acc < 0.60:
        print("\nCONFIRMED: Model has low generalizability. High WaveFake results were likely LJSpeech environment bias.")
    else:
        print("\nRESULT: Partial generalizability observed.")

if __name__ == "__main__":
    # Settings for ASVspoof2019 LA Dev
    PROTOCOL = r"N:\ASVspoof2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"
    DATA_DIR = r"N:\ASVspoof2019\LA\ASVspoof2019_LA_dev\flac"
    
    import os
    script_dir = os.path.dirname(os.path.abspath(__file__))
    MODEL_PATH = os.path.join(script_dir, 'weights', 'epoch_075.pt')
    
    if not os.path.exists(MODEL_PATH):
        # Fallback to current folder if weights subfolder isn't there
        MODEL_PATH = os.path.join(script_dir, 'best_lfcc_lcnn_wavefake.pt')

    evaluate_asvspoof(MODEL_PATH, PROTOCOL, DATA_DIR)
