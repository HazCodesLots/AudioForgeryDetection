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
import os
import json

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
                    
                    label = 1 if label_str == 'bonafide' else 0
                    file_path = self.data_dir / f"{file_name}.flac"
                    
                    if file_path.exists():
                        self.samples.append({
                            'path': str(file_path),
                            'label': label
                        })
        
        print(f"Loaded {len(self.samples)} samples")

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
            
            peak = torch.max(torch.abs(waveform))
            if peak > 0:
                waveform = waveform / peak
            
            if waveform.shape[1] < self.max_length:
                waveform = F.pad(waveform, (0, self.max_length - waveform.shape[1]))
            else:
                waveform = waveform[:, :self.max_length]
                
        except Exception as e:
            waveform = torch.zeros(1, self.max_length)
            
        return waveform, label

def calculate_eer(labels, scores):
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    return eer * 100

def evaluate_lovo_on_asvspoof(protocol_path, data_path, device='cuda'):
    device = torch.device(device if torch.cuda.is_available() else 'cpu')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    results_json = os.path.join(script_dir, 'weights', 'protocol', 'lovo_results.json')
    with open(results_json, 'r') as f:
        lovo_results = json.load(f)
    
    vocoders = sorted(lovo_results.keys(), key=lambda x: lovo_results[x]['fold'])
    

    dataset = ASVspoof2019Dataset(protocol_path, data_path)
    loader = DataLoader(dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
    
    cross_results = {}

    print(f"\nEvaluating 10 LOVO Models on ASVspoof 2019...")
    
    for held_out in vocoders:
        fold_id = lovo_results[held_out]['fold']
        weight_path = os.path.join(script_dir, 'weights', 'protocol', f'lovo_{held_out}', f'lovo_{held_out}_best.pt')
        
        if not os.path.exists(weight_path):
            print(f"Skipping Fold {fold_id} ({held_out}): Weights not found.")
            continue
            
        print(f"\n--- [FOLD {fold_id}/10] Model (Held-out: {held_out}) ---")
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)

        lfcc_extractor = LFCCExtractor(n_lfcc=60, n_filter=60).to(device)
        lfcc_extractor.load_state_dict(checkpoint['lfcc_extractor_state_dict'])
        lfcc_extractor.eval()
        
        model = LCNN(n_lfcc=60, num_classes=2).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        all_labels = []
        all_scores = []
        
        with torch.no_grad():
            for waveforms, labels in tqdm(loader, desc=f"Eval Fold {fold_id}", leave=False):
                waveforms = waveforms.to(device)
                lfcc = lfcc_extractor(waveforms.squeeze(1))
                lfcc = lfcc.unsqueeze(1)
                outputs = model(lfcc)
                probs = F.softmax(outputs, dim=1)
                
                all_labels.extend(labels.numpy())
                all_scores.extend(probs[:, 1].cpu().numpy())
        
        all_labels = np.array(all_labels)
        all_scores = np.array(all_scores)
        
        acc = accuracy_score(all_labels, (all_scores > 0.5).astype(int))
        eer = calculate_eer(all_labels, all_scores)
        
        cross_results[held_out] = {'eer': eer, 'acc': acc}
        print(f"  Result -> EER: {eer:.4f}%, Acc: {acc*100:.2f}%")


    print("ASVSPOOF 2019 CROSS-DATASET LOVO PERFORMANCE")
    print(f"{'Held-out Vocoder (Model)':<40} {'EER (%)':<12} {'Acc (%)':<10}")
    
    avg_eer = []
    for held_out in vocoders:
        r = cross_results.get(held_out)
        if r:
            print(f"{held_out[0:38]:<40} {r['eer']:>10.4f}%  {r['acc']*100:>8.2f}%")
            avg_eer.append(r['eer'])
            
    final_avg = np.mean(avg_eer)
    print(f"{'AVERAGE CROSS-DATASET EER':<40} {final_avg:>10.4f}%")

    output_json = os.path.join(script_dir, 'weights', 'protocol', 'asvspoof_cross_lovo_results.json')
    save_data = {
        "summary": {
            "average_eer": float(final_avg),
            "num_folds": len(avg_eer)
        },
        "folds": cross_results
    }
    with open(output_json, 'w') as f:
        json.dump(save_data, f, indent=4)
    print(f"\n✓ Cross-dataset results saved to: {output_json}")

if __name__ == "__main__":
    PROTOCOL = r"N:\ASVspoof2019\LA\ASVspoof2019_LA_cm_protocols\ASVspoof2019.LA.cm.dev.trl.txt"
    DATA_DIR = r"N:\ASVspoof2019\LA\ASVspoof2019_LA_dev\flac"
    
    evaluate_lovo_on_asvspoof(PROTOCOL, DATA_DIR)
