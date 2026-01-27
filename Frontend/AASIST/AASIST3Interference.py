import torch
import torch.nn.functional as F
from torch.utils.data import Dataset, DataLoader
import torchaudio
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
import argparse
import json
from sklearn.metrics import roc_curve
from scipy.optimize import brentq
from scipy.interpolate import interp1d


class PreEmphasis(torch.nn.Module):
    def __init__(self, coef=0.97):
        super(PreEmphasis, self).__init__()
        self.coef = coef
        self.register_buffer('flipped_filter', torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0))
    
    def forward(self, x):
        if x.dim() == 2:
            x = x.unsqueeze(1)
        x = F.pad(x, (1, 0), mode='reflect')
        x = F.conv1d(x, self.flipped_filter)
        return x.squeeze(1)


class ASVspoof5Dataset(Dataset):
    def __init__(self, data_root, partition='dev', track='open', protocol_file=None, sample_rate=16000, max_length=64000):
        self.data_root = Path(data_root)
        self.partition = partition
        self.track = track
        self.sample_rate = sample_rate
        self.max_length = max_length
        self.pre_emphasis = PreEmphasis(coef=0.97)
        
        if track == 'open':
            self.audio_dir = self.data_root / 'ASVspoof5' / 'Open' / partition / 'flac'
        else:
            self.audio_dir = self.data_root / 'ASVspoof5' / 'Closed' / partition / 'flac'
        
        if protocol_file is None:
            if track == 'open':
                protocol_file = self.data_root / 'ASVspoof5' / 'Open' / partition / f'ASVspoof5.Open.{partition}.metadata.txt'
            else:
                protocol_file = self.data_root / 'ASVspoof5' / 'Closed' / partition / f'ASVspoof5.Closed.{partition}.metadata.txt'
        
        self.protocol = self._load_protocol(protocol_file)
        
    def _load_protocol(self, protocol_file):
        protocol_file = Path(protocol_file)
        if not protocol_file.exists():
            raise FileNotFoundError(f"Protocol file not found: {protocol_file}")
        
        data = []
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 5:
                    speaker_id = parts[0]
                    file_id = parts[1]
                    label = parts[4]
                    attack_type = parts[3] if len(parts) > 3 else '-'
                    
                    data.append({
                        'speaker_id': speaker_id,
                        'file_id': file_id,
                        'label': 0 if label == 'bonafide' else 1,
                        'attack_type': attack_type,
                        'filepath': self.audio_dir / f"{file_id}.flac"
                    })
        
        return pd.DataFrame(data)
    
    def _load_audio(self, filepath):
        waveform, sr = torchaudio.load(filepath)
        
        if waveform.shape[0] > 1:
            waveform = waveform.mean(dim=0, keepdim=True)
        
        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)
        
        return waveform.squeeze(0)
    
    def _pad_or_truncate(self, audio):
        if len(audio) > self.max_length:
            start = (len(audio) - self.max_length) // 2
            audio = audio[start:start + self.max_length]
        elif len(audio) < self.max_length:
            padding = self.max_length - len(audio)
            audio = F.pad(audio, (0, padding))
        return audio
    
    def _normalize(self, audio):
        audio = audio - audio.mean()
        std = audio.std()
        if std > 0:
            audio = audio / std
        return torch.clamp(audio, -1, 1)
    
    def __len__(self):
        return len(self.protocol)
    
    def __getitem__(self, idx):
        row = self.protocol.iloc[idx]
        
        audio = self._load_audio(row['filepath'])
        audio = self._pad_or_truncate(audio)
        audio = self._normalize(audio)
        audio = self.pre_emphasis(audio.unsqueeze(0)).squeeze(0)
        
        return {
            'audio': audio,
            'label': row['label'],
            'file_id': row['file_id'],
            'speaker_id': row['speaker_id'],
            'attack_type': row['attack_type']
        }


class AASIST3Inference:
    def __init__(self, model_path, device='cuda', sample_rate=16000):
        self.device = device
        self.sample_rate = sample_rate
        
        checkpoint = torch.load(model_path, map_location=device)
        
        from aasist3_model import AASIST3
        self.model = AASIST3()
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        
    def compute_eer(self, labels, scores):
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr
        eer_threshold = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
        eer = 100 * (1 - interp1d(fpr, tpr)(eer_threshold))
        idx = np.nanargmin(np.abs(fnr - fpr))
        threshold = thresholds[idx]
        return eer, threshold
    
    def compute_min_dcf(self, labels, scores, p_target=0.05, c_miss=1, c_fa=1):
        fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
        fnr = 1 - tpr
        dcf = c_miss * fnr * p_target + c_fa * fpr * (1 - p_target)
        min_dcf = np.min(dcf)
        return min_dcf
    
    def predict_dataset(self, dataset, batch_size=32, num_workers=4):
        dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=num_workers)
        
        all_scores = []
        all_labels = []
        all_file_ids = []
        
        with torch.no_grad():
            for batch in tqdm(dataloader, desc="Inference"):
                audio = batch['audio'].to(self.device)
                labels = batch['label'].numpy()
                file_ids = batch['file_id']
                
                logits = self.model(audio)
                probs = F.softmax(logits, dim=1)
                scores = probs[:, 1].cpu().numpy()
                
                all_scores.extend(scores)
                all_labels.extend(labels)
                all_file_ids.extend(file_ids)
        
        return np.array(all_scores), np.array(all_labels), all_file_ids
    
    def evaluate(self, dataset, batch_size=32, num_workers=4):
        scores, labels, file_ids = self.predict_dataset(dataset, batch_size, num_workers)
        
        eer, eer_threshold = self.compute_eer(labels, scores)
        min_dcf = self.compute_min_dcf(labels, scores)
        
        predictions = (scores > eer_threshold).astype(int)
        accuracy = 100 * (predictions == labels).sum() / len(labels)
        
        results = {
            'eer': float(eer),
            'eer_threshold': float(eer_threshold),
            'min_dcf': float(min_dcf),
            'accuracy': float(accuracy),
            'num_samples': len(labels),
            'num_bonafide': int((labels == 0).sum()),
            'num_spoofed': int((labels == 1).sum())
        }
        
        return results, scores, labels, file_ids


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_root', type=str, required=True)
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--partition', type=str, default='dev', choices=['train', 'dev', 'eval'])
    parser.add_argument('--track', type=str, default='open', choices=['open', 'closed'])
    parser.add_argument('--batch_size', type=int, default=32)
    parser.add_argument('--num_workers', type=int, default=4)
    parser.add_argument('--device', type=str, default='cuda')
    parser.add_argument('--output_dir', type=str, default='results')
    parser.add_argument('--save_scores', action='store_true')
    args = parser.parse_args()
    
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print(f"Loading dataset: {args.track.upper()} track, {args.partition} partition")
    dataset = ASVspoof5Dataset(
        data_root=args.data_root,
        partition=args.partition,
        track=args.track,
        sample_rate=16000,
        max_length=64000
    )
    print(f"Dataset size: {len(dataset)}")
    
    print(f"Loading model from: {args.model_path}")
    inference = AASIST3Inference(
        model_path=args.model_path,
        device=args.device,
        sample_rate=16000
    )
    
    print("Running inference...")
    results, scores, labels, file_ids = inference.evaluate(
        dataset,
        batch_size=args.batch_size,
        num_workers=args.num_workers
    )
    
    print("EVALUATION RESULTS")
    print(f"Track:           {args.track.upper()}")
    print(f"Partition:       {args.partition}")
    print(f"Total samples:   {results['num_samples']}")
    print(f"Bonafide:        {results['num_bonafide']}")
    print(f"Spoofed:         {results['num_spoofed']}")
    print(f"EER:             {results['eer']:.2f}%")
    print(f"minDCF:          {results['min_dcf']:.4f}")
    print(f"Accuracy:        {results['accuracy']:.2f}%")
    print(f"EER Threshold:   {results['eer_threshold']:.4f}")
    
    results_file = output_dir / f"results_{args.track}_{args.partition}.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_file}")
    
    if args.save_scores:
        scores_df = pd.DataFrame({
            'file_id': file_ids,
            'label': labels,
            'score': scores,
            'prediction': (scores > results['eer_threshold']).astype(int)
        })
        scores_file = output_dir / f"scores_{args.track}_{args.partition}.csv"
        scores_df.to_csv(scores_file, index=False)
        print(f"Scores saved to: {scores_file}")


if __name__ == '__main__':
    main()
