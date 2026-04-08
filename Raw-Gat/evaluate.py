
import torch
import numpy as np
from tqdm import tqdm
import argparse
import os
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve

from rawgat_st_model import RawGAT_ST
from dataloader import get_dataloader


def compute_eer(scores, labels):
    '''Compute Equal Error Rate'''
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer * 100, thresh


def compute_tDCF(bonafide_scores, spoof_scores, Pfa_asv, Pmiss_asv, Pmiss_cm, Pfa_cm, C_miss, C_fa):
    '''
    Compute tandem Detection Cost Function (t-DCF)
    Based on ASVspoof 2019 evaluation plan
    '''
    assert C_fa >= 0 and C_miss >= 0
    assert Pmiss_cm >= 0 and Pmiss_cm <= 1
    assert Pfa_cm >= 0 and Pfa_cm <= 1
    assert Pmiss_asv >= 0 and Pmiss_asv <= 1
    assert Pfa_asv >= 0 and Pfa_asv <= 1

    Pmiss_cm_tar = Pmiss_cm
    Pfa_cm_spo = Pfa_cm
    tDCF = C_miss * Pmiss_cm_tar * Pmiss_asv + C_fa * Pfa_cm_spo * Pfa_asv

    return tDCF


def compute_min_tDCF(bonafide_scores, spoof_scores):
    '''
    Compute minimum normalized t-DCF for ASVspoof 2021
    Reference: ASVspoof 2021 evaluation plan
    '''
    Pfa_asv = 0.01
    Pmiss_asv = 0.01
    C_miss = 1
    C_fa = 10

    all_scores = np.sort(np.unique(np.concatenate([bonafide_scores, spoof_scores])))
    
    n_bonafide = len(bonafide_scores)
    n_spoof = len(spoof_scores)
    
    min_tdcf = float('inf')
    
    bonafide_scores_sorted = np.sort(bonafide_scores)
    spoof_scores_sorted = np.sort(spoof_scores)
    
    for threshold in all_scores:

        Pmiss_cm = np.searchsorted(bonafide_scores_sorted, threshold, side='left') / n_bonafide

        Pfa_cm = (n_spoof - np.searchsorted(spoof_scores_sorted, threshold, side='left')) / n_spoof


        tdcf = C_miss * Pmiss_cm * Pmiss_asv + C_fa * Pfa_cm * Pfa_asv
        if tdcf < min_tdcf:
            min_tdcf = tdcf

    baseline = min(C_miss * Pmiss_asv, C_fa * Pfa_asv) 
    return min_tdcf / baseline if baseline > 0 else min_tdcf


def evaluate(args):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f'Using device: {device}')

    model = RawGAT_ST(num_classes=2).to(device)

    if os.path.exists(args.model_path):
        print(f'Loading model from {args.model_path}')
        checkpoint = torch.load(args.model_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        print(f"Loaded checkpoint from epoch {checkpoint.get('epoch', 'unknown')}")
        if 'eer' in checkpoint:
            print(f"  Checkpoint EER: {checkpoint['eer']:.2f}%")
    else:
        print(f'WARNING: Model file not found at {args.model_path}')
        print('Using randomly initialized model')

    model.eval()

    print(f'\nLoading evaluation data from:')
    print(f'  Audio: {args.eval_data}')
    print(f'  Protocol: {args.eval_protocol}')

    eval_loader = get_dataloader(
        data_dir=args.eval_data,
        protocol_file=args.eval_protocol,
        batch_size=args.batch_size,
        num_workers=args.num_workers,
        is_train=False,
        is_eval=True
    )


    all_scores = []
    all_labels = []
    all_ids = []

    print('\nEvaluating...')
    with torch.no_grad():
        for audio, labels, audio_ids in tqdm(eval_loader):
            audio = audio.to(device)

            outputs = model(audio)
            scores = outputs[:, 1] - outputs[:, 0]
            
            all_scores.extend(scores.cpu().numpy())
            all_labels.extend(labels.numpy())
            all_ids.extend(audio_ids)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)

    metadata = {}
    metadata_type = None

    try:
        with open(args.eval_protocol, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) == 8:
                    metadata[parts[1]] = parts[2]
                    metadata_type = "Codec"
                elif len(parts) == 5:
                    metadata[parts[1]] = parts[3]
                    metadata_type = "Attack"
    except Exception as e:
        print(f"Warning: Could not parse metadata for breakdown: {e}")

    print('\nComputing metrics...')

    eer, eer_thresh = compute_eer(all_scores, all_labels)

    bonafide_scores = all_scores[all_labels == 1]
    spoof_scores = all_scores[all_labels == 0]
    min_tdcf = compute_min_tDCF(bonafide_scores, spoof_scores)

    print('\n' + '='*70)
    print(f"{'EVALUATION SUMMARY':^70}")
    print('='*70)
    print(f'Total samples: {len(all_scores):<10} Bonafide: {len(bonafide_scores):<10} Spoof: {len(spoof_scores)}')
    print(f'\nOVERALL METRICS:')
    print(f'  EER:            {eer:.2f}%')
    print(f'  Threshold:      {eer_thresh:.4f}')
    print(f'  min t-DCF:      {min_tdcf:.4f} (normalized)')
    print('='*70)

    if metadata:
        print(f"\n{f'{metadata_type.upper()}-WISE BREAKDOWN':^70}")
        print('-' * 70)
        print(f"{metadata_type:<12} | {'Count':<8} | {'EER (%)':<10} | {'min t-DCF':<10}")
        print('-' * 70)
        
        categories = sorted(set(metadata.values()))
        
        if '-' in categories:
            categories.remove('-')
            categories = ['-'] + categories
            
        for cat in categories:
            cat_mask = np.array([metadata.get(aid) == cat for aid in all_ids])
            if cat_mask.sum() == 0: continue
            
            c_scores = all_scores[cat_mask]
            c_labels = all_labels[cat_mask]
            
            if metadata_type == "Attack" and cat != '-':
                c_bonafide = bonafide_scores
                c_spoof = c_scores[c_labels == 0]
                
                if len(c_bonafide) == 0 or len(c_spoof) == 0:
                    print(f"{cat:<12} | {len(c_spoof):<8} | {'N/A':<10} | {'N/A':<10}")
                    continue
                    
                cat_labels = np.concatenate([np.ones(len(c_bonafide)), np.zeros(len(c_spoof))])
                cat_scores = np.concatenate([c_bonafide, c_spoof])
                c_eer, _ = compute_eer(cat_scores, cat_labels)
                c_tdcf = compute_min_tDCF(c_bonafide, c_spoof)
                
                print(f"{cat:<12} | {len(c_spoof):<8} | {c_eer:<10.2f} | {c_tdcf:<10.4f}")
            else:

                if len(np.unique(c_labels)) < 2:
                    print(f"{cat:<12} | {len(c_scores):<8} | {'-':<10} | {'-':<10}")
                    continue
                    
                c_eer, _ = compute_eer(c_scores, c_labels)
                c_bonafide = c_scores[c_labels == 1]
                c_spoof = c_scores[c_labels == 0]
                c_tdcf = compute_min_tDCF(c_bonafide, c_spoof)
                
                print(f"{cat:<12} | {len(c_scores):<8} | {c_eer:<10.2f} | {c_tdcf:<10.4f}")
        print('-' * 70)

    if args.output_file:
        print(f'\nSaving scores to {args.output_file}')
        with open(args.output_file, 'w') as f:
            for audio_id, score, label in zip(all_ids, all_scores, all_labels):
                f.write(f'{audio_id} {score:.6f} {"bonafide" if label == 1 else "spoof"}\n')
        print('✓ Scores saved')

    return eer, min_tdcf


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Evaluate RawGAT-ST model')

    parser.add_argument('--model_path', type=str, required=True,
                       help='Path to trained model checkpoint')

    parser.add_argument('--eval_data', type=str,
                       default=r'N:\ASVspoof2021\ASVspoof2021_LA_eval\flac',
                       help='Path to evaluation audio files')
    parser.add_argument('--eval_protocol', type=str,
                       default=r'N:\ASVspoof2021\keys\LA\trial_metadata.txt',
                       help='Path to evaluation protocol file')

    parser.add_argument('--batch_size', type=int, default=32, help='Batch size')
    parser.add_argument('--num_workers', type=int, default=4, help='Number of workers')
    parser.add_argument('--output_file', type=str, default='scores.txt',
                       help='Output file for scores')

    args = parser.parse_args()

    evaluate(args)
