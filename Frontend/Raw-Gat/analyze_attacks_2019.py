
import numpy as np
import argparse
from scipy.optimize import brentq
from scipy.interpolate import interp1d
from sklearn.metrics import roc_curve
import os

def compute_eer(scores, labels):
    '''Compute Equal Error Rate'''
    fpr, tpr, thresholds = roc_curve(labels, scores, pos_label=1)
    eer = brentq(lambda x: 1. - x - interp1d(fpr, tpr)(x), 0., 1.)
    thresh = interp1d(fpr, thresholds)(eer)
    return eer * 100, thresh

def compute_min_tDCF(bonafide_scores, spoof_scores):
    '''Compute minimum normalized t-DCF for ASVspoof 2019/2021'''
    # Constant parameters for ASVspoof 2021 LA (Standard defaults)
    Pfa_asv = 0.01
    Pmiss_asv = 0.01
    C_miss = 1
    C_fa = 10

    all_scores = np.sort(np.unique(np.concatenate([bonafide_scores, spoof_scores])))
    n_bonafide = len(bonafide_scores)
    n_spoof = len(spoof_scores)
    
    bonafide_scores_sorted = np.sort(bonafide_scores)
    spoof_scores_sorted = np.sort(spoof_scores)
    
    min_tdcf = float('inf')
    for threshold in all_scores:
        Pmiss_cm = np.searchsorted(bonafide_scores_sorted, threshold, side='left') / n_bonafide
        Pfa_cm = (n_spoof - np.searchsorted(spoof_scores_sorted, threshold, side='left')) / n_spoof
        tdcf = C_miss * Pmiss_cm * Pmiss_asv + C_fa * Pfa_cm * Pfa_asv
        if tdcf < min_tdcf:
            min_tdcf = tdcf

    baseline = min(C_miss * Pmiss_asv, C_fa * Pfa_asv) 
    return min_tdcf / baseline if baseline > 0 else min_tdcf

def main():
    parser = argparse.ArgumentParser(description='Analyze attack-wise performance from scores.txt')
    parser.add_argument('--scores', type=str, default='scores.txt', help='Path to scores.txt')
    parser.add_argument('--protocol', type=str, required=True, help='Path to ASVspoof 2019 protocol file')
    args = parser.parse_args()

    # 1. Load scores
    print(f"Loading scores from {args.scores}...")
    scores_dict = {}
    with open(args.scores, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) >= 2:
                scores_dict[parts[0]] = float(parts[1])

    # 2. Load protocol and map attacks
    print(f"Loading protocol from {args.protocol}...")
    attack_metadata = {}
    all_audio_ids = []
    all_scores = []
    all_labels = []
    
    with open(args.protocol, 'r') as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) == 5:
                audio_id = parts[1]
                attack_id = parts[3]
                label = 1 if parts[4] == 'bonafide' else 0
                
                if audio_id in scores_dict:
                    attack_metadata[audio_id] = attack_id
                    all_audio_ids.append(audio_id)
                    all_scores.append(scores_dict[audio_id])
                    all_labels.append(label)

    all_scores = np.array(all_scores)
    all_labels = np.array(all_labels)
    bonafide_scores = all_scores[all_labels == 1]

    # 3. Compute Metrics
    eer, _ = compute_eer(all_scores, all_labels)
    min_tdcf = compute_min_tDCF(bonafide_scores, all_scores[all_labels == 0])

    print('\n' + '='*70)
    print(f"{'EVALUATION SUMMARY':^70}")
    print('='*70)
    print(f'Total samples: {len(all_scores):<10}')
    print(f'Overall EER:   {eer:.2f}%')
    print(f'Overall min t-DCF: {min_tdcf:.4f}')
    print('='*70)

    # 4. Attack-wise breakdown
    print(f"\n{'ATTACK-WISE BREAKDOWN':^70}")
    print('-' * 70)
    print(f"{'Attack':<12} | {'Count':<8} | {'EER (%)':<10} | {'min t-DCF':<10}")
    print('-' * 70)
    
    attacks = sorted(set(attack_metadata.values()))
    if '-' in attacks:
        attacks.remove('-')
        print(f"{'- (Bona)':<12} | {len(bonafide_scores):<8} | {'-':<10} | {'-':<10}")
        
    for atk in attacks:
        atk_indices = [i for i, aid in enumerate(all_audio_ids) if attack_metadata[aid] == atk]
        atk_scores = all_scores[atk_indices]
        
        # Merge current attack with ALL bonafide
        combined_scores = np.concatenate([bonafide_scores, atk_scores])
        combined_labels = np.concatenate([np.ones(len(bonafide_scores)), np.zeros(len(atk_scores))])
        
        atk_eer, _ = compute_eer(combined_scores, combined_labels)
        atk_tdcf = compute_min_tDCF(bonafide_scores, atk_scores)
        
        print(f"{atk:<12} | {len(atk_scores):<8} | {atk_eer:<10.2f} | {atk_tdcf:<10.4f}")
    print('-' * 70)

if __name__ == '__main__':
    main()
