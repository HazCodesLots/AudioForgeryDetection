"""
Official SONYC-UST Evaluation Script
=====================================
Computes the full evaluation protocol:
  - Fine-level (23 classes): macro-AUPRC, micro-AUPRC
  - Coarse-level (8 classes): macro-AUPRC, micro-AUPRC
  - Per-class AUPRC breakdown
  - Additional: AUC-ROC, Macro F1, sample-level precision/recall
"""

import os
import sys
import argparse
import json
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    f1_score,
    precision_recall_curve
)
from tqdm import tqdm
import importlib.util

def import_module_by_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

script_dir = os.path.dirname(os.path.abspath(__file__))
script_path = os.path.join(script_dir, "ConvNeXt_SONYC-UST.py")
model_script = import_module_by_path("ConvNeXt_SONYC_UST", script_path)

DeepFakeDetectionModel = model_script.DeepFakeDetectionModel
EnhancedAudioFrontend = model_script.EnhancedAudioFrontend
SONYCUSTDataset = model_script.SONYCUSTDataset
pad_truncate_collate = model_script.pad_truncate_collate

FINE_LABELS = [
    '1-1_small-sounding-engine',
    '1-2_medium-sounding-engine',
    '1-3_large-sounding-engine',
    '2-1_rock-drill',
    '2-2_jackhammer',
    '2-3_hoe-ram',
    '2-4_pile-driver',
    '3-1_non-machinery-impact',
    '4-1_chainsaw',
    '4-2_small-medium-rotating-saw',
    '4-3_large-rotating-saw',
    '5-1_car-horn',
    '5-2_car-alarm',
    '5-3_siren',
    '5-4_reverse-beeper',
    '6-1_stationary-music',
    '6-2_mobile-music',
    '6-3_ice-cream-truck',
    '7-1_person-or-small-group-talking',
    '7-2_person-or-small-group-shouting',
    '7-3_large-crowd',
    '7-4_amplified-speech',
    '8-1_dog-barking-whining'
]

COARSE_LABELS = [
    '1_engine',
    '2_machinery-impact',
    '3_non-machinery-impact',
    '4_powered-saw',
    '5_alert-signal',
    '6_music',
    '7_human-voice',
    '8_dog'
]

# Mapping: fine class index -> coarse class index
FINE_TO_COARSE = {
    0: 0, 1: 0, 2: 0,          # 1-1, 1-2, 1-3 -> engine
    3: 1, 4: 1, 5: 1, 6: 1,    # 2-1, 2-2, 2-3, 2-4 -> machinery-impact
    7: 2,                       # 3-1 -> non-machinery-impact
    8: 3, 9: 3, 10: 3,         # 4-1, 4-2, 4-3 -> powered-saw
    11: 4, 12: 4, 13: 4, 14: 4, # 5-1, 5-2, 5-3, 5-4 -> alert-signal
    15: 5, 16: 5, 17: 5,       # 6-1, 6-2, 6-3 -> music
    18: 6, 19: 6, 20: 6, 21: 6, # 7-1, 7-2, 7-3, 7-4 -> human-voice
    22: 7                       # 8-1 -> dog
}

COARSE_TO_FINE = {}
for fine_idx, coarse_idx in FINE_TO_COARSE.items():
    if coarse_idx not in COARSE_TO_FINE:
        COARSE_TO_FINE[coarse_idx] = []
    COARSE_TO_FINE[coarse_idx].append(fine_idx)


def aggregate_to_coarse(fine_array):
    """Aggregate fine-level predictions/labels to coarse level using max."""
    n_samples = fine_array.shape[0]
    coarse_array = np.zeros((n_samples, 8))
    for coarse_idx, fine_indices in COARSE_TO_FINE.items():
        coarse_array[:, coarse_idx] = fine_array[:, fine_indices].max(axis=1)
    return coarse_array


def compute_auprc(targets, predictions, label_names, level_name="Fine"):
    """
    Compute macro-AUPRC and micro-AUPRC for a given level.
    Returns a dict with summary metrics and per-class breakdown.
    """
    n_classes = targets.shape[1]

    per_class = {}
    valid_aps = []
    for i in range(n_classes):
        n_pos = int(targets[:, i].sum())
        if n_pos > 0:
            ap = average_precision_score(targets[:, i], predictions[:, i])
            valid_aps.append(ap)
            per_class[label_names[i]] = {
                'AUPRC': round(float(ap), 4),
                'n_positive': n_pos,
                'prevalence': round(n_pos / len(targets), 4)
            }
        else:
            per_class[label_names[i]] = {
                'AUPRC': None,
                'n_positive': 0,
                'prevalence': 0.0,
                'note': 'skipped (no positive samples)'
            }

    macro_auprc = float(np.mean(valid_aps)) if valid_aps else 0.0

    micro_auprc = float(average_precision_score(
        targets.ravel(), predictions.ravel()
    ))

    auc_scores = []
    for i in range(n_classes):
        if len(np.unique(targets[:, i])) > 1:
            auc_scores.append(roc_auc_score(targets[:, i], predictions[:, i]))
    macro_auc = float(np.mean(auc_scores)) if auc_scores else 0.0

    return {
        'level': level_name,
        'macro_AUPRC': round(macro_auprc, 4),
        'micro_AUPRC': round(micro_auprc, 4),
        'macro_AUC_ROC': round(macro_auc, 4),
        'n_classes_evaluated': len(valid_aps),
        'n_classes_total': n_classes,
        'per_class': per_class
    }


def run_evaluation(dataset_path, weights_path, split='validate', batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Weights: {weights_path}")
    print(f"Split: {split}")

    csv_path = os.path.join(dataset_path, 'annotations.csv')
    frontend = EnhancedAudioFrontend(n_mels=128)
    dataset = SONYCUSTDataset(csv_path, dataset_path, split=split, frontend=frontend)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=pad_truncate_collate, num_workers=0)

    convnext_params = {
        'input_channels': 1, 'depths': [2, 2, 6, 2],
        'dims': [64, 128, 256, 512], 'drop_path_rate': 0.2,
        'layer_scale_init_value': 1e-6
    }
    attention_params = {'input_dim': 512, 'attention_dim': 256, 'num_heads': 4, 'dropout_rate': 0.15}
    mlp_params = {'input_dim': 256, 'num_classes': 23, 'dropout_rate': 0.3}

    model = DeepFakeDetectionModel(convnext_params, attention_params, mlp_params)
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    all_logits = []
    all_targets = []

    print("\nRunning inference...")
    with torch.no_grad():
        for mels, labels, _ in tqdm(loader, desc="Evaluating"):
            mels = mels.to(device)
            outputs = model(mels)
            all_logits.append(outputs.cpu().numpy())
            all_targets.append(labels.numpy())

    logits = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    predictions = 1 / (1 + np.exp(-logits))

    print(f"Evaluated {len(targets)} samples")

    print("FINE-LEVEL EVALUATION (23 classes)")
    fine_results = compute_auprc(targets, predictions, FINE_LABELS, "Fine (23 classes)")
    print(f"  Macro-AUPRC:  {fine_results['macro_AUPRC']:.4f}  (PRIMARY METRIC)")
    print(f"  Micro-AUPRC:  {fine_results['micro_AUPRC']:.4f}")
    print(f"  Macro AUC-ROC: {fine_results['macro_AUC_ROC']:.4f}")
    print(f"  Classes evaluated: {fine_results['n_classes_evaluated']}/{fine_results['n_classes_total']}")

    print("\n  Per-class AUPRC:")
    for name, info in fine_results['per_class'].items():
        if info['AUPRC'] is not None:
            print(f"    {name:45s}  AUPRC={info['AUPRC']:.4f}  (n={info['n_positive']:4d}, prev={info['prevalence']:.3f})")
        else:
            print(f"    {name:45s}  SKIPPED (no positives)")

    coarse_targets = aggregate_to_coarse(targets)
    coarse_preds = aggregate_to_coarse(predictions)

    print("COARSE-LEVEL EVALUATION (8 classes)")
    coarse_results = compute_auprc(coarse_targets, coarse_preds, COARSE_LABELS, "Coarse (8 classes)")
    print(f"  Macro-AUPRC:  {coarse_results['macro_AUPRC']:.4f}")
    print(f"  Micro-AUPRC:  {coarse_results['micro_AUPRC']:.4f}")
    print(f"  Macro AUC-ROC: {coarse_results['macro_AUC_ROC']:.4f}")
    print(f"  Classes evaluated: {coarse_results['n_classes_evaluated']}/{coarse_results['n_classes_total']}")

    print("\n  Per-class AUPRC:")
    for name, info in coarse_results['per_class'].items():
        if info['AUPRC'] is not None:
            print(f"    {name:30s}  AUPRC={info['AUPRC']:.4f}  (n={info['n_positive']:4d}, prev={info['prevalence']:.3f})")
        else:
            print(f"    {name:30s}  SKIPPED (no positives)")

    binary_preds = (predictions > 0.5).astype(int)
    f1_macro = f1_score(targets, binary_preds, average='macro', zero_division=0)
    f1_micro = f1_score(targets, binary_preds, average='micro', zero_division=0)

    print("ADDITIONAL METRICS (threshold=0.5)")
    print(f"  Macro F1: {f1_macro:.4f}")
    print(f"  Micro F1: {f1_micro:.4f}")

    results = {
        'model_weights': weights_path,
        'split': split,
        'n_samples': len(targets),
        'fine_level': fine_results,
        'coarse_level': coarse_results,
        'additional': {
            'macro_F1': round(float(f1_macro), 4),
            'micro_F1': round(float(f1_micro), 4),
        }
    }

    results_path = os.path.join(script_dir, 'results', 'official_evaluation.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nResults saved to: {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official SONYC-UST Evaluation")
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to model weights. Default: results/best_sonyc_model.pth')
    parser.add_argument('--split', type=str, default='validate',
                        choices=['train', 'validate', 'test'])
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    if args.weights is None:
        args.weights = os.path.join(script_dir, 'results', 'best_sonyc_model.pth')

    run_evaluation(args.dataset_path, args.weights, args.split, args.batch_size)
