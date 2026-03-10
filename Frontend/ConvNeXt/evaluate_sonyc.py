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
script_path = os.path.join(script_dir, "ConvNeXt_QueryAtt.py")
model_script = import_module_by_path("ConvNeXt_QueryAtt", script_path)

ConvNeXtTagger = model_script.ConvNeXtTagger
EnhancedAudioFrontend = model_script.EnhancedAudioFrontend
SONYCUSTDataset = model_script.SONYCUSTDataset
pad_truncate_collate = model_script.pad_truncate_collate

class LegacyAttentionPooling(nn.Module):
    """
    Legacy pooling: Standard Multihead Self-Attention followed by mean pooling.
    Fixes the mismatch where missing 'time_query'/'freq_query' caused noise in aggregation.
    """
    def __init__(self, input_dim=512, num_heads=4, output_dim=512):
        super().__init__()
        self.time_attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.freq_attn = nn.MultiheadAttention(input_dim, num_heads, batch_first=True)
        self.time_norm = nn.LayerNorm(input_dim)
        self.freq_norm = nn.LayerNorm(input_dim)
        self.out_proj = nn.Linear(input_dim, output_dim)

    def forward(self, x):
        B, C, F, T = x.shape
        # Time dimension
        x_t = x.permute(0, 2, 3, 1).contiguous().view(B * F, T, C)
        attn_out, time_w = self.time_attn(x_t, x_t, x_t)
        x_t_pooled = self.time_norm(attn_out.mean(dim=1)) # Global Average over time
        x_t_pooled = x_t_pooled.view(B, F, C)

        # Frequency dimension
        attn_out, freq_w = self.freq_attn(x_t_pooled, x_t_pooled, x_t_pooled)
        x_f_pooled = self.freq_norm(attn_out.mean(dim=1)) # Global Average over frequency
        
        return self.out_proj(x_f_pooled), time_w, freq_w

class GAPPooling(nn.Module):
    """
    Pure Global Average Pooling (Pure GAP) baseline.
    Uses no attention mechanisms.
    """
    def __init__(self, input_dim=512, output_dim=512):
        super().__init__()
        self.pooling = nn.AdaptiveAvgPool2d(1)
        self.out_proj = nn.Linear(input_dim, output_dim) if input_dim != output_dim else nn.Identity()

    def forward(self, x):
        # x: (B, C, F, T)
        pooled = self.pooling(x).view(x.size(0), -1)
        return self.out_proj(pooled), None, None

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


def run_evaluation(dataset_path, weights_path, split='validate', batch_size=16, output_name='official_evaluation.json', calibrate=False, thresholds_path=None):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    print(f"Weights: {weights_path}")
    print(f"Split: {split}")

    csv_path = os.path.join(dataset_path, 'annotations.csv')
    frontend = EnhancedAudioFrontend(n_mels=128)
    dataset = SONYCUSTDataset(csv_path, dataset_path, split=split, frontend=frontend)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                        collate_fn=pad_truncate_collate, num_workers=0)

    # Detect architecture based on weights path or directory
    specialized_module = None
    abs_weights_path = os.path.abspath(weights_path)
    p = abs_weights_path.lower()
    is_gap       = "globalaveragepooling" in p or "global average pooling" in p or "gap" in p and "gapgmp" not in p
    is_legacy    = ("globalattentionpooling" in p or "global attention pooling" in p) and not is_gap
    is_gapgmp    = "gapgmp" in p
    is_splitband = "splitband" in p and not any(x in p for x in ["splitbandasl", "splitband_m-bce", "splitbandbcelese"])
    is_splitbandasl   = "splitbandasl" in p
    is_splitband_mbce = "splitband_m-bce" in p
    is_bcelese        = "splitbandbcelese" in p
    
    # Base params
    convnext_params = {
        'input_channels': 1, 'depths': [2, 2, 6, 2],
        'dims': [64, 128, 256, 512], 'drop_path_rate': 0.2,
        'layer_scale_init_value': 1e-6
    }
    mlp_params = {'input_dim': 512, 'num_classes': 23, 'dropout_rate': 0.3}
    attention_params = None

    if is_legacy:
        print("DETECTED: Legacy GlobalAttentionPooling Architecture (Self-Attention + Mean)")
        attention_params = {'input_dim': 512, 'num_heads': 4, 'output_dim': 512}
        model_type = "GlobalAttentionPooling"
    elif is_gap:
        print("DETECTED: Pure GlobalAveragePooling Baseline (No Attention)")
        attention_params = {'input_dim': 512, 'num_heads': 4, 'output_dim': 512}
        model_type = "GlobalAveragePooling"
    elif is_gapgmp or is_splitband or is_splitbandasl or is_splitband_mbce or is_bcelese:
        if is_splitband:
            print("DETECTED: SplitBand Fused Pooling")
            target_script = "ConvNeXt_SplitBand.py"
            model_type = "ConvNeXt_SplitBand"
            pooling_params = {'input_dim': 512, 'output_dim': 512, 'dropout': 0.1}
        elif is_gapgmp:
            print("DETECTED: GAP+GMP Fused Pooling")
            target_script = "ConvNeXt_GAPGMP.py"
            model_type = "ConvNeXt_GAPGMP"
            pooling_params = {'input_dim': 512, 'output_dim': 512, 'dropout': 0.1}
        elif is_splitband_mbce:
            print("DETECTED: SplitBand M-BCE (MarginBCE + High-Beta LSE + 128 Mels + 16kHz)")
            target_script = "ConvNeXt_SplitBand_M-BCE.py"
            model_type = "ConvNeXt_SplitBand_M-BCE"
            pooling_params = {'input_dim': 512, 'output_dim': 512, 'dropout': 0.1, 'lse_beta': 10.0}
        elif is_bcelese:
            print("DETECTED: SplitBand BCE-LSE  (Weighted BCE + LSE Pooling, no margin — Ablation A)")
            target_script = "ConvNeXt_SplitbandBCELSE.py"
            model_type = "ConvNeXt_SplitbandBCELSE"
            pooling_params = {'input_dim': 512, 'output_dim': 512, 'dropout': 0.1, 'lse_beta': 10.0}
        else:
            print("DETECTED: SplitBandASL (ASL + LSE + 256 Mels + 32kHz)")
            target_script = "ConvNeXt_SplitBandASL.py"
            model_type = "ConvNeXt_SplitBandASL"
            pooling_params = {'input_dim': 512, 'output_dim': 512, 'dropout': 0.1, 'lse_beta': 5.0}

        specialized_module = import_module_by_path("SpecializedModel", os.path.join(script_dir, target_script))

    # Instantiate model
    if is_gapgmp or is_splitband or is_splitbandasl or is_splitband_mbce or is_bcelese:
        model = specialized_module.ConvNeXtTagger(convnext_params, pooling_params, mlp_params)
    else:
        model = ConvNeXtTagger(convnext_params, attention_params, mlp_params)
    
    if is_splitbandasl or is_splitband_mbce or is_bcelese:
        # Re-instantiate dataset using specialized class
        dataset = specialized_module.SONYCUSTDataset(csv_path, dataset_path, split=split, frontend=frontend)
        
        if is_splitbandasl:
            frontend = specialized_module.EnhancedAudioFrontend(n_mels=256)
            dataset.frontend = frontend
            # V2/ASL: 256 mels, num_workers=0
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=specialized_module.RareClassMixupCollate(dataset),
                                num_workers=0, pin_memory=True)
        else: # is_splitband_mbce
            frontend = specialized_module.EnhancedAudioFrontend(n_mels=128)
            dataset.frontend = frontend
            # M-BCE: 128 mels, num_workers=0 
            loader = DataLoader(dataset, batch_size=batch_size, shuffle=False,
                                collate_fn=specialized_module.RareClassMixupCollate(dataset),
                                num_workers=0, pin_memory=True)
    
    if is_legacy:
        # Swap the pooling layer to the legacy version before loading weights
        model.att_pool = LegacyAttentionPooling(
            input_dim=attention_params['input_dim'],
            num_heads=attention_params['num_heads'],
            output_dim=attention_params.get('output_dim', attention_params['input_dim'])
        )
        print("Model architecture swapped to LegacyAttentionPooling")
    elif is_gap:
        # Swap to Pure GAP pooling
        model.att_pool = GAPPooling(
            input_dim=attention_params['input_dim'],
            output_dim=attention_params.get('output_dim', attention_params['input_dim'])
        )
        print("Model architecture swapped to GAPPooling (Pure GAP)")

    state_dict = torch.load(weights_path, map_location=device)
    # Filter state_dict keys for GAP model as it doesn't have att_pool.time_attn etc.
    load_status = model.load_state_dict(state_dict, strict=True) 
    if is_legacy:
        print(f"Legacy Load Status: {load_status}")
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
    # Load thresholds if provided
    class_thresholds = np.full(23, 0.5)
    if thresholds_path and os.path.exists(thresholds_path):
        print(f"Loading custom thresholds from: {thresholds_path}")
        with open(thresholds_path, 'r') as f:
            thresh_dict = json.load(f)
            # Match by class name or index
            for i, name in enumerate(FINE_LABELS):
                if name in thresh_dict:
                    class_thresholds[i] = thresh_dict[name]

    if calibrate:
        print("Calibrating thresholds per-class on this split...")
        if specialized_module and hasattr(specialized_module, 'calibrate_thresholds'):
            class_thresholds = specialized_module.calibrate_thresholds(predictions, targets)
        else:
            # Fallback inline calibration
            for i in range(23):
                best_f1, best_t = 0, 0.5
                for t in np.arange(0.05, 0.96, 0.05):
                    p_bin = (predictions[:, i] > t).astype(int)
                    f1 = f1_score(targets[:, i], p_bin, zero_division=0)
                    if f1 > best_f1:
                        best_f1, best_t = f1, t
                class_thresholds[i] = best_t
        
        # Save calibrated thresholds
        results_root = os.path.join(os.getcwd(), 'results')
        cal_save_dir = os.path.join(results_root, model_type)
        os.makedirs(cal_save_dir, exist_ok=True)
        
        cal_path = os.path.join(cal_save_dir, 'thresholds.json')
        with open(cal_path, 'w') as f:
            json.dump({FINE_LABELS[i]: float(class_thresholds[i]) for i in range(23)}, f, indent=2)
        print(f"Calibrated thresholds saved to: {cal_path}")

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

    binary_preds = (predictions > class_thresholds).astype(int)
    f1_macro = f1_score(targets, binary_preds, average='macro', zero_division=0)
    f1_micro = f1_score(targets, binary_preds, average='micro', zero_division=0)

    print(f"ADDITIONAL METRICS (Thresholds: {'calibrated' if calibrate or thresholds_path else 'global 0.5'})")
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

    results_dir = os.path.join(script_dir, 'results', model_type)
    os.makedirs(results_dir, exist_ok=True)
    results_path = os.path.join(results_dir, output_name)
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"\nEvaluation Results saved to: {results_path}")

    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Official SONYC-UST Evaluation")
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--weights', type=str, default=None,
                        help='Path to model weights. Default: results/best_sonyc_model.pth')
    parser.add_argument('--split', type=str, default='test',
                        choices=['train', 'validate', 'test'])
    parser.add_argument('--batch_size', type=int, default=16)
    parser.add_argument('--output_name', type=str, default='official_evaluation.json')
    parser.add_argument('--calibrate', action='store_true', help='Calibrate thresholds per-class')
    parser.add_argument('--thresholds_path', type=str, default=None, help='Path to thresholds.json')
    args = parser.parse_args()

    if args.weights is None:
        args.weights = os.path.join(script_dir, 'results', 'best_sonyc_model.pth')

    run_evaluation(args.dataset_path, args.weights, args.split, args.batch_size, args.output_name, args.calibrate, args.thresholds_path)
