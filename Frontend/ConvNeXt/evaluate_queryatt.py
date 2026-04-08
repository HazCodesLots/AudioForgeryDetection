import os
import json
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from sklearn.metrics import average_precision_score
from tqdm import tqdm
import numpy as np
import argparse

# Modular Import: Use QueryAtt classes
from ConvNeXt_QueryAtt import ConvNeXtTagger, EnhancedAudioFrontend, SONYCUSTDataset, pad_truncate_collate

FINE_LABELS = [
    '1-1_small-sounding-engine', '1-2_medium-sounding-engine', '1-3_large-sounding-engine',
    '2-1_rock-drill', '2-2_jackhammer', '2-3_hoe-ram', '2-4_pile-driver',
    '3-1_non-machinery-impact', '4-1_chainsaw', '4-2_small-medium-rotating-saw',
    '4-3_large-rotating-saw', '5-1_car-horn', '5-2_car-alarm', '5-3_siren',
    '5-4_reverse-beeper', '6-1_stationary-music', '6-2_mobile-music',
    '6-3_ice-cream-truck', '7-1_person-or-small-group-talking',
    '7-2_person-or-small-group-shouting', '7-3_large-crowd', '7-4_amplified-speech',
    '8-1_dog-barking-whining'
]

def aggregate_to_coarse(fine_array):
    FINE_TO_COARSE = {
        0: 0, 1: 0, 2: 0, 3: 1, 4: 1, 5: 1, 6: 1, 7: 2, 8: 3, 9: 3, 10: 3,
        11: 4, 12: 4, 13: 4, 14: 4, 15: 5, 16: 5, 17: 5, 18: 6, 19: 6, 20: 6, 21: 6, 22: 7
    }
    coarse_array = np.zeros((fine_array.shape[0], 8))
    for fine_idx, coarse_idx in FINE_TO_COARSE.items():
        coarse_array[:, coarse_idx] = np.maximum(coarse_array[:, coarse_idx], fine_array[:, fine_idx])
    return coarse_array

def run_evaluation(dataset_path, weights_path, split='test', batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device} | Split: {split} | Model: QueryAtt")
    
    csv_path = os.path.join(dataset_path, 'annotations.csv')
    frontend = EnhancedAudioFrontend(n_mels=128)
    dataset = SONYCUSTDataset(csv_path, dataset_path, split=split, frontend=frontend)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_truncate_collate)

    # Standardized params
    convnext_params = {'depths': [2, 2, 6, 2], 'dims': [64, 128, 256, 512], 'drop_path_rate': 0.2, 'layer_scale_init_value': 1e-6}
    attention_params = {'input_dim': 512, 'num_heads': 4}
    mlp_params = {'input_dim': 512, 'num_classes': 23, 'dropout_rate': 0.3}

    model = ConvNeXtTagger(convnext_params, attention_params, mlp_params).to(device)
    model.load_state_dict(torch.load(weights_path, map_location=device))
    model.eval()

    all_logits, all_targets = [], []
    with torch.no_grad():
        for mels, labels, _ in tqdm(loader, desc="Evaluating"):
            logits = model(mels.to(device))
            all_logits.append(logits.cpu().numpy())
            all_targets.append(labels.numpy())

    logits = np.concatenate(all_logits)
    targets = np.concatenate(all_targets)
    probs = 1 / (1 + np.exp(-logits))

    fine_aps = [average_precision_score(targets[:, i], probs[:, i]) for i in range(23) if targets[:, i].sum() > 0]
    macro_auprc_fine = np.mean(fine_aps)

    coarse_targets = aggregate_to_coarse(targets)
    coarse_probs = aggregate_to_coarse(probs)
    coarse_aps = [average_precision_score(coarse_targets[:, i], coarse_probs[:, i]) for i in range(8)]
    macro_auprc_coarse = np.mean(coarse_aps)

    print(f"\nFinal Results for {split} split (QueryAtt):")
    print(f"  Macro-AUPRC (Fine):   {macro_auprc_fine:.4f}")
    print(f"  Macro-AUPRC (Coarse): {macro_auprc_coarse:.4f}")

    results = {
        'model': 'QueryAtt',
        'split': split,
        'weights': weights_path,
        'macro_auprc_fine': macro_auprc_fine,
        'macro_auprc_coarse': macro_auprc_coarse,
        'per_class_fine': {label: float(ap) for label, ap in zip(FINE_LABELS, fine_aps)}
    }
    
    out_path = os.path.join(os.path.dirname(weights_path), f"eval_queryatt_{split}.json")
    with open(out_path, 'w') as f:
        json.dump(results, f, indent=4)
    print(f"Saved results to {out_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()
    run_evaluation(args.dataset_path, args.weights, args.split)
