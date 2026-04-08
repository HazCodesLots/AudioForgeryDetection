import os
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import json
import argparse
from tqdm import tqdm

# Import components directly from the baseline script
from ConvNeXt_GAPGMP import (
    ConvNeXtTagger, 
    EnhancedAudioFrontend, 
    SONYCUSTDataset, 
    pad_truncate_collate, 
    comprehensive_evaluation
)

def evaluate(model_path, dataset_path, split='test', batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating GAP+GMP Baseline: {model_path} on {split} using {device}...")

    # Standardized Tiny Backbone
    convnext_params = {
        'input_channels': 1,
        'depths': [2, 2, 6, 2],
        'dims': [64, 128, 256, 512],
        'drop_path_rate': 0.0,
        'layer_scale_init_value': 1e-6
    }
    pooling_params = {
        'input_dim': 512,
        'output_dim': 512,
        'dropout': 0.0,
    }
    mlp_params = {
        'input_dim': 512,
        'num_classes': 23,
        'dropout_rate': 0.0
    }

    model = ConvNeXtTagger(convnext_params, pooling_params, mlp_params)
    state_dict = torch.load(model_path, map_location=device)
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    csv_path = os.path.join(dataset_path, 'annotations.csv')
    frontend = EnhancedAudioFrontend(n_mels=128)
    dataset  = SONYCUSTDataset(csv_path, dataset_path, split=split, frontend=frontend)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_truncate_collate)

    all_logits, all_targets = [], []
    with torch.no_grad():
        for mels, labels, _ in tqdm(loader, desc=f"Eval ({split})"):
            mels, labels = mels.to(device), labels.to(device).float()
            logits = model(mels)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    predictions = 1 / (1 + np.exp(-logits))

    metrics = comprehensive_evaluation(predictions, targets)
    print(f"\nGAP+GMP Baseline ({split}): mAP={metrics['mAP']:.4f}, AUC={metrics['AUC']:.4f}")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    args = parser.parse_args()
    evaluate(args.model_path, args.dataset_path, args.split)
