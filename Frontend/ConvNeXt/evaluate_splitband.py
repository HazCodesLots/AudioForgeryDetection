import os
import torch
import torch.nn as nn
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import json
import argparse
from tqdm import tqdm

# Import the champion components directly from the training script
from ConvNeXt_SplitBand_GAPGMP import (
    ConvNeXtTagger, 
    EnhancedAudioFrontend, 
    SONYCUSTDataset, 
    pad_truncate_collate, 
    comprehensive_evaluation
)

def evaluate(model_path, dataset_path, split='test', batch_size=16):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating {model_path} on {split} split using {device}...")

    # 1. Initialize Model (Standardized Tiny Backbone + SplitBand Pooling)
    convnext_params = {
        'input_channels': 1,
        'depths': [2, 2, 6, 2],
        'dims': [64, 128, 256, 512],
        'drop_path_rate': 0.0, # Eval mode
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
    
    # 2. Load Weights
    state_dict = torch.load(model_path, map_location=device)
    # Handle both full checkpoints and state dicts
    if 'model_state_dict' in state_dict:
        state_dict = state_dict['model_state_dict']
    
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # 3. Setup Dataset
    csv_path = os.path.join(dataset_path, 'annotations.csv')
    frontend = EnhancedAudioFrontend(n_mels=128)
    dataset  = SONYCUSTDataset(csv_path, dataset_path, split=split, frontend=frontend)
    loader   = DataLoader(dataset, batch_size=batch_size, shuffle=False, collate_fn=pad_truncate_collate)

    # 4. Inference
    all_logits = []
    all_targets = []
    
    with torch.no_grad():
        for mels, labels, _ in tqdm(loader, desc=f"Eval ({split})"):
            mels, labels = mels.to(device), labels.to(device).float()
            logits = model(mels)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(labels.cpu().numpy())

    logits = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    predictions = 1 / (1 + np.exp(-logits)) # Sigmoid

    # 5. Metrics
    metrics = comprehensive_evaluation(predictions, targets)
    
    print("\n" + "="*30)
    print(f"Results for {split.upper()} split:")
    print(f"  mAP:      {metrics['mAP']:.4f}")
    print(f"  AUC:      {metrics['AUC']:.4f}")
    print(f"  F1-micro: {metrics['F1_micro']:.4f}")
    print(f"  F1-macro: {metrics['F1_macro']:.4f}")
    print("="*30)

    # 6. Save results
    save_dir = os.path.dirname(model_path)
    output_fn = f"official_evaluation_{split}.json"
    output_path = os.path.join(save_dir, output_fn)
    
    result_data = {
        'model_path': model_path,
        'split': split,
        'metrics': metrics
    }
    
    with open(output_path, 'w') as f:
        json.dump(result_data, f, indent=4)
    print(f"Evaluation saved to {output_path}")

    return metrics

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--split', type=str, default='test')
    parser.add_argument('--batch_size', type=int, default=16)
    args = parser.parse_args()

    evaluate(args.model_path, args.dataset_path, args.split, args.batch_size)
