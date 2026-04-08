import os
import torch
import numpy as np
import pandas as pd
from tqdm import tqdm
from sklearn.metrics import average_precision_score, f1_score, roc_auc_score
import json

import importlib
mbce_module = importlib.import_module("ConvNeXt_SplitBand_M-BCE")
ConvNeXtTagger = mbce_module.ConvNeXtTagger
SONYCUSTDataset = mbce_module.SONYCUSTDataset
EnhancedAudioFrontend = mbce_module.EnhancedAudioFrontend
pad_truncate_collate_simple = mbce_module.pad_truncate_collate_simple
from torch.utils.data import DataLoader

def comprehensive_evaluation(predictions, targets):
    """
    Computes mAP, AUC-ROC (macro), and F1 (macro/micro)
    """
    # mAP (Macro Average Precision)
    aps = []
    for i in range(targets.shape[1]):
        if targets[:, i].sum() > 0:
            aps.append(average_precision_score(targets[:, i], predictions[:, i]))
    mAP = np.mean(aps) if aps else 0.0

    # AUC-ROC
    aucs = []
    for i in range(targets.shape[1]):
        if len(np.unique(targets[:, i])) > 1:
            aucs.append(roc_auc_score(targets[:, i], predictions[:, i]))
    auc_roc = np.mean(aucs) if aucs else 0.0

    # F1 Scores
    binary_preds = (predictions > 0.5).astype(int)
    f1_macro = f1_score(targets, binary_preds, average='macro', zero_division=0)
    f1_micro = f1_score(targets, binary_preds, average='micro', zero_division=0)

    return {
        'mAP': float(mAP),
        'AUC': float(auc_roc),
        'F1_macro': float(f1_macro),
        'F1_micro': float(f1_micro)
    }

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--weights', type=str, required=True)
    parser.add_argument('--split', type=str, default='test', choices=['validate', 'test'])
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Evaluating M-BCE on {args.split} split using {device}")

    # Setup Dataset
    frontend = EnhancedAudioFrontend(n_mels=128)
    ds = SONYCUSTDataset(
        csv_path=os.path.join(args.dataset_path, 'annotations.csv'),
        root_dir=args.dataset_path,
        split=args.split,
        frontend=frontend
    )
    loader = DataLoader(ds, batch_size=16, shuffle=False, collate_fn=pad_truncate_collate_simple, num_workers=0)

    # Load Model
    # Note: M-BCE uses SplitBand architecture
    model = ConvNeXtTagger(
        convnext_params={'depths': [2, 2, 6, 2], 'dims': [64, 128, 256, 512], 'drop_path_rate': 0.0},
        pooling_params={'input_dim': 512, 'output_dim': 512, 'lse_beta': 10.0},
        mlp_params={'input_dim': 512, 'num_classes': 23}
    ).to(device)

    checkpoint = torch.load(args.weights, map_location=device)
    if 'model_state_dict' in checkpoint:
        model.load_state_dict(checkpoint['model_state_dict'])
    elif 'model' in checkpoint:
        model.load_state_dict(checkpoint['model'])
    else:
        model.load_state_dict(checkpoint)
    
    model.eval()

    all_logits, all_targets = [], []
    with torch.no_grad():
        for mels, labels, _ in tqdm(loader, desc="Evaluating"):
            mels = mels.to(device)
            logits = model(mels)
            all_logits.append(logits.cpu().numpy())
            all_targets.append(labels.numpy())

    logits = np.concatenate(all_logits, axis=0)
    targets = np.concatenate(all_targets, axis=0)
    predictions = 1 / (1 + np.exp(-logits))

    metrics = comprehensive_evaluation(predictions, targets)
    print(f"\nResults ({args.split}):")
    print(json.dumps(metrics, indent=4))

if __name__ == "__main__":
    main()
