"""
Evaluation Script for Scenario C: Multi-Vocoder (All vs All)
Evaluates a pre-trained model on all available data.
"""

import torch
import json
from pathlib import Path
from LCNN import LCNN
from WaveFakeLoader import create_loaders_from_splits
from train import LFCCLCNNTrainer
import numpy as np
import os
from feature_extraction import LFCCExtractor

def get_vocoder_list(splits_json):
    """Get list of all vocoders"""
    with open(splits_json, 'r') as f:
        splits = json.load(f)
    return list(splits['vocoders'].keys())

def scenario_c_multi_vocoder(splits_json, device='cuda'):
    """
    Evaluate pre-trained model on all vocoders
    """

    print("SCENARIO C: MULTI-VOCODER EVALUATION (ALL VOCODERS)")

    
    # Load data
    train_loader, test_loader = create_loaders_from_splits(
        splits_json=splits_json,
        vocoders_train=None,  # All vocoders
        vocoders_test=None,   # All vocoders
        batch_size=64
    )
    
    # Verify 80/20 split
    print("\n[DATA SPLIT VERIFICATION]")
    train_size = len(train_loader.dataset)
    test_size = len(test_loader.dataset)
    total_size = train_size + test_size
    train_ratio = train_size / total_size
    test_ratio = test_size / total_size
    print(f"Train: {train_size} ({train_ratio*100:.1f}%)")
    print(f"Test:  {test_size} ({test_ratio*100:.1f}%)")
    print(f"Total: {total_size}")
    
    if abs(train_ratio - 0.8) > 0.05 or abs(test_ratio - 0.2) > 0.05:
        print(" WARNING: Not a proper 80/20 split!")
    else:
        print(" Proper 80/20 split confirmed")
    
    # Class distribution check
    train_dataset = train_loader.dataset
    test_dataset = test_loader.dataset
    print(f"\n[CLASS DISTRIBUTION]")
    print(f"Train - Real: {train_dataset.real_count}, Fake: {train_dataset.fake_count}")
    print(f"Test  - Real: {test_dataset.real_count}, Fake: {test_dataset.fake_count}")
    print("="*70 + "\n")
    
    # Initialize model
    model = LCNN(n_lfcc=60, num_classes=2)
    lfcc_extractor = LFCCExtractor(n_lfcc=60, n_filter=60)
    
    # Load pre-trained weights
    script_dir = os.path.dirname(os.path.abspath(__file__))
    weight_path = os.path.join(script_dir, 'weights', 'epoch_075.pt')
    
    if not os.path.exists(weight_path):
        raise FileNotFoundError(f"Pre-trained weights not found at {weight_path}. Train the model first.")
    
    print(f"Loading weights from {weight_path}...")
    checkpoint = torch.load(weight_path, map_location=device)
    model.load_state_dict(checkpoint['model_state_dict'])
    lfcc_extractor.load_state_dict(checkpoint['lfcc_extractor_state_dict'])
    print(f" Checkpoint loaded - Epoch: {checkpoint.get('epoch', 'Unknown') + 1}, "
          f"Previous EER: {checkpoint.get('val_eer', 'Unknown'):.4f}%")
    
    # Evaluate
    trainer = LFCCLCNNTrainer(
        model=model,
        lfcc_extractor=lfcc_extractor,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        lr=0.0001
    )
    
    _, acc, auc, eer = trainer.validate(desc='Evaluating')
    
    print("SCENARIO C RESULTS: ALL-VOCODERS EVALUATION")
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Test AUC:      {auc:.4f}")
    print(f"Test EER:      {eer:.4f}%")
    
    return {'accuracy': acc, 'auc': auc, 'eer': eer}

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    splits_json = os.path.join(script_dir, 'wavefake_splits.json')
    
    if not os.path.exists(splits_json):
        print(f"Error: {splits_json} not found. Run create_wavefake_splits.py first.")
    else:
        scenario_c_multi_vocoder(splits_json, device)
