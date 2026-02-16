import torch
import json
import os
import numpy as np
from tqdm import tqdm
from LCNN import LCNN
from feature_extraction import LFCCExtractor
from WaveFakeLoader import create_loaders_from_splits, WaveFakeDatasetFixed
from train import LFCCLCNNTrainer

def evaluate_lovo_checkpoints(splits_json, device='cuda'):
    """
    Load the 10 LOVO checkpoints and verify their performance on the held-out vocoders.
    """
    print("\n" + "="*70)
    print("SCENARIO B: FINAL LOVO EVALUATION (10-FOLD)")
    print("="*70)
    
    with open(splits_json, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    vocoders = list(splits['vocoders'].keys())
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results = {}
    
    for fold_idx, held_out_vocoder in enumerate(vocoders, 1):
        print(f"\n--- [FOLD {fold_idx}/10] Evaluating held-out: {held_out_vocoder} ---")
        
        # 1. Prepare Data Loader (Full held-out vocoder + Real test set)
        test_dataset = WaveFakeDatasetFixed(
            splits_json=splits_json,
            split_type='all', 
            vocoders_to_include=[held_out_vocoder],
            include_real=True,
            lfcc_extractor=LFCCExtractor(n_lfcc=60, n_filter=60),
            noise_std=0.0
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=64, shuffle=False,
            num_workers=4, pin_memory=True
        )
        
        # 2. Setup Model
        model = LCNN(n_lfcc=60, num_classes=2).to(device)
        
        # 3. Load Weights
        # Path used in train_wavefake_lovo.py:
        weight_path = os.path.join(script_dir, 'weights', 'protocol', f'lovo_{held_out_vocoder}', f'lovo_{held_out_vocoder}_best.pt')
        
        if not os.path.exists(weight_path):
            print(f"  🚨 Error: Weights not found at {weight_path}")
            continue
            
        print(f"  ✓ Loading weights: {os.path.basename(weight_path)}")
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # 4. Evaluate
        trainer = LFCCLCNNTrainer(
            model=model,
            lfcc_extractor=LFCCExtractor(n_lfcc=60, n_filter=60),
            train_loader=None, # Not needed for eval
            val_loader=test_loader,
            device=device
        )
        
        _, acc, auc, eer = trainer.validate(desc=f'Eval Fold {fold_idx}')
        
        results[held_out_vocoder] = {
            'eer': eer,
            'acc': acc,
            'auc': auc
        }
        print(f"  ✓ Result - EER: {eer:.4f}%, Acc: {acc*100:.2f}%")

    # 5. Print Final Report
    print("\n" + "="*70)
    print("FINAL SCENARIO B (LOVO) VERIFICATION REPORT")
    print("="*70)
    print(f"{'Held-out Vocoder':<30} {'EER':<10} {'Accuracy':<10}")
    print("-" * 70)
    
    eers = []
    for k in sorted(results.keys()):
        v = results[k]
        print(f"{k:<30} {v['eer']:>7.4f}%  {v['acc']*100:>8.2f}%")
        eers.append(v['eer'])
        
    print("-" * 70)
    print(f"{'AVERAGE (aEER)':<30} {np.mean(eers):>7.4f}%")
    print("="*70)

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    splits_json = os.path.join(script_dir, 'wavefake_splits.json')
    
    if not os.path.exists(splits_json):
        print(f"Error: {splits_json} not found.")
    else:
        evaluate_lovo_checkpoints(splits_json, device)
