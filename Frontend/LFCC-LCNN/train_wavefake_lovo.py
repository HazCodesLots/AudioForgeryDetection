import torch
import json
import argparse
from pathlib import Path
from LCNN import LCNN
from WaveFakeLoader import create_loaders_from_splits, WaveFakeDatasetFixed
from train import LFCCLCNNTrainer
import numpy as np
import os
from feature_extraction import LFCCExtractor

def get_vocoder_list(splits_json):
    """Get list of all vocoders"""
    with open(splits_json, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    return list(splits['vocoders'].keys())

def scenario_b_lovo(splits_json, device='cuda', start_idx=1, end_idx=None):
    """
    Train on N-1 vocoders, test on held-out vocoder.
    start_idx, end_idx: 1-based indices for which vocoders to run.
    """
    print("\n" + "="*70)
    print("SCENARIO B: LEAVE-ONE-VOCODER-OUT (LOVO)")
    print("="*70)
    
    vocoders = get_vocoder_list(splits_json)
    num_total = len(vocoders)
    
    if end_idx is None:
        end_idx = num_total
        
    print(f"\n[VOCODER LIST] Total vocoders: {num_total}")
    for i, v in enumerate(vocoders, 1):
        status = " (RUNNING)" if start_idx <= i <= end_idx else ""
        print(f"  {i}. {v}{status}")
    print("="*70)
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(script_dir, 'weights', 'protocol', 'lovo_cumulative_results.json')
    
    # Load existing results if any
    results = {}
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except:
            pass

    # Master results file
    results_file = os.path.join(script_dir, 'weights', 'protocol', 'lovo_results.json')
    
    # Load existing results if any (allows resuming across nights)
    results = {}
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
        except:
            pass

    for fold_idx_0, held_out_vocoder in enumerate(vocoders):
        fold_idx_1 = fold_idx_0 + 1
        
        # Skip if outside requested range
        if fold_idx_1 < start_idx or fold_idx_1 > end_idx:
            continue
            
        print(f"\n--- FOLD {fold_idx_1}/{num_total}: Holding out {held_out_vocoder} ---")
        
        # Folders for weights (keep .pt files organized, but results go to one JSON)
        save_dir = os.path.join(script_dir, 'weights', 'protocol', f'lovo_{held_out_vocoder}')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'lovo_{held_out_vocoder}_best.pt')

        # Train on all except held_out
        train_vocoders = [v for v in vocoders if v != held_out_vocoder]
        train_loader, _ = create_loaders_from_splits(
            splits_json=splits_json,
            vocoders_train=train_vocoders,
            vocoders_test=train_vocoders,
            batch_size=64
        )
        
        # Evaluate on the FULL held-out vocoder
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
        
        # Calculate class weights
        train_dataset = train_loader.dataset
        fake_count = train_dataset.fake_count
        real_count = train_dataset.real_count
        total = fake_count + real_count
        class_weights = torch.tensor([total/(2*fake_count), total/(2*real_count)], dtype=torch.float32).to(device)
        
        # Train model
        model = LCNN(n_lfcc=60, num_classes=2)
        lfcc_extractor = LFCCExtractor(n_lfcc=60, n_filter=60)
        
        trainer = LFCCLCNNTrainer(
            model=model,
            lfcc_extractor=lfcc_extractor,
            train_loader=train_loader,
            val_loader=test_loader,
            device=device,
            lr=0.0001,
            class_weights=class_weights
        )
        
        # Train (trainer will save a training_metrics.json in save_dir, but we focus on the master one)
        trainer.train(num_epochs=30, save_path=save_path)
        
        # Final Verification
        _, acc, auc, eer = trainer.validate(desc='Final Eval')
        
        # Update MASTER results
        results[held_out_vocoder] = {
            'eer': float(eer),
            'accuracy': float(acc),
            'auc': float(auc),
            'fold': fold_idx_1
        }
        
        # Overwrite the SINGLE master results file
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)
        print(f"\n✓ Master results updated: {results_file}")
    
    # Final Summary
    print("\n" + "="*70)
    print("SCENARIO B CUMULATIVE RESULTS (LOVO)")
    print("="*70)
    print(f"{'Held-out Vocoder':<30} {'EER':<10} {'Accuracy':<12}")
    print("-"*70)
    for vocoder in sorted(results.keys()):
        r = results[vocoder]
        print(f"{vocoder:<30} {r['eer']:>7.4f}%  {r['accuracy']*100:>9.2f}%")
    
    if len(results) == num_total:
        avg_eer = np.mean([r['eer'] for r in results.values()])
        print("-"*70)
        print(f"{'AVERAGE (aEER)':<30} {avg_eer:>7.4f}%")
    print("="*70)
    
    # Print summary of all available results
    print("\n" + "="*70)
    print("SCENARIO B CUMULATIVE RESULTS")
    print("="*70)
    print(f"{'Held-out Vocoder':<30} {'EER':<10} {'Accuracy':<12}")
    print("-"*70)
    for vocoder in sorted(results.keys()):
        r = results[vocoder]
        print(f"{vocoder:<30} {r['eer']:>7.4f}%  {r['accuracy']*100:>9.2f}%")
    
    if len(results) == num_total:
        avg_eer = np.mean([r['eer'] for r in results.values()])
        print("-"*70)
        print(f"{'AVERAGE (aEER)':<30} {avg_eer:>7.4f}%")
    print("="*70)

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Scenario B (LOVO) in batches.')
    parser.add_argument('--start', type=int, default=1, help='Start fold index (1-based)')
    parser.add_argument('--end', type=int, default=10, help='End fold index (1-based)')
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    splits_json = os.path.join(script_dir, 'wavefake_splits.json')
    
    if not os.path.exists(splits_json):
        print(f"Error: {splits_json} not found.")
    else:
        scenario_b_lovo(splits_json, device, start_idx=args.start, end_idx=args.end)
