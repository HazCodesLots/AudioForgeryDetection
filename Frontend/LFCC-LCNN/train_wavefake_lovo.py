"""
Training Script for Scenario B: Leave-One-Vocoder-Out (LOVO)
Trains 10 separate models (30 epochs each) to test generalization.
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
    with open(splits_json, 'r', encoding='utf-8') as f:
        splits = json.load(f)
    return list(splits['vocoders'].keys())

def scenario_b_lovo(splits_json, device='cuda'):
    """
    Train on N-1 vocoders, test on held-out vocoder
    """
    print("\n" + "="*70)
    print("SCENARIO B: LEAVE-ONE-VOCODER-OUT (LOVO)")
    print("="*70)
    
    vocoders = get_vocoder_list(splits_json)
    print(f"\n[VOCODER LIST] Total vocoders: {len(vocoders)}")
    for i, v in enumerate(vocoders, 1):
        print(f"  {i}. {v}")
    print("="*70)
    
    results = {}
    
    for fold_idx, held_out_vocoder in enumerate(vocoders, 1):
        print(f"\n--- FOLD {fold_idx}/{len(vocoders)}: Holding out {held_out_vocoder} ---")
        
        # Train on all except held_out (use train split only)
        train_vocoders = [v for v in vocoders if v != held_out_vocoder]
        
        train_loader, _ = create_loaders_from_splits(
            splits_json=splits_json,
            vocoders_train=train_vocoders,
            vocoders_test=train_vocoders,  # dummy, not used
            batch_size=64
        )
        
        # Evaluate on the FULL held-out vocoder (all data, not just 20% test split)
        # WaveFake paper protocol: LOVO tests on the entire held-out vocoder output
        from WaveFakeLoader import WaveFakeDatasetFixed
        test_dataset = WaveFakeDatasetFixed(
            splits_json=splits_json,
            split_type='all',  # combine train+test = 100% of held-out vocoder
            vocoders_to_include=[held_out_vocoder],
            include_real=True,
            lfcc_extractor=LFCCExtractor(n_lfcc=60, n_filter=60),
            noise_std=0.0  # No noise during evaluation
        )
        test_loader = torch.utils.data.DataLoader(
            test_dataset, batch_size=64, shuffle=False,
            num_workers=4, pin_memory=True
        )
        
        # Verify data integrity
        print(f"[DATA INTEGRITY CHECK]")
        train_size = len(train_loader.dataset)
        test_size = len(test_loader.dataset)
        print(f"  Train set size: {train_size}")
        print(f"  Test set size (held-out vocoder): {test_size}")
        
        # Calculate class weights for this fold
        train_dataset = train_loader.dataset
        fake_count = train_dataset.fake_count
        real_count = train_dataset.real_count
        total = fake_count + real_count
        
        print(f"[CLASS DISTRIBUTION]")
        print(f"  Total: {total}")
        print(f"  Real: {real_count} ({real_count/total*100:.2f}%)")
        print(f"  Fake: {fake_count} ({fake_count/total*100:.2f}%)")
        
        weight_fake = total / (2 * fake_count)
        weight_real = total / (2 * real_count)
        class_weights = torch.tensor([weight_fake, weight_real], dtype=torch.float32).to(device)
        print(f"[CLASS WEIGHTS]")
        print(f"  Fake weight: {weight_fake:.4f}")
        print(f"  Real weight: {weight_real:.4f}")
        
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
        
        # Folders for weights
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, 'weights', 'protocol', f'lovo_{held_out_vocoder}')
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'lovo_{held_out_vocoder}_best.pt')
        
        print(f"[TRAINING CONFIG]")
        print(f"  Epochs: 30")
        print(f"  Learning rate: 0.0001")
        print(f"  Save path: {save_path}")
        print(f"[STARTING TRAINING]")
        
        trainer.train(num_epochs=30, save_path=save_path)
        
        # Evaluate on held-out vocoder
        print(f"[FINAL EVALUATION ON HELD-OUT VOCODER: {held_out_vocoder}]")
        _, acc, auc, eer = trainer.validate(desc='Final Eval')
        results[held_out_vocoder] = {
            'eer': eer,
            'accuracy': acc,
            'auc': auc,
            'train_size': train_size,
            'test_size': test_size
        }
        print(f"✓ Results - EER: {eer:.4f}%, Acc: {acc*100:.2f}%, AUC: {auc:.4f}")
    
    # Print summary
    print("\n" + "="*70)
    print("SCENARIO B RESULTS: LEAVE-ONE-VOCODER-OUT (LOVO)")
    print("="*70)
    print(f"{'Held-out Vocoder':<30} {'EER':<10} {'Accuracy':<12} {'AUC':<10}")
    print("-"*70)
    for vocoder in sorted(results.keys()):
        eer = results[vocoder]['eer']
        acc = results[vocoder]['accuracy']
        auc = results[vocoder]['auc']
        print(f"{vocoder:<30} {eer:>7.4f}%  {acc*100:>9.2f}%  {auc:>8.4f}")
    
    avg_eer = np.mean([r['eer'] for r in results.values()])
    avg_acc = np.mean([r['accuracy'] for r in results.values()])
    avg_auc = np.mean([r['auc'] for r in results.values()])
    
    print("-"*70)
    print(f"{'AVERAGE (aEER)':<30} {avg_eer:>7.4f}%  {avg_acc*100:>9.2f}%  {avg_auc:>8.4f}")
    print("="*70)
    
    return results

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    splits_json = os.path.join(script_dir, 'wavefake_splits.json')
    
    if not os.path.exists(splits_json):
        print(f"Error: {splits_json} not found. Run create_wavefake_splits.py first.")
    else:
        scenario_b_lovo(splits_json, device)
