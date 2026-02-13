"""
Replicate exact WaveFake paper evaluation protocol
Implements all evaluation scenarios from the paper
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

# ============================================================
# SCENARIO A: Single-Vocoder Training (In-Distribution)
# ============================================================
def scenario_a_single_vocoder(splits_json, device='cuda'):
    """
    Train on each vocoder individually, test on its hold-out
    Paper Table 2: "in-distribution" results
    """
    print("\n" + "="*70)
    print("SCENARIO A: SINGLE-VOCODER TRAINING (IN-DISTRIBUTION)")
    print("="*70)
    
    vocoders = get_vocoder_list(splits_json)
    results = {}
    
    for vocoder in vocoders:
        print(f"\n--- Training on: {vocoder} ---")
        
        # Create loaders (train on vocoder, test on vocoder)
        train_loader, test_loader = create_loaders_from_splits(
            splits_json=splits_json,
            vocoders_train=[vocoder],
            vocoders_test=[vocoder],
            batch_size=64
        )
        
        # Train model
        model = LCNN(n_lfcc=60, num_classes=2)
        lfcc_extractor = LFCCExtractor(n_lfcc=60, n_filter=60)
        
        trainer = LFCCLCNNTrainer(
            model=model,
            lfcc_extractor=lfcc_extractor,
            train_loader=train_loader,
            val_loader=test_loader,
            device=device,
            lr=0.0001
        )
        
        os.makedirs(f'weights/protocol/scenario_a/{vocoder}', exist_ok=True)
        save_path = f'weights/protocol/scenario_a/{vocoder}/{vocoder}.pt'
        trainer.train(num_epochs=30, save_path=save_path)
        
        # Evaluate
        _, _, _, eer = trainer.validate()
        results[vocoder] = eer
        print(f"✓ {vocoder} EER: {eer:.4f}%")
    
    # Print summary
    print("\n" + "="*70)
    print("SCENARIO A RESULTS: IN-DISTRIBUTION PER-VOCODER EER")
    print("="*70)
    for vocoder, eer in sorted(results.items()):
        print(f"{vocoder:40s} {eer:7.4f}%")
    print(f"{'Average':40s} {np.mean(list(results.values())):7.4f}%")
    print("="*70)
    
    return results

# ============================================================
# SCENARIO B: Leave-One-Vocoder-Out (LOVO)
# ============================================================
def scenario_b_lovo(splits_json, device='cuda'):
    """
    Train on N-1 vocoders, test on held-out vocoder
    Paper Table 3: "LOVO" results
    """
    print("\n" + "="*70)
    print("SCENARIO B: LEAVE-ONE-VOCODER-OUT (LOVO)")
    print("="*70)
    
    vocoders = get_vocoder_list(splits_json)
    results = {}
    
    for held_out_vocoder in vocoders:
        print(f"\n--- Holding out: {held_out_vocoder} ---")
        
        # Train on all except held_out
        train_vocoders = [v for v in vocoders if v != held_out_vocoder]
        
        train_loader, _ = create_loaders_from_splits(
            splits_json=splits_json,
            vocoders_train=train_vocoders,
            vocoders_test=train_vocoders,  # dummy, not used
            batch_size=64
        )
        
        # Evaluate on the FULL held-out vocoder (all data per WaveFake protocol)
        from WaveFakeLoader import WaveFakeDatasetFixed
        test_dataset = WaveFakeDatasetFixed(
            splits_json=splits_json,
            split_type='all',
            vocoders_to_include=[held_out_vocoder],
            include_real=True,
            lfcc_extractor=LFCCExtractor(n_lfcc=60, n_filter=60),
            noise_std=0.0
        )
        from torch.utils.data import DataLoader as DL
        test_loader = DL(test_dataset, batch_size=64, shuffle=False, num_workers=4, pin_memory=True)
        
        # Train model
        model = LCNN(n_lfcc=60, num_classes=2)
        lfcc_extractor = LFCCExtractor(n_lfcc=60, n_filter=60)
        
        trainer = LFCCLCNNTrainer(
            model=model,
            lfcc_extractor=lfcc_extractor,
            train_loader=train_loader,
            val_loader=test_loader,
            device=device,
            lr=0.0001
        )
        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        save_dir = os.path.join(script_dir, 'weights', 'protocol', 'scenario_b', held_out_vocoder)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{held_out_vocoder}.pt')
        trainer.train(num_epochs=30, save_path=save_path)
        
        # Evaluate on held-out vocoder
        _, _, _, eer = trainer.validate()
        results[held_out_vocoder] = eer
        print(f"✓ Held-out {held_out_vocoder} EER: {eer:.4f}%")
    
    # Print summary
    print("\n" + "="*70)
    print("SCENARIO B RESULTS: LEAVE-ONE-VOCODER-OUT EER")
    print("="*70)
    for vocoder, eer in sorted(results.items()):
        print(f"Held out: {vocoder:30s} EER: {eer:7.4f}%")
    print(f"{'Average (aEER)':40s} {np.mean(list(results.values())):7.4f}%")
    print("="*70)
    
    return results

# ============================================================
# SCENARIO C: Multi-Vocoder Training
# ============================================================
def scenario_c_multi_vocoder(splits_json, device='cuda'):
    """
    Train on all vocoders, test on combined hold-out
    This is what you've been doing!
    """
    print("\n" + "="*70)
    print("SCENARIO C: MULTI-VOCODER TRAINING (ALL VOCODERS)")
    print("="*70)
    
    # Train on all vocoders
    train_loader, test_loader = create_loaders_from_splits(
        splits_json=splits_json,
        vocoders_train=None,  # All vocoders
        vocoders_test=None,   # All vocoders
        batch_size=64
    )
    
    # Train model
    model = LCNN(n_lfcc=60, num_classes=2)
    lfcc_extractor = LFCCExtractor(n_lfcc=60, n_filter=60)
    
    # Calculate class weights
    train_dataset = train_loader.dataset
    fake_count = train_dataset.fake_count
    real_count = train_dataset.real_count
    total = len(train_dataset)
    
    weight_fake = total / (2 * fake_count)
    weight_real = total / (2 * real_count)
    class_weights = torch.tensor([weight_fake, weight_real], dtype=torch.float32).to(device)
    
    trainer = LFCCLCNNTrainer(
        model=model,
        lfcc_extractor=lfcc_extractor,
        train_loader=train_loader,
        val_loader=test_loader,
        device=device,
        lr=0.0001,
        class_weights=class_weights
    )
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, 'weights', 'protocol')
    os.makedirs(save_dir, exist_ok=True)
    
    # Consistent weight path: check main_weights first, then protocol dir
    weight_path_main = os.path.join(script_dir, 'weights', 'main_weights', 'best_lfcc_lcnn_wavefake.pt')
    weight_path_protocol = os.path.join(save_dir, 'scenario_c_all_vocoders.pt')
    
    if os.path.exists(weight_path_main):
        print(f"Loading existing weights from {weight_path_main}...")
        checkpoint = torch.load(weight_path_main, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        lfcc_extractor.load_state_dict(checkpoint['lfcc_extractor_state_dict'])
    elif os.path.exists(weight_path_protocol):
        print(f"Loading existing weights from {weight_path_protocol}...")
        checkpoint = torch.load(weight_path_protocol, map_location=device)
        model.load_state_dict(checkpoint['model_state_dict'])
        lfcc_extractor.load_state_dict(checkpoint['lfcc_extractor_state_dict'])
    else:
        print(f"No existing weights found. Starting training...")
        trainer.train(num_epochs=50, save_path=weight_path_protocol)
    
    # Evaluate
    _, acc, _, eer = trainer.validate()
    
    print("\n" + "="*70)
    print("SCENARIO C RESULTS: ALL-VOCODERS TRAINING")
    print("="*70)
    print(f"Test Accuracy: {acc*100:.2f}%")
    print(f"Test EER:      {eer:.4f}%")
    print("="*70)
    
    return {'accuracy': acc, 'eer': eer}

# ============================================================
# SCENARIO D: Cross-Vocoder Matrix
# ============================================================
def scenario_d_cross_vocoder_matrix(splits_json, device='cuda'):
    """
    Train on vocoder X, test on vocoder Y for all X,Y
    Paper Table 4-5: Cross-vocoder generalization matrix
    """
    print("\n" + "="*70)
    print("SCENARIO D: CROSS-VOCODER GENERALIZATION MATRIX")
    print("="*70)
    
    vocoders = get_vocoder_list(splits_json)
    matrix = {}
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    for train_vocoder in vocoders:
        matrix[train_vocoder] = {}
        
        # Train once on train_vocoder
        print(f"\n--- Training on: {train_vocoder} ---")
        
        # Use train split for training, test split for validation/early stopping
        train_loader, val_loader = create_loaders_from_splits(
            splits_json=splits_json,
            vocoders_train=[train_vocoder],
            vocoders_test=[train_vocoder],
            batch_size=64
        )
        
        model = LCNN(n_lfcc=60, num_classes=2)
        lfcc_extractor = LFCCExtractor(n_lfcc=60, n_filter=60)
        
        trainer = LFCCLCNNTrainer(
            model=model,
            lfcc_extractor=lfcc_extractor,
            train_loader=train_loader,
            val_loader=val_loader,  # Proper test split for early stopping
            device=device,
            lr=0.0001
        )
        
        save_dir = os.path.join(script_dir, 'weights', 'protocol', 'scenario_d', train_vocoder)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{train_vocoder}.pt')
        trainer.train(num_epochs=30, save_path=save_path)
        
        # Test on all vocoders (use test split)
        for test_vocoder in vocoders:
            _, test_loader = create_loaders_from_splits(
                splits_json=splits_json,
                vocoders_train=[test_vocoder],  # Dummy
                vocoders_test=[test_vocoder],
                batch_size=64
            )
            
            trainer.val_loader = test_loader
            _, _, _, eer = trainer.validate()
            matrix[train_vocoder][test_vocoder] = eer
            print(f"  Test on {test_vocoder}: {eer:.4f}% EER")
    
    # Print matrix
    print("\n" + "="*70)
    print("CROSS-VOCODER GENERALIZATION MATRIX (EER %)")
    print("="*70)
    print(f"{'Train \\ Test':<25}", end='')
    for v in vocoders:
        print(f"{v[:10]:>12}", end='')
    print()
    print("-" * 70)
    
    for train_v in vocoders:
        print(f"{train_v:<25}", end='')
        for test_v in vocoders:
            print(f"{matrix[train_v][test_v]:11.4f}%", end='')
        print()
    print("="*70)
    
    return matrix

# ============================================================
# MAIN: Run All Scenarios
# ============================================================
def run_complete_evaluation(splits_json_name='wavefake_splits.json'):
    """
    Run ALL WaveFake paper evaluation scenarios
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    splits_json = os.path.join(script_dir, splits_json_name)
    
    if not os.path.exists(splits_json):
        print(f"Error: {splits_json} not found. Run create_wavefake_splits.py first.")
        return

    results = {}
    
    # We allow the user to choose which scenario to run by toggling these or passing args
    # For now, we'll provide the logic and a way to run them selectively
    
    # To run a scenario, uncomment it:
    # results['scenario_a'] = scenario_a_single_vocoder(splits_json, device)
    results['scenario_b'] = scenario_b_lovo(splits_json, device)
    # results['scenario_c'] = scenario_c_multi_vocoder(splits_json, device)
    # results['scenario_d'] = scenario_d_cross_vocoder_matrix(splits_json, device)
    
    output_results = os.path.join(script_dir, 'wavefake_protocol_results.json')
    # Save all results
    with open(output_results, 'w') as f:
        json.dump(results, f, indent=2)
    
    print(f"\n✓ All results saved to {output_results}")

if __name__ == '__main__':
    run_complete_evaluation()
