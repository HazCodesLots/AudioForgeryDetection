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

# Official 6-Fold LOO Configuration (Strict LJSpeech Only)
OFFICIAL_6_FOLDS = {
    "Fold1_MelGAN": {
        "test": ['ljspeech_melgan'],
        "train": ['ljspeech_melgan_large', 'ljspeech_multi_band_melgan', 'ljspeech_full_band_melgan', 'ljspeech_hifiGAN', 'ljspeech_parallel_wavegan', 'ljspeech_waveglow']
    },
    "Fold2_MelGAN_Large": {
        "test": ['ljspeech_melgan_large'],
        "train": ['ljspeech_melgan', 'ljspeech_full_band_melgan', 'ljspeech_hifiGAN', 'ljspeech_parallel_wavegan', 'ljspeech_waveglow']
    },
    "Fold3_FB_MelGAN": {
        "test": ['ljspeech_full_band_melgan'],
        "train": ['ljspeech_melgan', 'ljspeech_melgan_large', 'ljspeech_multi_band_melgan', 'ljspeech_hifiGAN', 'ljspeech_parallel_wavegan', 'ljspeech_waveglow']
    },
    "Fold4_MB_MelGAN": {
        "test": ['ljspeech_multi_band_melgan'],
        "train": ['ljspeech_melgan', 'ljspeech_melgan_large', 'ljspeech_full_band_melgan', 'ljspeech_hifiGAN', 'ljspeech_parallel_wavegan', 'ljspeech_waveglow']
    },
    "Fold5_HiFiGAN": {
        "test": ['ljspeech_hifiGAN'],
        "train": ['ljspeech_melgan', 'ljspeech_melgan_large', 'ljspeech_full_band_melgan', 'ljspeech_multi_band_melgan', 'ljspeech_parallel_wavegan', 'ljspeech_waveglow']
    },
    "Fold6_ParallelWaveGAN": {
        "test": ['ljspeech_parallel_wavegan'],
        "train": ['ljspeech_melgan', 'ljspeech_melgan_large', 'ljspeech_full_band_melgan', 'ljspeech_multi_band_melgan', 'ljspeech_hifiGAN', 'ljspeech_waveglow']
    }
}

def scenario_b_official_loo(splits_json, device='cuda', start_idx=1, end_idx=None):
    """
    Official Scenario B: 6-Fold Leave-One-Out (LOO) with 3-way Split
    """
    print("SCENARIO B: OFFICIAL 6-FOLD LEAVE-ONE-OUT (LOO)")
    
    fold_keys = sorted(list(OFFICIAL_6_FOLDS.keys()))
    num_total = len(fold_keys)
    
    if end_idx is None:
        end_idx = num_total
        
    print(f"\n[PROTOCOL] 6 Official Folds detected. (3-Way Split & Balanced Testing)")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results_file = os.path.join(script_dir, 'weights', 'protocol', 'loo_6fold_official_results.json')
    
    # JSON Persistence: Load existing results
    results = {}
    if os.path.exists(results_file):
        try:
            with open(results_file, 'r') as f:
                results = json.load(f)
            print(f"  Loaded {len(results)} existing fold results from JSON.")
        except Exception as e:
            print(f"  Warning: Could not load existing results: {e}")

    lfcc_extractor = LFCCExtractor(
        sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_lfcc=60, n_filter=60
    )

    for fold_idx_0, fold_name in enumerate(fold_keys):
        fold_idx_1 = fold_idx_0 + 1
        if fold_idx_1 < start_idx or fold_idx_1 > end_idx:
            continue
            
        config = OFFICIAL_6_FOLDS[fold_name]
        held_out_vocoders = config['test']
        train_vocoders = config['train']
        
        print(f"\n--- {fold_name} ({fold_idx_1}/{num_total}) ---")
        
        save_dir = os.path.join(script_dir, 'weights', 'protocol', fold_name)
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f'{fold_name}_best.pt')

        # 3-WAY SPLIT IMPLEMENTATION
        # 1. True Train: 'train' split of the training vocoders
        train_dataset = WaveFakeDatasetFixed(
            splits_json=splits_json, split_type='train', vocoders_to_include=train_vocoders,
            include_real=True, lfcc_extractor=lfcc_extractor
        )
        # 2. Validation: 'test' split of the SAME training vocoders (In-Distribution Validation)
        val_dataset = WaveFakeDatasetFixed(
            splits_json=splits_json, split_type='test', vocoders_to_include=train_vocoders,
            include_real=True, lfcc_extractor=lfcc_extractor, balance_classes=True
        )
        # 3. Final Test: 'test' split of the HELD-OUT vocoder (Unseen Architecture Test)
        test_dataset = WaveFakeDatasetFixed(
            splits_json=splits_json, split_type='test', vocoders_to_include=held_out_vocoders,
            include_real=True, lfcc_extractor=lfcc_extractor, balance_classes=True
        )

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False, num_workers=4)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False, num_workers=4)
        
        print(f"  Split Summary: Train={len(train_dataset)}, Val={len(val_dataset)}, Test={len(test_dataset)}")

        fake_count = train_dataset.fake_count
        real_count = train_dataset.real_count
        total = fake_count + real_count
        class_weights = torch.tensor([total/(2*fake_count), total/(2*real_count)], dtype=torch.float32).to(device)
        
        model = LCNN(n_lfcc=60, num_classes=2)
        
        trainer = LFCCLCNNTrainer(
            model=model, lfcc_extractor=lfcc_extractor, train_loader=train_loader,
            val_loader=val_loader, device=device, lr=0.0001, class_weights=class_weights
        )
        
        # Train using the In-Distribution Validation set
        trainer.train(num_epochs=30, save_path=save_path)
        
        # FINAL BENCHMARK on Unseen Architecture
        print(f"\n[FINAL BENCHMARK] Evaluating on held-out {held_out_vocoders}...")
        _, acc, auc, eer, _, _ = trainer.validate(desc='Unseen Eval', loader=test_loader)
        
        results[fold_name] = {'eer': float(eer), 'accuracy': float(acc), 'auc': float(auc)}
        with open(results_file, 'w') as f:
            json.dump(results, f, indent=4)

    print("\nOFFICIAL 6-FOLD LOO FINAL RESULTS (BALANCED & PROTECTED)")
    eers = [r['eer'] for r in results.values()]
    if eers:
        print(f"Average LOO EER: {np.mean(eers):.4f}%")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Run Official 6-Fold LOO.')
    parser.add_argument('--start', type=int, default=1)
    parser.add_argument('--end', type=int, default=6)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    splits_json = os.path.join(script_dir, 'wavefake_splits.json')
    
    if os.path.exists(splits_json):
        scenario_b_official_loo(splits_json, device, start_idx=args.start, end_idx=args.end)