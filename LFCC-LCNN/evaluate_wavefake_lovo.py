import torch
import json
import os
import numpy as np
from LCNN import LCNN
from feature_extraction import LFCCExtractor
from WaveFakeLoader import WaveFakeDatasetFixed
from train import LFCCLCNNTrainer

# Official 6-Fold LOO Configuration
OFFICIAL_FOLDS = [
    "Fold1_MelGAN", "Fold2_MelGAN_Large", "Fold3_FB_MelGAN",
    "Fold4_MB_MelGAN", "Fold5_HiFiGAN", "Fold6_ParallelWaveGAN"
]

def evaluate_official_benchmarks(splits_json, device='cuda'):
    print("OFFICIAL WAVEFAKE BENCHMARK EVALUATION")
    script_dir = os.path.dirname(os.path.abspath(__file__))
    results = {}

    # 1. Evaluate ID Baseline
    id_weight = os.path.join(script_dir, 'weights', 'protocol', 'id_baseline', 'id_baseline_best.pt')
    if os.path.exists(id_weight):
        print("\n--- Evaluating ID Baseline (In-Distribution) ---")
        checkpoint = torch.load(id_weight, map_location=device, weights_only=False)
        model = LCNN(n_lfcc=60, num_classes=2).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        test_dataset = WaveFakeDatasetFixed(
            splits_json=splits_json, split_type='test', vocoders_to_include=None, # All
            include_real=True, lfcc_extractor=LFCCExtractor(n_lfcc=60, n_filter=60)
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        trainer = LFCCLCNNTrainer(model=model, lfcc_extractor=LFCCExtractor(n_lfcc=60, n_filter=60),
                                  train_loader=None, val_loader=test_loader, device=device)
        _, acc, _, eer, _, _ = trainer.validate(desc='ID Baseline Eval')
        results['ID_Baseline'] = {'eer': eer, 'acc': acc}
        print(f"  ID Baseline Result -> EER: {eer:.4f}%, Acc: {acc*100:.2f}%")

    # 2. Evaluate 6 LOO Folds
    loo_results = {}
    print("\n--- Evaluating 6-Fold LOO (Architecture-Agnostic) ---")
    for fold in OFFICIAL_FOLDS:
        weight_path = os.path.join(script_dir, 'weights', 'protocol', fold, f'{fold}_best.pt')
        if not os.path.exists(weight_path):
            continue
            
        print(f"  Evaluating {fold}...")
        checkpoint = torch.load(weight_path, map_location=device, weights_only=False)
        model = LCNN(n_lfcc=60, num_classes=2).to(device)
        model.load_state_dict(checkpoint['model_state_dict'])
        
        # Determine test vocoders from the fold name
        # Mapping back from fold name to vocoder key
        mapping = {
            "Fold1_MelGAN": ["ljspeech_melgan"],
            "Fold2_MelGAN_Large": ["ljspeech_melgan_large"],
            "Fold3_FB_MelGAN": ["ljspeech_full_band_melgan"],
            "Fold4_MB_MelGAN": ["ljspeech_multi_band_melgan"],
            "Fold5_HiFiGAN": ["ljspeech_hifiGAN"],
            "Fold6_ParallelWaveGAN": ["ljspeech_parallel_wavegan"]
        }
        test_vocoders = mapping[fold]
        
        test_dataset = WaveFakeDatasetFixed(
            splits_json=splits_json, split_type='test', vocoders_to_include=test_vocoders,
            include_real=True, lfcc_extractor=LFCCExtractor(n_lfcc=60, n_filter=60)
        )
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)
        
        trainer = LFCCLCNNTrainer(model=model, lfcc_extractor=LFCCExtractor(n_lfcc=60, n_filter=60),
                                  train_loader=None, val_loader=test_loader, device=device)
        _, acc, _, eer, _, _ = trainer.validate(desc=f'{fold} Eval')
        loo_results[fold] = {'eer': eer, 'acc': acc}

    # Print Summary
    print("\n--- FINAL BENCHMARK SUMMARY ---")
    if 'ID_Baseline' in results:
        print(f"ID Baseline EER: {results['ID_Baseline']['eer']:.4f}%")
        
    if loo_results:
        avg_eer = np.mean([v['eer'] for v in loo_results.values()])
        print(f"Average LOO EER (6-fold): {avg_eer:.4f}%")
        for f, r in loo_results.items():
            print(f"  - {f}: {r['eer']:.4f}%")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    splits_json = os.path.join(script_dir, 'wavefake_splits.json')
    if os.path.exists(splits_json):
        evaluate_official_benchmarks(splits_json, device)
