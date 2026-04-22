import torch
import json
import os
import random
from LCNN import LCNN
from WaveFakeLoader import WaveFakeDatasetFixed
from train import LFCCLCNNTrainer
from feature_extraction import LFCCExtractor

def train_id_baseline(splits_json, device='cuda'):
    """
    In-Distribution (ID) Baseline with Validation Protection
    """
    print("SCENARIO A: IN-DISTRIBUTION (ID) BASELINE (Protected)")
    
    script_dir = os.path.dirname(os.path.abspath(__file__))
    save_dir = os.path.join(script_dir, 'weights', 'protocol', 'id_baseline')
    os.makedirs(save_dir, exist_ok=True)
    save_path = os.path.join(save_dir, 'id_baseline_best.pt')
    results_file = os.path.join(save_dir, 'id_baseline_results.json')
    
    lfcc_extractor = LFCCExtractor(
        sample_rate=16000, n_fft=512, win_length=400, hop_length=160, n_lfcc=60, n_filter=60
    )

    train_dataset = WaveFakeDatasetFixed(
        splits_json=splits_json, split_type='train', vocoders_to_include=None,
        include_real=True, lfcc_extractor=lfcc_extractor
    )

    full_test_dataset = WaveFakeDatasetFixed(
        splits_json=splits_json, split_type='test', vocoders_to_include=None,
        include_real=True, lfcc_extractor=lfcc_extractor, balance_classes=True
    )

    random.seed(42)
    indices = list(range(len(full_test_dataset.samples)))
    random.shuffle(indices)
    split_idx = int(len(indices) * 0.5)
    
    val_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    original_samples = full_test_dataset.samples
    full_test_dataset.samples = [original_samples[i] for i in val_indices]
    val_dataset = full_test_dataset
    
    test_dataset = WaveFakeDatasetFixed(
        splits_json=splits_json, split_type='test', vocoders_to_include=None,
        include_real=True, lfcc_extractor=lfcc_extractor, balance_classes=True
    )
    test_dataset.samples = [original_samples[i] for i in test_indices]

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=64, shuffle=True, num_workers=4)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=64, shuffle=False)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=64, shuffle=False)

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
    
    trainer.train(num_epochs=30, save_path=save_path)
    
    print("\n[FINAL BENCHMARK] Evaluating on protected ID Test set...")
    _, acc, auc, eer, _, _ = trainer.validate(desc='ID Final Eval', loader=test_loader)
    
    results = {'id_baseline': {'eer': float(eer), 'accuracy': float(acc), 'auc': float(auc)}}
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=4)
    
    print(f"\n✓ ID Baseline training complete. Results: {results_file}")

if __name__ == '__main__':
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    script_dir = os.path.dirname(os.path.abspath(__file__))
    splits_json = os.path.join(script_dir, 'wavefake_splits.json')
    
    if os.path.exists(splits_json):
        train_id_baseline(splits_json, device)
    else:
        print(f"Error: {splits_json} not found.")
