import torch
import torch.nn as nn
import os
import sys
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
import importlib.util

# Add current directory to path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

# Import logic from main script
def import_module_by_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

script_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "ConvNeXt_SONYC-UST.py")
model_script = import_module_by_path("ConvNeXt_SONYC_UST", script_path)

DeepFakeDetectionModel = model_script.DeepFakeDetectionModel
EnhancedAudioFrontend = model_script.EnhancedAudioFrontend
SONYCUSTDataset = model_script.SONYCUSTDataset
pad_truncate_collate = model_script.pad_truncate_collate

def inspect(dataset_path):
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    # Setup Data
    audio_dir = dataset_path
    csv_path = os.path.join(dataset_path, 'annotations.csv')
    frontend = EnhancedAudioFrontend(n_mels=128)
    val_dataset = SONYCUSTDataset(csv_path, audio_dir, split='validate', frontend=frontend, limit_samples=None)
    val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False, collate_fn=pad_truncate_collate)

    # Setup Model
    convnext_params = {
        'input_channels': 1,
        'depths': [2, 2, 6, 2],
        'dims': [64, 128, 256, 512],
        'drop_path_rate': 0.2,
        'layer_scale_init_value': 1e-6
    }
    attention_params = {
        'input_dim': 512,
        'attention_dim': 256,
        'num_heads': 4,
        'dropout_rate': 0.15
    }
    mlp_params = {
        'input_dim': 256,
        'num_classes': 23,
        'dropout_rate': 0.3
    }

    model = DeepFakeDetectionModel(convnext_params, attention_params, mlp_params)
    
    # Load Weights
    results_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
    weights_path = os.path.join(results_dir, 'model_epoch_1.pth')
    
    if not os.path.exists(weights_path):
        print(f"Error: Weights not found at {weights_path}")
        return

    print(f"Loading weights from: {weights_path}")
    state_dict = torch.load(weights_path, map_location=device)
    model.load_state_dict(state_dict)
    model.to(device)
    model.eval()

    # User's Inspection Loop
    print("\n--- Model Prediction Inspection ---")
    with torch.no_grad():
        for inputs, labels, _ in val_loader:
            inputs = inputs.to(device)
            outputs = model(inputs)
            probs = torch.sigmoid(outputs)
            
            # Check first 3 samples
            for i in range(min(3, len(probs))):
                print(f"\nSample {i}:")
                # print(f"  Labels:     {labels[i].numpy()}")
                print(f"  Label sum:  {labels[i].sum().item()}")
                # print(f"  Probs:      {probs[i].cpu().numpy()}")
                print(f"  Prob max:   {probs[i].max().item():.4f}")
            
            print(f"\n=== Overall Batch Stats ===")
            print(f"Pred mean:  {probs.mean():.4f}")
            print(f"Pred max:   {probs.max():.4f}")
            print(f"Preds>0.5:  {(probs > 0.5).sum().item()} / {probs.numel()}")
            print(f"Labels sum: {labels.sum().item()} / {labels.numel()}")
            break

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset_path', type=str, required=True)
    args = parser.parse_args()
    
    inspect(args.dataset_path)
