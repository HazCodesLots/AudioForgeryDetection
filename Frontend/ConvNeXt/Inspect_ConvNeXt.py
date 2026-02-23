import os
import argparse
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

import importlib.util

def import_module_by_path(module_name, file_path):
    spec = importlib.util.spec_from_file_location(module_name, file_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module

script_dir = os.path.dirname(os.path.abspath(__file__))
script_path = os.path.join(script_dir, "ConvNeXt_SONYC-UST.py")
model_script = import_module_by_path("ConvNeXt_SONYC_UST", script_path)

ConvNeXtTagger = model_script.ConvNeXtTagger
SONYCUSTDataset = model_script.SONYCUSTDataset
EnhancedAudioFrontend = model_script.EnhancedAudioFrontend
pad_truncate_collate = model_script.pad_truncate_collate

LABEL_NAMES = [
    "small-engine", "medium-engine", "large-engine",
    "rock-drill", "jackhammer", "hoe-ram", "pile-driver",
    "non-mach-impact", "chainsaw", "sm-med-saw", "large-saw",
    "car-horn", "car-alarm", "siren", "reverse-beeper",
    "stat-music", "mobile-music", "ice-cream-truck",
    "talking", "shouting", "large-crowd", "amplified-speech",
    "dog-barking"
]

def check_grad_flow(model):
    print("CHECK 1: WEIGHT NORMS (Pooling + Attention Layers)")
    healthy, dead = 0, 0
    for name, param in model.named_parameters():
        if 'att_pool' in name:
            norm = param.data.norm().item()
            if norm < 1e-4:
                status = "🔴 DEAD"; dead += 1
            elif norm > 50:
                status = "⚠️  LARGE"; healthy += 1
            else:
                status = "✅ OK"; healthy += 1
            print(f"  {name:<50} {norm:>8.4f}  {status}")
    print(f"\n  Result: {healthy} OK, {dead} dead layers")
    return dead == 0

def check_entropy(model, val_loader, device, n_batches=20):
    print("CHECK 2: ATTENTION ENTROPY (Freq + Time)")
    freq_weights_list = []
    time_weights_list = []

    def freq_hook(module, input, output):
        if output[1] is not None:
            freq_weights_list.append(output[1].detach().cpu())

    def time_hook(module, input, output):
        if output[1] is not None:
            time_weights_list.append(output[1].detach().cpu())

    h1 = model.att_pool.freq_attn.register_forward_hook(freq_hook)
    h2 = model.att_pool.time_attn.register_forward_hook(time_hook)

    model.eval()
    with torch.no_grad():
        for i, (mels, labels, _) in enumerate(val_loader):
            if i >= n_batches: break
            model(mels.to(device))

    h1.remove(); h2.remove()

    def entropy_from_weights(weights_list, name, seq_len):
        if not weights_list: return None
        w = torch.cat(weights_list, dim=0)
        # Global Query Attention weights are (B, 1, seq_len) and already normalized
        pool = w.squeeze(1) 
        ent  = -(pool * (pool + 1e-9).log()).sum(dim=-1).mean().item()
        max_ent = np.log(seq_len)
        pct = ent / max_ent * 100
        status = "✅ FOCUSED" if pct < 40 else ("⚠️  MODERATE" if pct < 70 else "🔴 UNIFORM")
        print(f"  {name} attention entropy: {ent:.3f} / {max_ent:.3f} ({pct:.1f}% of max)  {status}")
        return ent

    entropy_from_weights(freq_weights_list, "Freq", seq_len=8)
    entropy_from_weights(time_weights_list, "Time", seq_len=39)

def visualise_attention_maps(model, val_loader, device, save_dir, n_samples=6):
    print("CHECK 3: ATTENTION MAP VISUALISATION")
    os.makedirs(save_dir, exist_ok=True)
    freq_list, time_list = [], []
    mels_list, labels_list, preds_list = [], [], []

    def freq_hook(module, input, output):
        if output[1] is not None: freq_list.append(output[1].detach().cpu())

    def time_hook(module, input, output):
        if output[1] is not None: time_list.append(output[1].detach().cpu())

    h1 = model.att_pool.freq_attn.register_forward_hook(freq_hook)
    h2 = model.att_pool.time_attn.register_forward_hook(time_hook)
    model.eval()

    with torch.no_grad():
        for mels, labels, _ in val_loader:
            logits = model(mels.to(device))
            preds_list.append(torch.sigmoid(logits).cpu())
            mels_list.append(mels.cpu())
            labels_list.append(labels.cpu())
            if sum(m.shape[0] for m in mels_list) >= n_samples: break

    h1.remove(); h2.remove()

    mels_all = torch.cat(mels_list, dim=0)[:n_samples]
    labels_all = torch.cat(labels_list, dim=0)[:n_samples]
    preds_all = torch.cat(preds_list, dim=0)[:n_samples]

    T, F_out = 39, 8
    freq_all = torch.cat(freq_list, dim=0)[:n_samples] # (n_samples, 1, F_out)
    time_all = torch.cat(time_list, dim=0)[:n_samples * F_out] # (n_samples * F_out, 1, T)
    time_all = time_all.view(n_samples, F_out, T) # (n_samples, F_out, T)

    for i in range(n_samples):
        mel = mels_all[i, 0].numpy()
        freq_attn_map = freq_all[i, 0].numpy()
        freq_attn_map = freq_attn_map / (freq_attn_map.max() + 1e-9)
        time_attn_map = time_all[i].mean(dim=0).numpy()
        time_attn_map = time_attn_map / (time_attn_map.max() + 1e-9)
        attn_2d = np.outer(freq_attn_map, time_attn_map)
        attn_up = F.interpolate(torch.tensor(attn_2d).unsqueeze(0).unsqueeze(0).float(), size=mel.shape, mode='bilinear', align_corners=False).squeeze().numpy()

        true_idx = labels_all[i].nonzero().squeeze(-1).tolist()
        true_str = " | ".join([LABEL_NAMES[c] for c in (true_idx if isinstance(true_idx, list) else [true_idx])])
        top3 = preds_all[i].argsort(descending=True)[:3]
        top_str = " | ".join([f"{LABEL_NAMES[c]}:{preds_all[i][c]:.2f}" for c in top3])

        fig, axes = plt.subplots(1, 3, figsize=(18, 4))
        axes[0].imshow(mel, aspect='auto', origin='lower', cmap='magma')
        axes[0].set_title(f"Mel Spectrogram\nTrue: {true_str[:55]}", fontsize=8)
        axes[1].barh(range(F_out), freq_attn_map, color='steelblue')
        axes[1].set_yticks(range(F_out))
        axes[1].set_yticklabels([f"F{j}" for j in range(F_out)], fontsize=7)
        axes[2].imshow(mel, aspect='auto', origin='lower', cmap='magma', alpha=0.5)
        axes[2].imshow(attn_up, aspect='auto', origin='lower', cmap='hot', alpha=0.6)
        axes[2].set_title(f"2D Attention Overlay\nTop: {top_str[:55]}", fontsize=8)
        plt.tight_layout()
        plt.savefig(os.path.join(save_dir, f"attn_sample_{i:02d}.png"), dpi=120, bbox_inches='tight')
        plt.close()

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model_path', type=str, required=True)
    parser.add_argument('--dataset_path', type=str, required=True)
    parser.add_argument('--save_dir', type=str, default='results/attention_maps')
    parser.add_argument('--batch_size', type=int, default=8)
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    convnext_params  = {'input_channels':1,'depths':[2,2,6,2], 'dims':[64,128,256,512],'drop_path_rate':0.2, 'layer_scale_init_value':1e-6}
    attention_params = {'input_dim': 512, 'num_heads': 4}
    mlp_params       = {'input_dim': 512, 'num_classes': 23, 'dropout_rate': 0.3}

    model = ConvNeXtTagger(convnext_params, attention_params, mlp_params)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.to(device).eval()

    frontend = EnhancedAudioFrontend(n_mels=128)
    csv_path = os.path.join(args.dataset_path, 'annotations.csv')
    val_ds = SONYCUSTDataset(csv_path, args.dataset_path, split='validate', frontend=frontend)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, collate_fn=pad_truncate_collate)

    check_grad_flow(model)
    check_entropy(model, val_loader, device)
    visualise_attention_maps(model, val_loader, device, args.save_dir)

if __name__ == '__main__':
    main()
