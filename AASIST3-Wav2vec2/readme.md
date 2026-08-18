# AASIST3-Wav2Vec2: Self-Supervised Graph Attention with Kolmogorov-Arnold Networks

An end-to-end deepfake speech detection architecture combining pre-trained Self-Supervised Learning (SSL) representations, Kolmogorov-Arnold Networks (KAN), and Heterogeneous Spectro-Temporal Graph Attention (HS-GAL) for audio anti-spoofing under the **ASVspoof 5 Track 1** protocol. 

[AASIST3: KAN-Enhanced AASIST Speech Deepfake Detection using SSL
Features and Additional Regularization for the ASVspoof 2024 Challenge](https://arxiv.org/pdf/2408.17352)

---

## 📊 Benchmark Results (ASVspoof 5 Track 1)

All metrics are evaluated under the official Track 1 evaluation protocol across **681,872 evaluation audio files** (680,774 official protocol trials) using 4.0-second sliding-window inference with 2.0-second overlap.

| Evaluation Subset | EER (%) | minDCF ($\beta=1.90$) | Accuracy (%) | Bonafide Acc (%) | Spoof Acc (%) |
| :--- | :---: | :---: | :---: | :---: | :---: |
| **Development (Epoch 13)** | **6.79%** | — | **84.23%** | 99.46% | 79.88% |
| **Evaluation Set (Full, 681k files)** | **8.5869%** | **0.2195** | — | — | — |

- **Operating Threshold (EER)**: $\tau_{\text{EER}} = 0.025565$.

### 📈 Training Metrics (20 Epochs)

<img width="3870" height="2670" alt="training_metrics" src="https://github.com/user-attachments/assets/026c1971-fca2-47f7-9173-083e987e1f54" />


---

## Architectural Overview

```text
Raw Audio Waveform (16 kHz)
             ↓
┌─────────────────────────────────────────────────────────────┐
│ Wav2Vec 2.0 Frontend (facebook/wav2vec2-base)               │
│  • CNN Feature Extractor (Frozen)                           │
│  • Transformer Layers 0–5 (Frozen)                          │
│  • Transformer Layers 6–11 (Trainable, LR = 5e-6)           │
└─────────────────────────────────────────────────────────────┘
             ↓ Latent Frame Embeddings (768 dim)
┌─────────────────────────────────────────────────────────────┐
│ Dimension Projection & Graph Formation                      │
│  • Linear Projection (768 → 256 dim)                        │
│  • Temporal Graph ($G_t$): 25 nodes                         │
│  • Spatial Graph ($G_s$): 25 nodes                          │
└─────────────────────────────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────────────────┐
│ 4-Branch Hierarchical Heterogeneous Graph Module            │
│  Each Branch:                                               │
│   • KAN-HS-GAL (Spectral-Temporal Heterogeneous Attention)   │
│   • Adaptive Graph Pooling (Node Coarsening)                │
│   • Learnable Stack Memory Node ($S$)                        │
└─────────────────────────────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────────────────┐
│ Readout & Global Aggregation                                │
│  • Max + Mean temporal node pool                            │
│  • Max + Mean spatial node pool                             │
│  • Stack node representation concatenation                  │
└─────────────────────────────────────────────────────────────┘
             ↓
┌─────────────────────────────────────────────────────────────┐
│ KAN Classification Head                                     │
│  • B-Spline Basis Function Layer (256 → 64 → 2)             │
│  • Output: [Bonafide Logit, Spoof Logit]                    │
└─────────────────────────────────────────────────────────────┘
```

---

## Architectural Components

### 1. Fine-Tuned Self-Supervised Frontend
The 7-layer convolutional feature encoder and bottom 6 transformer layers remain frozen to preserve foundational acoustic representations, while the top 6 transformer layers (layers 6–11) are unfrozen and fine-tuned discriminatively at a low learning rate ($5 \times 10^{-6}$).

### 2. Kolmogorov-Arnold Network (KAN) Layers
Replaces standard Multi-Layer Perceptrons with learnable 1D B-spline basis function activations:
- **Grid Intervals**: `grid_size = 16`
- **Spline Order**: `spline_order = 4` (cubic B-splines)
- **Base Activation**: Dual-path PReLU + parametric spline curve

### 3. Heterogeneous Stacking Graph Attention (HS-GAL)
Constructs two parallel graph topologies from the projected feature sequence:
- **Temporal Graph ($G_t$)**: Captures time-domain boundary transitions and phase inconsistencies across 25 nodes.
- **Spatial Graph ($G_s$)**: Models cross-channel and sub-band harmonic anomalies across 25 nodes.
- **Stack Memory Node ($S$)**: A learnable global accumulator facilitating bi-directional message exchange between temporal and spatial sub-graphs.

### 4. Multi-Branch Progressive Graph Coarsening
Processes representations across 4 parallel branches with hierarchical top-k graph pooling, systematically distilling fine-grained acoustic artifacts into high-level topological embeddings before multi-scale readout aggregation.

---

## Execution & Reproduction

### Model Training
```bash
python AASIST3-Wav2vec2/AASIST3_Wav2Vec2.py \
    --batch_size 24 \
    --amp_dtype bf16 \
    --num_temporal_nodes 25 \
    --num_spatial_nodes 25 \
    --unfreeze_top_n 6 \
    --lr_backbone 1e-4 \
    --lr_frontend 5e-6 \
    --weight_decay 1e-4 \
    --patience 10 \
    --label_smoothing 0.05 \
    --augment \
    --epochs 30
```

### Sliding-Window Inference Scoring
```bash
python AASIST3-Wav2vec2/evaluate_eval_set.py \
    --eval_dir path/to/eval_flac/ \
    --checkpoint path/to/checkpoint.pth
```

### Official Track 1 Metric Computation (EER & minDCF)
```bash
python AASIST3-Wav2vec2/calculate_eval_eer.py \
    --scores path/to/eval_scores.txt \
    --protocol path/to/ASVspoof5.eval.track_1.tsv
```
