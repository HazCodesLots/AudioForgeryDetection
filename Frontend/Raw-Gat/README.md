# RawGAT-ST: Raw Waveform Graph Attention for Audio Deepfake Detection
[End-to-End Spectro-Temporal Graph Attention Networks for Speaker Verification Anti-Spoofing and Speech Deepfake Detection](https://arxiv.org/abs/2107.12710)  

A specialized deep learning architecture for audio anti-spoofing using raw waveform processing, Graph Attention Networks, and spectral-temporal feature fusion.

## Performance (ASVspoof2019 LA)

| Metric | Development | Evaluation |
|--------|------------|------------|
| **EER** | 0.73% | 2.45% |
| **min-tDCF** | - | 0.1713 |
| **Accuracy** | 99.61% | - |

### Attack-Wise Performance (Evaluation Set)

| Attack | EER (%) | min-tDCF | Description |
|--------|---------|----------|-------------|
| A09 | **0.03%** | 0.0005 | Neural waveform (best) |
| A14 | 0.22% | 0.0046 | Neural waveform |
| A08 | 0.24% | 0.0094 | Neural waveform |
| A11 | 0.45% | 0.0189 | Griffin-Lim vocoder |
| A13 | 0.57% | 0.0158 | WORLD vocoder |
| A15 | 0.61% | 0.0261 | MERLIN vocoder |
| A07 | 0.87% | 0.0258 | WORLD vocoder |
| A12 | 1.21% | 0.0311 | Neural waveform |
| A19 | 1.21% | 0.0501 | Griffin-Lim vocoder |
| A16 | 1.38% | 0.0917 | Neural waveform |
| A10 | 1.44% | 0.0674 | Griffin-Lim vocoder |
| A17 | 5.93% | 0.3949 | WaveNet vocoder |
| A18 | **7.45%** | 0.4987 | WaveNet vocoder |


## Training Progression

| Epoch | Train Acc | Val Acc | Val EER | Notes |
|-------|-----------|---------|---------|-------|
| 1 | 63.17% | 64.82% | 28.73% | Initial |
| 5 | 97.70% | 97.25% | 3.30% | Fast convergence |
| 13 | 99.27% | 99.25% | 1.25% | Plateau |
| 20 | 98.82% | 99.05% | 1.60% | **Augmentation added** |
| **27** | **99.56%** | **99.61%** | **0.73%** | **Best model** |


## Key Features

- **Raw Waveform Processing**: Learns optimal frequency representations through learnable SincConv filters
- **Dual-Stream Architecture**: Separate spectral and temporal graph attention pathways
- **Graph Attention Networks**: Models complex relationships between audio features
- **Spectral-Temporal Fusion**: Multiplicative interaction captures cross-domain spoof patterns
- **SELU Activations**: Self-normalizing properties for stable deep network training

## Core Components

### 1. SincConv Layer (Learnable Filterbank)
- **Purpose**: Raw waveform feature extraction with learnable band-pass filters
- **Architecture**:
  - 70 mel-scale filters
  - Kernel size: 129
  - Frequency range: 30 Hz - 8000 Hz
- **Advantage**: Learns optimal frequency representations for spoof detection

### 2. Residual CNN Backbone
- **Stage 1**: 2× ResBlocks (1→32→32 channels)
- **Stage 2**: 4× ResBlocks (32→64→64→64→64 channels)
- **Features**:
  - SELU activations for self-normalizing properties
  - Batch normalization
  - Max pooling for dimensionality reduction
  - Residual connections

### 3. Dual Graph Attention Processing

#### Spectral GAT
- Processes frequency-domain features
- Graph pooling ratio: 0.64
- Output: 32-dimensional spectral embeddings

#### Temporal GAT
- Processes time-domain features
- Graph pooling ratio: 0.81
- Output: 32-dimensional temporal embeddings

### 4. Spectral-Temporal Fusion
- Element-wise multiplication of spectral × temporal features
- Final GAT layer: 32 → 16 dimensions
- Graph pooling ratio: 0.64
- Produces robust combined representation

### 5. Classification Head
- Fully connected layer: 7 → 2 classes (bonafide/spoof)
- Output clamping: [-10, 10] for numerical stability

## Model Architecture

```text
Input: Raw waveform (16 kHz)
↓
SincConv (70 learnable filters)
↓
ResNet Backbone (6 blocks)
↓
├─→ Spectral GAT → Pool → [Spectral Features]
└─→ Temporal GAT → Pool → [Temporal Features]
↓
Fusion (Spectral × Temporal)
↓
Spectral-Temporal GAT
↓
Classification
```
