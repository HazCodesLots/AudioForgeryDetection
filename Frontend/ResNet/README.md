# ResNetSE: ResNet with Squeeze-and-Excitation for Audio Anti-Spoofing
[Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)  
[Squeeze-and-Excitation Networks](https://arxiv.org/abs/1709.01507)  
A deep learning architecture for audio deepfake detection using ResNet18 with Squeeze-and-Excitation blocks, temporal/spectral attention, and LFCC features enhanced with phase derivatives.

## Key Features

- **LFCC Feature Extraction**: Linear Frequency Cepstral Coefficients optimized for spoofing detection
- **Phase Derivative Features**: Additional channel capturing temporal artifacts for WaveNet attacks (A17/A18)
- **SE-ResNet Architecture**: ResNet18 backbone with Squeeze-and-Excitation blocks for channel-wise attention
- **Dual Attention Mechanisms**: Temporal and spectral attention layers for artifact detection
- **Advanced Training**: Focal loss, class weighting, SpecAugment, and label smoothing

## Architecture Overview

### Feature Extraction

**LFCC (4-Channel Input)**:
1. **Static LFCC**: Linear-scale filterbank energies (70 coefficients)
2. **Delta LFCC**: First-order temporal derivatives
3. **Delta-Delta LFCC**: Second-order temporal derivatives  
4. **Phase Derivatives**: Group delay features from STFT phase

**Parameters**:
- Sample rate: 16 kHz
- FFT size: 512
- Window: 25ms (Hamming)
- Hop: 10ms
- Filters: 70 linear-scale
- Pre-emphasis: 0.97

### Model Architecture

```
Input: [B, 4, 400, 70] (4-channel LFCC + Phase)
↓
Conv1 (7×7, stride=2) → BN → ReLU
↓
Temporal Attention (1D conv, kernel=15)
↓
Spectral Attention (cross-frequency modeling)
↓
MaxPool (3×3)
↓
ResNet Layers with SE Blocks:
Layer1: 2 × SEBasicBlock (64 channels)
Layer2: 2 × SEBasicBlock (128 channels)
Layer3: 2 × SEBasicBlock (256 channels)
Layer4: 2 × SEBasicBlock (512 channels)
↓
Global Average Pooling
↓
Dropout (0.2)
↓
FC (512 → 2 classes)
```

### Core Components

#### 1. SE Block (Squeeze-and-Excitation)
- **Purpose**: Channel-wise attention for feature recalibration
- **Architecture**:
  - Global average pooling
  - FC: channels → channels/16 → channels
  - Sigmoid activation
  - Channel-wise multiplication

#### 2. Temporal Attention
- **Purpose**: Detect temporal discontinuities (clicks, edits)
- **Architecture**:
  - 1D convolution along time axis (kernel=15)
  - Two-layer bottleneck (reduction=4)
  - Sigmoid gating

#### 3. Spectral Attention
- **Purpose**: Emphasize discriminative frequency bands
- **Architecture**:
  - Time-averaged pooling
  - 1×1 convolutions (channels → channels/4 → channels)
  - Sigmoid gating

## Model Parameters

```python
Total parameters: ~11M
Input shape: (batch_size, 4, 400, 70)  # [channels, time, freq]
Output shape: (batch_size, 2)           # bonafide/spoof logits
Sequence length: 400 frames (~4 seconds)
```

## Performance (ASVspoof2019 LA)

| Metric | Development | Evaluation |
|--------|------------|------------|
| **EER** | 0.60% | 7.86% |
| **min-tDCF** | - | 0.7503 |

### Attack-Wise Performance (Evaluation Set)

| Attack | EER (%) | Description | Notes |
|--------|---------|-------------|-------|
| A07 | **0.00%** | WORLD vocoder | Perfect detection |
| A09 | 0.08% | Neural waveform | Excellent |
| A16 | 0.08% | Neural waveform | Excellent |
| A11 | 0.22% | Griffin-Lim vocoder | Excellent |
| A10 | 0.45% | Griffin-Lim vocoder | Very good |
| A19 | 0.59% | Griffin-Lim vocoder | Very good |
| A14 | 0.67% | Neural waveform | Very good |
| A15 | 0.83% | MERLIN vocoder | Very good |
| A13 | 1.49% | WORLD vocoder | Good |
| A12 | 1.67% | Neural waveform | Good |
| A08 | 3.66% | Neural waveform | Moderate |
| A18 | **9.91%** | WaveGlow vocoder | **Challenging** |
| A17 | **40.31%** | WaveNet vocoder | **Very BAD** |

### Performance Analysis

**Strengths**:
- Excellent performance on traditional vocoders (WORLD, Griffin-Lim, MERLIN): EER < 1.5%
- Strong detection of most neural waveform attacks: EER < 4%
- Perfect detection on A07 (WORLD vocoder)

**Weaknesses**:
- **A17 (WaveNet)**: 40.31% EER - temporal click artifacts not fully captured
- **A18 (WaveGlow)**: 9.91% EER - phase discontinuities challenging
- Phase derivative features help but insufficient for advanced neural vocoders

**Dev-Eval Gap**: 0.60% → 7.86% EER indicates some overfitting to development set, primarily due to A17/A18 attacks not present in training distribution.

## Known Limitations

1. **WaveNet Attacks (A17)**: Current phase derivative features insufficient for detecting subtle temporal artifacts
2. **Generalization Gap**: Model optimized for development set attacks may not transfer to unseen vocoder architectures
3. **min-tDCF**: 0.7503 is above ASVspoof2019 baseline, suggesting room for improvement in tandem decision cost
