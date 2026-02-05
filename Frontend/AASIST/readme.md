AASIST and what I understand from reading the research paper  

# AASIST: Audio Anti-Spoofing using Integrated Spectro-Temporal Graph Attention Networks

Spoofing artifacts that distinguish fake audio from genuine speech can appear in different forms:

- Spectral artifacts: Anomalies in specific frequency bands or sub-bands
- Temporal artifacts: Irregularities in time-domain patterns and transitions
- Hybrid artifacts: Complex patterns spanning both domains

AASIST learns these patterns through integrated spectro-temporal modeling.

```text
Input: Raw Waveform
         ↓
    RawNet2 Encoder
         ↓
   Max Pooling Split
         ↓
    ┌────┴────┐
    ↓         ↓
Temporal   Spectral
 Graph      Graph
   (Gt)      (Gs)
    └────┬────┘
         ↓
    HS-GAL Layer
    (Heterogeneous
     Attention +
     Stack Node)
         ↓
  Max Graph Op (MGO)
         ↓
    Readout Layer
    [Max|Mean|Stack]
         ↓
   Classification
```

<details>
<summary>AASIST Architecture</summary>

  ## Architecture Overview

#### RawNet2-based encoder

The RawNet2-based encoder extracts high level representations with dimensions C (channels), S (spectral bins), and T (temporal sequence length) directly from raw waveforms. However, the RawNet2 used in AASIST is modified to output F ∈ ℝ^{C×S×T} which is reorganized via Max pooling into graph node features, with separate temporal and spectral node representations.  

#### Heterogeneous Stacking Graph Attention Layer (HS-GAL)

The HS-GAL (Heterogeneous Stacking of Graph Attention Layers) seems to be the core component of this architecture, consisting of:

##### Heterogeneous Attention
- Temporal Nodes (Gt): Capture time-domain features and temporal patterns
- Spectral Nodes (Gs): Model frequency-domain characteristics and spectral anomalies

##### Stack Node
- Aggregates information from both temporal and spectral domains

#### Max Graph Operation

The Max Graph Operation (MGO) introduces a selection mechanism that aggregates node features by taking element-wise maxima across graph representations, allowing AASIST to emphasize the most discriminative spoofing-related activations while suppressing less informative responses in both temporal and spectral domains.

#### Readout

An extended aggregation that concatenates max, mean, and stack node graph level summaries.  

### Benchmarks (from the paper)

<img width="335" height="88" alt="image" src="https://github.com/user-attachments/assets/3af4498a-6691-4842-801f-03b77db290fe" />

| # Parameters | Front-end     | Architecture | min t-DCF | EER (%) |
|-------------|---------------|-------------|-----------|---------|
| 297K        | Raw waveform  | AASIST      | 0.0275    | 0.83    |
| 85K         | Raw waveform  | AASIST-L    | 0.0309    | 0.99    |
</details>

 # AASIST3: Graph Attention Anti-Spoofing with Kolmogorov-Arnold Networks
A state-of-the-art audio deepfake detection model combining graph neural networks, heterogeneous attention mechanisms, and Kolmogorov-Arnold Network (KAN) layers for robust spoofing detection.

Overview
AASIST3 (Anti-Spoofing using Attention and Stack with KAN Integration) extends graph-based audio forensics by replacing traditional MLPs with KAN layers that use learnable B-spline basis functions for more expressive non-linear transformations.

KAN-based Graph Attention (KAN-GAL): Replaces standard linear projections with adaptive spline-based transformations for improved feature learning.​

Heterogeneous Temporal-Spatial Graphs: Constructs dual graph representations capturing both temporal patterns and spectral relationships.​

Multi-Branch Hierarchical Pooling: Progressive graph coarsening across 4 branches with learnable stack aggregation.​

Gradient-Aware Training: Real-time gradient health monitoring with automatic overfitting detection during training.

```text
Input (Mel Spectrogram 128×T)
    ↓
┌─────────────────────────────────┐
│ AASIST3 Encoder (6 ResBlocks)   │  → Feature maps (256×T')
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Graph Formation Module          │
│  • Temporal Graph (KAN-GAL)     │  → h_t (100 nodes × 64 dim)
│  • Spatial Graph (KAN-GAL)      │  → h_s (100 nodes × 64 dim)
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Multi-Branch Architecture (×4)  │
│  Each Branch:                   │
│   • KAN_HS_GAL (Hetero Attn)    │
│   • KAN_GraphPool               │
│   • Stack Node Fusion           │
└─────────────────────────────────┘
    ↓
┌─────────────────────────────────┐
│ Global Aggregation              │
│  • Max/Mean pooling per branch  │
│  • Stack aggregation            │
└─────────────────────────────────┘
    ↓
Output Head (KAN-based) → Logits (2 classes)
```

<details>
<summary>AASIST3 Architecture</summary>

  ## Architecture Overview

  
#### KANLayer
Implements Kolmogorov-Arnold representation using vectorized B-spline basis computation with dual-path activation (PReLU + spline).​
Parameters:
grid_size=16: Number of spline intervals
spline_order=4: Polynomial degree
grid_range=(-1,1): Input normalization range

#### AASIST3 Encoder
6-block ResNet-style encoder with:
BatchNorm → SELU → Conv1D architecture
Stride-3 downsampling at blocks 2, 4, 6
Output: 256-channel feature maps​

#### Graph Formation
Converts encoder features into dual graphs:  
- Temporal graph: Captures time-series dependencies (adaptive max pooling → 100 nodes)
- Spatial graph: Models frequency bin relationships (channel-wise pooling → 100 nodes)
Each graph processed by KAN-GAL + KAN-GraphPool
​
#### KAN_HS_GAL (Heterogeneous Stack Graph Attention)
Fuses temporal and spatial graphs using:  
- Primary attention: Node-to-node message passing with heterogeneous edge weights
- Stack attention: Learnable memory node aggregating cross-branch information
​

#### Multi-Branch Architecture
4 cascaded branches with progressive pooling (ratio=0.5):

- Branch 1: 100 → 50 nodes

- Branch 2: 50 → 25 nodes

- Branch 3: 25 → 12 nodes

- Branch 4: 12 → 6 nodes

Final embedding: [H_max_t, H_mean_t, H_max_s, H_mean_s, S_max] → 256 dim

</details>​
