# ConvNeXt-SelfAttention: Multi-Scale Audio Classification Architecture

A general-purpose deep learning architecture combining ConvNeXt backbone, multi-head self-attention pooling, and Graph Guided Convolutional Networks (GGCN) for complex audio classification tasks.

## Use Cases

This model excels at audio tasks requiring rich feature extraction and long-range dependency modeling:

- **Music Classification**: Genre recognition, instrument identification, mood detection
- **Speech Emotion Recognition**: Emotion detection from prosodic and spectral features
- **Environmental Sound Classification**: Multi-source acoustic scene analysis
- **Audio Event Detection**: Multi-label tagging of simultaneous audio events
- **Speaker Identification**: Multi-class speaker recognition tasks

## Key Features

- **Multi-Scale Feature Extraction**: ConvNeXt captures local patterns and textures from mel-spectrograms
- **Temporal Modeling**: Multi-head self-attention handles long-range dependencies across time
- **Feature Relationship Learning**: Graph neural networks model co-occurrence patterns and feature interactions

## Core Components

### 1. ConvNeXt1D Backbone
- **Purpose**: Feature extraction from mel-spectrograms
- **Architecture**: 
  - Depths: [3, 3, 9, 3] blocks per stage
  - Dimensions: [96, 192, 384, 768] channels
  - Depthwise convolutions with kernel size 9
  - Squeeze-and-Excitation (SE) blocks for channel attention
  - Stochastic depth (DropPath) for regularization
  - Layer scaling for stable training

### 2. Self-Attentive Pooling
- **Purpose**: Temporal aggregation with attention mechanism
- **Features**:
  - Multi-head self-attention (8 heads)
  - Attention dimension: 512
  - Residual connections
  - Layer normalization
  - Dropout for regularization

### 3. Graph Guided ConvNet (GGCN)
- **Purpose**: Feature relational modeling
- **Architecture**:
  - k-NN graph construction (k=5)
  - 2-layer Graph Convolutional Network
  - Hidden dimension: 256
  - Residual connections
  - Layer normalization and GELU activations

### 4. MLP Classifier
- **Purpose**: Multi-class classification head
- **Architecture**:
  - Hidden layers: [512, 256, 128]
  - ReLU activations
  - Dropout: 0.3
  - Configurable output classes

## Model Parameters

```python
Total parameters: ~30M (estimated)
Input shape: (batch_size, 1, 64, 250)  # mel-spectrogram
Output shape: (batch_size, num_classes)  # logits
```
## Training Metrics

<img width="1693" height="690" alt="image" src="https://github.com/user-attachments/assets/2a7e54dc-4440-42d2-88d8-f67ea0c86539" />

## Custom Evaluation Performance 

<img width="691" height="731" alt="image" src="https://github.com/user-attachments/assets/d48cfefa-da92-40fa-931b-addbf5e5a744" />
<img width="738" height="523" alt="image" src="https://github.com/user-attachments/assets/2a523c73-40b8-49ea-a24b-6c8bce707e67" />
