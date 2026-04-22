# Audio Deepfake Detection — LFCC + LCNN

A deep learning pipeline for detecting AI-generated (fake) speech using **Linear Frequency Cepstral Coefficients (LFCC)** as audio features and a **Light CNN (LCNN)** classifier with Max-Feature-Map (MFM) activations. Trained and evaluated on the **WaveFake** dataset.

## Model Architecture

| Layer Group      | Details                                          |
|------------------|--------------------------------------------------|
| Conv1            | MFMConv2d(1→48, 5×5), MaxPool 2×2               |
| Conv2a + Conv2   | MFMConv2d(48→48, 1×1), MFMConv2d(48→96, 3×3)    |
| Conv3a + Conv3   | MFMConv2d(96→96, 1×1), MFMConv2d(96→128, 3×3)   |
| Conv4a + Conv4   | MFMConv2d(128→128, 1×1), MFMConv2d(128→192, 3×3)|
| Conv5a + Conv5   | MFMConv2d(192→192, 1×1), MFMConv2d(192→256, 3×3)|
| Global AvgPool   | AdaptiveAvgPool2d(1×1)                          |
| FC1 + MFM        | Linear(256→320), MFM → 160-dim                  |
| Dropout + FC2    | Dropout(0.7), Linear(160→2)                     |

---

## Performance

### Multi-Vocoder

**Dataset:** 147,366 samples total — 80/20 split.

| Split | Total | Real | Fake |
|-------|------:|-----:|-----:|
| Train | 117,892 | 10,480 | 107,412 |
| Test  | 29,474  | 2,620  | 26,854  |

| Metric | WaveFake (Test) |
|--------|:--------------:|
| Accuracy | ~99.84% |
| AUC | ~0.9996 |
| EER | 0.3817% |

### Leave One Out Evaluation

| Fold | Held-out Vocoder           | EER (%) |
| ---- | -------------------------- | ------- |
| 1    | ljspeech_melgan            | 2.10    |
| 2    | ljspeech_melgan_large      | 0.34    |
| 3    | ljspeech_full_band_melgan  | 3.86    |
| 4    | ljspeech_multi_band_melgan | 2.71    |
| 5    | ljspeech_hifiGAN           | 6.22    |
| 6    | ljspeech_parallel_wavegan  | 3.21    |
| —    | Average (aEER)             | 3.07    |

### ASVspoof2019 Evaluation (Cross Dataset)
| Metric   | Value  |
| -------- | ------ |
| Accuracy | ~82.68% |
| AUC      | ~0.7891 |
| EER      | 24.35% |

> Partial generalizability observed on ASVspoof 2019 LA — the model performs near-perfectly in-domain but shows expected degradation on an unseen spoofing benchmark.

### In The Wild Evaluation (Cross Dataset)
| Metric   | Value  |
| -------- | ------ |
| Accuracy | ~20%   |
| AUC      | ~0.21  |
| EER      | 79.95% |

| Fold | Model                 | ITW EER (%) |
| ---- | --------------------- | ----------- |
| —    | ID Baseline           | 80.63       |
| 1    | Fold1_MelGAN          | 78.71       |
| 2    | Fold2_MelGAN_Large    | 79.96       |
| 3    | Fold3_FB_MelGAN       | 84.05       |
| 4    | Fold4_MB_MelGAN       | 77.79       |
| 5    | Fold5_HiFiGAN         | 80.04       |
| 6    | Fold6_ParallelWaveGAN | 79.16       |
| —    | Average               | 79.95       |

> Complete generalization collapse on In-The-Wild audio (79.95% EER).
---


**Default training configuration:**

| Parameter         | Value              |
|-------------------|--------------------|
| Batch size        | 64                 |
| Epochs            | 30                 |
| Learning rate     | 0.0001             |
| Weight decay      | 0.0001             |
| Scheduler         | Cosine Annealing   |
| Gradient clipping | 1.0                |
| LFCC coefficients | 60                 |
| Sample rate       | 16,000 Hz          |

---
