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
| Global AvgPool   | AdaptiveAvgPool2d(1×1)                           |
| FC1 + MFM        | Linear(256→320), MFM → 160-dim                  |
| Dropout + FC2    | Dropout(0.7), Linear(160→2)                     |

---

## 📊 Performance

### Multi-Vocoder

**Dataset:** 147,366 samples total — 80/20 split, evaluated at epoch 75 (best).

| Split | Total | Real | Fake |
|-------|------:|-----:|-----:|
| Train | 117,892 | 10,480 | 107,412 |
| Test  | 29,474  | 2,620  | 26,854  |

| Metric | WaveFake (Test) | ASVspoof 2019 LA (Cross-Dataset) |
|--------|:--------------:|:--------------------------------:|
| Accuracy | 99.84% | 82.68% |
| AUC | 0.9996 | 0.7891 |
| EER | 0.3817% | 24.3452% |

> Partial generalizability observed on ASVspoof 2019 LA — the model performs near-perfectly in-domain but shows expected degradation on an unseen spoofing benchmark.

### LOVO - Leave One Vocoder Out (WaveFake)

Each model is trained on 9 vocoders and tested on the held-out one. **Average EER: 0.51%**

| Fold | Held-out Vocoder | EER (%) | Accuracy (%) | AUC |
|------|-----------------|---------|--------------|-----|
| 1 | common_voices_conformer_fastspeech2_pwg | 0.0798 | 99.95 | 1.0000 |
| 2 | jsut_multi_band_melgan | 0.0000 | 99.87 | 1.0000 |
| 3 | jsut_parallel_wavegan | 0.7400 | 98.36 | 0.9988 |
| 4 | ljspeech_full_band_melgan | 0.3664 | 98.37 | 0.9997 |
| 5 | ljspeech_hifiGAN | 1.2177 | 88.25 | 0.9965 |
| 6 | ljspeech_melgan | 0.6584 | 96.78 | 0.9991 |
| 7 | ljspeech_melgan_large | 0.0229 | 99.89 | 1.0000 |
| 8 | ljspeech_multi_band_melgan | 0.9389 | 97.06 | 0.9993 |
| 9 | ljspeech_parallel_wavegan | 0.5420 | 98.39 | 0.9997 |
| 10 | ljspeech_waveglow | 1.3353 | 89.61 | 0.9975 |
| — | **Average (aEER)** | **0.51** | — | — |

> The model generalizes well to unseen vocoders within WaveFake, achieving near-perfect AUC across all folds. The hardest vocoders to detect are `ljspeech_waveglow` and `ljspeech_hifiGAN`.

---

### Cross-Dataset — ASVspoof 2019 LA

Each of the 10 LOVO models is evaluated on the ASVspoof 2019 LA dev set without any retraining. **Average cross-dataset EER: 24.91%**

| Model (Held-out Vocoder) | EER (%) | Accuracy (%) |
|--------------------------|---------|--------------|
| common_voices_conformer_fastspeech2_pwg | 25.12 | 76.47 |
| jsut_multi_band_melgan | 25.69 | 82.17 |
| jsut_parallel_wavegan | 22.39 | 79.12 |
| ljspeech_full_band_melgan | 24.69 | 81.13 |
| ljspeech_hifiGAN | 21.62 | 82.56 |
| ljspeech_melgan | 23.73 | 81.03 |
| ljspeech_melgan_large | 25.39 | 79.37 |
| ljspeech_multi_band_melgan | 25.00 | 81.64 |
| ljspeech_parallel_wavegan | 28.51 | 78.70 |
| ljspeech_waveglow | 26.97 | 80.23 |
| **Average** | **24.91** | — |

> The significant EER increase from ~0.5% (WaveFake) to ~25% (ASVspoof 2019) confirms that the model learns WaveFake/LJSpeech-specific patterns and does not fully generalize to unseen spoofing methods and recording conditions.


**Default training configuration:**

| Parameter         | Value              |
|-------------------|--------------------|
| Batch size        | 64                 |
| Epochs            | 100                |
| Learning rate     | 0.0001             |
| Weight decay      | 0.0001             |
| Scheduler         | Cosine Annealing   |
| Gradient clipping | 1.0                |
| LFCC coefficients | 60                 |
| Sample rate       | 16,000 Hz          |

---


## Key Components

### `LCNN.py`
Defines `MaxFeatureMap2D`, `MFMConv2d`, and the full `LCNN` model.

### `WaveFakeLoader.py`
Defines `WaveFakeDatasetFixed` (a PyTorch `Dataset`) and `create_loaders_from_splits()` — a factory function that returns train and test `DataLoader`s, each with inline LFCC extraction.

### `feature_extraction.py`
Defines `LFCCExtractor`, `nn.Module` that computes LFCC features on-the-fly directly on the GPU. It uses `torch.stft` with a Hamming window, applies a linearly-spaced triangular filterbank (60 filters), log-compresses the output, then applies an orthonormal DCT to extract the first 60 cepstral coefficients. Input: `(batch, samples)` → Output: `(batch, 60, time_frames)`.

---

### `train_wavefake_lovo.py`
Implements **Leave-One-Vocoder-Out (LOVO)** training. Iterates over all 10 vocoders, trains a fresh LCNN on the remaining 9, and evaluates on the fully held-out vocoder (all samples, no train split). Supports `--start` and `--end` CLI arguments for running specific fold ranges in parallel or in batches. Results are saved cumulatively to `weights/protocol/lovo_results.json`, and each fold's best checkpoint is stored at `weights/protocol/lovo_<vocoder>/lovo_<vocoder>_best.pt`.

---

### `evaluate_wavefake_all.py`
Evaluates a pre-trained model on the WaveFake test set across **all vocoders** (Scenario C). Loads a checkpoint (defaults to `weights/epoch_075.pt`), verifies the 80/20 data split and class balance, then reports Accuracy, AUC, and EER on the held-out test set.

---

### `evaluate_wavefake_lovo.py`
Verification script for the LOVO experiment. Loads each of the 10 `lovo_<vocoder>_best.pt` checkpoints and evaluates them on their respective held-out vocoder test set. Prints a per-fold breakdown and computes the **average EER (aEER)** across all folds. Final results are saved to `weights/protocol/wavefake_lovo_verified_results.json`.

---

### `evaluateASVSpoof2019.py`
Cross-dataset evaluation script. Loads a single WaveFake-trained model and runs it on the **ASVspoof 2019 LA** dev set using its official protocol file (`.txt`). Parses `bonafide`/`spoof` labels from the protocol, runs inference on `.flac` audio files, and reports Accuracy, AUC, and EER. Includes automatic generalizability interpretation — flags whether the model transfers well, partially, or poorly to the new domain.

---

### `evaluateASVspoof2019Lovo.py`
Cross-dataset evaluation using **all 10 LOVO models**. Reads `lovo_results.json` to determine fold order, then runs each LOVO checkpoint against the full ASVspoof 2019 LA dev set. Reports per-fold EER and accuracy, computes the average cross-dataset EER, and saves results to `weights/protocol/asvspoof_cross_lovo_results.json`.

### `train.py`
Defines `LFCCLCNNTrainer` with:
- `train_epoch()` — one full training pass
- `validate()` — evaluation with AUC & EER
- `calculate_eer()` — computes EER via Brent's method on the ROC curve
- `save_checkpoint()` / `save_metrics()` — persistence utilities
- `train()` — full training loop with configurable scheduler
---

