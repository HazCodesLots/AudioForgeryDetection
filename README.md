# Audio Forgery Detection

[![Python](https://img.shields.io/badge/Python-3.10%2B-3776AB?style=flat&logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0%2B-EE4C2C?style=flat&logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Benchmarks](https://img.shields.io/badge/Benchmarks-ASVspoof5%20%7C%20ASVspoof2019%20%7C%20WaveFake%20%7C%20ODSS-blue)](https://www.asvspoof.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)

A modular deep learning framework spanning a broad cross-section of audio anti-spoofing and synthetic speech forensics, organizing a diverse hierarchy of model architectures, feature extraction pipelines, augmentation strategies, and parameter configurations to enable extensive experimentation into how fundamentally different methodological approaches can be formulated and applied to solve the same detection task.

## 🗂️ Sub-Module Documentation


- 🔗 **[AASIST3-Wav2Vec2](AASIST3-Wav2vec2/readme.md)**  
  *Audio Anti-Spoofing with Kolmogorov-Arnold Networks (KAN) & SSL Wav2Vec 2.0 on ASVspoof5 Track 1.*

- 🔗 **[RawGAT-ST](RawGAT-ST/README.md)**  
  *End-to-end raw waveform processing via SincNet filterbanks and dual-stream spectral-temporal Graph Attention Networks on ASVspoof 2019 LA.*

- 🔗 **[LFCC-LCNN](LFCC-LCNN/README.md)**  
  *Linear Frequency Cepstral Coefficients combined with Light CNN (Max-Feature-Map activation) evaluated on the WaveFake multi-vocoder benchmark.*

- 🔗 **[ResNet-SE](ResNet/README.md)**  
  *ResNet-18 augmented with Squeeze-and-Excitation channel attention across LFCC and MFCC acoustic feature spaces.*

---
