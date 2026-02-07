## Overview
A comprehensive training and evaluation framework for audio classification, speaker verification, and spoof detection with a variety of architectural configurations. The implementation is organized to enable scalable experimentation across feature types and augmentation, maintaining a clean separation between data preprocessing, model components, and training logic.  
## 📊 RawGAT-ST Overall EER: 2.45%, min‑tDCF: 0.1713 ASVspoof2019 (Closed condition)  
## 📊 ResNet-18 Overall EER: 1.41% LFCC 80/20 split  ODSS
## 📊 ResNet-18 Overall EER: 3.15% MFCC 80/20 split  ODSS
### **Detailed architectures, explanations, and training metrics are provided in the respective sub-folder READMEs.**

Each component is isolated into a dedicated submodule for readability:  
```text
├── Classifier/          # Decision heads
├── Data_Augmentation/   # Audio transformations
├── Feature_Extraction/  # Acoustic representations
├── Frontend/            # Encoder networks
├── Pooling/             # Frame-level aggregation
```

![Python](https://img.shields.io/badge/Python-3776AB?style=for-the-badge&logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
![CUDA](https://img.shields.io/badge/CUDA-76B900?style=for-the-badge&logo=nvidia&logoColor=white)
![NumPy](https://img.shields.io/badge/NumPy-013243?style=for-the-badge&logo=numpy&logoColor=white)
![Matplotlib](https://img.shields.io/badge/Matplotlib-11557C?style=for-the-badge&logo=python&logoColor=white)
![Librosa](https://img.shields.io/badge/Librosa-FF6600?style=for-the-badge&logo=python&logoColor=white)
![Torchaudio](https://img.shields.io/badge/Torchaudio-EE4C2C?style=for-the-badge&logo=pytorch&logoColor=white)
