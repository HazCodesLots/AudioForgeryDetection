This Repository contains the implementation of Replay and Synthetic Speech Detection with Res2Net Architecture (https://arxiv.org/abs/2010.15006)<br>
Dataset utilized for training the model was a subset of 2000 files from the HAD dataset (https://arxiv.org/abs/2104.03617)<br>
I have studied the papers and their implementations mentioned below and have highlighted their merits and demerits to the best of my understanding.<br>

# Res2Net50
**The research paper Replay and Synthetic Speech Detection with Res2Net Architecture was more accessible to me than others due to my background in neural networks. Many of its concepts were easier for me to grasp, making the implementation more , well easier. Additionally, I incorporated training methods from previous models I’ve worked on, as they align with my understanding of visual pattern recognition and computer vision through deep learning.** <br>
# Technical Explaination of the model
This Res2net50 model processes audio files into mel-spectrograms, analyzes the patterns to classify if the audio is AI generated or genuine one from a human being. ResNet stands for residual networks designed to train and address neural networks. This is done through adding additional layers for different tasks. The resultant is a diverse net that can handle more complex operations. Res2Net50 exlusively performs well with its mutli-layer feature extraction making it effective in judging intricate patterns.<br> 
Due to hardware and time limitations, I cropped the dataset from 53,000 audio files to 2000 audio files for training. I used Adam optimizer for Optimization, cross entropy for perforamce as I have used these utilities to train images before. I tested the model on 5-6 randomly picked audio files and it performed well in my opinion. This project is based on Pytorch.<br>
I am not sure on how we can extend this model on a larger scale as I have never worked with bigger machines than my laptop so my knowledge is limited in this field. I feel like combining certain parts of other models can greatly benefit the Res2Net50. The paper suggests utilization of CQT to get resolution results at lower frequencies. I also believe combining raw-waveform input can speed up this process as mentioned in the AASIST paper but I am not sure how it can be integrated in this model.

# Challenges Encountered  

<details open>
  <summary><strong>Expand/Collapse</strong></summary>

##  Hardware Limitations Leading to Default Kernel Failure  
- Switched to ipykernel, as it is recommended for PyTorch.  
- Used a fraction of the dataset (1000 real audio files, 1000 fake audio files) instead of the full dataset.  
- Installed the CUDA Toolkit to enable GPU training on my NVIDIA GTX 1650m.  

##  Dependencies Installation  
- Due to the kernel switch, all dependencies had to be reinstalled (not sure why):  
  - PyTorch, Soundfile, Librosa, Torchvision, Torchaudio  

##  Unavailability of Code  
- Used ChatGPT, Perplexity, and Claude to debug and generate code snippets.  

</details>

# Paper and Approach

<details open>
  <summary><strong>Expand/Collapse</strong></summary>

# Replay and Synthetic Speech Detection with Res2Net Architecture  

**Xu Li, Na Li, Chao Weng, Xunying Liu, Dan Su, Dong Yu**  

## Understanding ResNet and Res2Net in Audio Detection  

ResNet stands for Residual Networks. I have worked with Res2Net in the computer vision domain, so I understand how they work. In the audio domain, Res2Net is combined with Neural Networks to detect specific patterns in spectrograms.  

Res2Net50 can also incorporate Squeeze-and-Excite (SE) blocks and Constant-Q Transform (CQT) for promising results in audio spoofing detection.  

---

## Key Takeaways  

- ResNet helps analyze audio spectrums while solving the vanishing gradient problem.  
- Constant-Q Transform (CQT) provides high resolution at low frequencies, making it useful for audio analysis.  
- Res2Net50 has demonstrated strong performance against Replay and Synthetic Speech attacks.  

![Res2Net Performance](https://github.com/user-attachments/assets/0de214ac-69dc-435b-aeb6-fec2eb6209df)  

---

## Challenges  

- Hardware requirements are relatively high and may require optimization.  
- Performance against adversarial attacks is not as strong compared to other metrics.  
- Dataset diversity is crucial—limited diversity can restrict model effectiveness.  

---
# AASIST: Audio Anti-Spoofing Using Integrated Spectro-Temporal Graph Attention Networks  

**Jee-weon Jung¹, Hee-Soo Heo¹, Hemlata Tak², Hye-jin Shim³, Joon Son Chung¹, Bong-Jin Lee¹, Ha-Jin Yu³, Nicholas Evans²**  

AASIST is based on the premise that bona-fide and spoofed utterances can be differentiated by analyzing Spectro-Temporal patterns. This system utilizes a RawNet2-based encoder for feature extraction and classification.  

---

## Key Takeaways  

- **AASIST** enables differentiation of subtle artifacts introduced to data through AI.  
- By leveraging Graph Attention Networks (GATs), AASIST can be used for both speaker verification and spoofing detection.  
- For limited environments, a lightweight version called AASIST-L is available.  

![AASIST Architecture](https://github.com/user-attachments/assets/04ee309f-2dce-45d8-a443-9bff572b3a28)  

---

## Challenges  

- High hardware requirements during training.  
- Susceptible to overtraining, leading to potential bias towards the training dataset.  

---
# Automatic Speaker Verification Spoofing and Deepfake Detection Using wav2vec 2.0 and Data Augmentation  

**Hemlata Tak, Massimiliano Todisco, Xin Wang, Jee-weon Jung, Junichi Yamagishi, Nicholas Evans**  

This approach is based on self-supervised learning through wav2vec 2.0. Trained only on bona-fide data, it has achieved some of the lowest error rates in the ASVspoof2021 Logical Access and Deepfake databases.  

## Takeaways  

- The model is strictly trained on genuine data.  
- Data augmentation is used to reduce overfitting and improve performance.  
- Wav2vec 2.0 is integrated by directly feeding raw waveforms into the system.  

![Image](https://github.com/user-attachments/assets/53f0bb58-56ef-4058-9b97-ac8e1cd4b33a) 

## Challenges  

- Since the training data is strictly genuine, the model may struggle with anomalies.  
- Models trained on certain datasets may perform well in controlled scenarios but struggle in real-world applications.  
</details>
