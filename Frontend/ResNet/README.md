<b>ResNet Reference paper<b> - https://arxiv.org/abs/2010.15006<br>
<b>AASIST Reference paper<b> - https://arxiv.org/abs/2110.01200<br>

<details open>
  <summary><b>ResNet</b></summary>
<h2>ResNet Classifier MFCC & LFCC features - ODSS Dataset</h2>

<p>
This module implements a <b>ResNet</b>-based classifier for automatic audio forgery detection using the <b>ODSS</b> (Open Deepfake Speech Synthesis) dataset. It explores the effect of different spectral features—<b>MFCC</b> (Mel-Frequency Cepstral Coefficients) and <b>LFCC</b> (Linear-Frequency Cepstral Coefficients)—to highlight feature sensitivity for modern audio deepfake detection.
</p>

<hr>

<h3>Key Features</h3>
<ul>
  <li><b>Feature Evaluation:</b> Process audio using MFCC and LFCC transforms to compare model behaviour and performance across feature types.</li>
  <li><b>Dataset:</b> Built and evaluated using the <b>ODSS</b> dataset, containing a wide array of genuine and synthetically generated/deepfake speech examples.</li>
  <li><b>Modular Notebooks:</b> Separate notebook pipelines for each configuration (<code>ResNetMFCC.ipynb</code> and <code>ResNetLFCC.ipynb</code>), enabling focused analysis and reproducible experiments.</li>
</ul>

<hr>

<h3>About the ODSS Dataset</h3>
<p>
The <b>ODSS</b> dataset (<i>Open Deepfake Speech Synthesis</i>) is a curated open-source benchmark for audio spoofing research. It contains:
</p>
<ul>
  <li><b>Real Speech:</b> Natural speech recordings from diverse speakers, environments, and sources.</li>
  <li><b>Fake Speech:</b> Synthetically generated audio using multiple state-of-the-art voice conversion, cloning, and text-to-speech models.</li>
  <li><b>Labels:</b> Each file is labelled as genuine or spoofed (fake) to enable supervised training and evaluation.</li>
</ul>

<hr>

<h3>Methodology</h3>
<ol>
  <li><b>Audio Pre-processing:</b>
    <ul>
      <li>Convert raw audio to either MFCC or LFCC spectrograms as 2D arrays.</li>
    </ul>
  </li>
  <li><b>ResNet Model:</b>
    <ul>
      <li>A convolutional ResNet used as the fraud detection backbone.</li>
      <li>Separate model checkpoints and code for MFCC and LFCC feature variants.</li>
    </ul>
  </li>
  <li><b>Training & Evaluation:</b>
    <ul>
      <li>Train separately on each feature type using the ODSS dataset.</li>
      <li>Compare testing accuracy, ROC-AUC, and other metrics to illustrate how spectral parametrization affects performance on deepfake detection tasks.</li>
    </ul>
  </li>
</ol>

<hr>

<h3>References</h3>
<ul>
  <li>
    <b>ODSS Dataset:</b> <i>https://zenodo.org/records/8370669</i>
  </li>
</ul>
</details>

