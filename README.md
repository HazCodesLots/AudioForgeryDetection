<b>ResNet Reference paper<b> - https://arxiv.org/abs/2010.15006<br>
<b>AASIST Reference paper<b> - https://arxiv.org/abs/2110.01200<br>

<details>
  <summary><b>AASIST</b></summary>
<h2>AASIST: Automated Anti-Spoofing for ASVspoof2021 LA Eval</h2>

<p>
<b>AASIST</b> deep learning model for detecting spoofed (synthetic or manipulated) speech in the newer <b>ASVspoof2021 Logical Access (LA) Evaluation</b> dataset. This pipeline batch scores audio files against a state-of-the-art anti-spoofing model.
</p>

<hr>

<h3>Features</h3>
<ul>
  <li><b>Model:</b> Loads and runs the AASIST neural network for spoof detection.</li>
  <li><b>Dataset Support:</b> Designed for the ASVspoof2021 LA Eval protocol with .flac audio files.</li>
  <li><b>Batch Processing:</b> Scores large numbers of utterances (up to 180K+) efficiently using DataLoader</li>
  <li><b>Results Output:</b> Exports per-utterance scores as JSON for further evaluation.</li>
</ul>

<hr>

<h3>Architecture</h3>
<ol>
  <li><b>Model Initialization:</b>
    <ul>
      <li>Loads <code>aasist/config/AASIST.conf</code> for model configuration and checkpoint path.</li>
      <li>Initializes the model on GPU if available.</li>
    </ul>
  </li>
  <li><b>Dataset Parsing:</b>
    <ul>
      <li>Reads protocol file for target utterance IDs.</li>
      <li>Locates and loads each corresponding <code>.flac</code> file, converting to 16kHz waveform.</li>
      <li>Applies zero-padding or trimming to each waveform for uniform input size (64,600 samples).</li>
    </ul>
  </li>
  <li><b>Scoring Loop:</b>
    <ul>
      <li>Iterates over the dataset with batch size 1 for maximum compatibility.</li>
      <li>For each file, runs inference and records the mean output score for the utterance ID.</li>
      <li>Missing files and load errors are handled with error logging.</li>
    </ul>
  </li>
  <li><b>Results:</b>
    <ul>
      <li>All scores are saved to <code>aasist_eval_results.json</code> in the format <code>{"utt_id": score, ...}</code>.</li>
      <li>Progress, missing files, and completion summary are printed to console.</li>
    </ul>
  </li>
</ol>

<h3>Usage Tips</h3>
<ul>
  <li>Requires <code>PyTorch</code>, <code>librosa</code>, <code>tqdm</code>, and all model code dependencies (see requirements.txt).</li>
  <li>Ensure <code>AASIST.conf</code> and checkpoint are properly configured and accessible.</li>
  <li>Point <code>protocol_path</code> and <code>flac_dir</code> to your local dataset.</li>
  <li>Progress bar enables transparent monitoring for large-scale evaluation.</li>
</ul>

<hr>

  </li>
  <li>
    <b>ASVspoof2021 Data & Protocols:</b>
    (<a href="https://www.asvspoof.org/" target="_blank">www.asvspoof.org</a>)
  </li>
  <li>
    <b>Official Implementation:</b>
    (<a href="https://github.com/clovaai/aasist" target="_blank">GitHub: clovaai/aasist</a>)
  </li>
</ul>

<hr>
