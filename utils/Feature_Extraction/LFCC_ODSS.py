import os
import numpy as np
import soundfile as sf
import scipy.signal
import scipy.fftpack
from pathlib import Path

def extract_lfcc(signal, samplerate=16000, n_fft=512, n_filters=20, n_ceps=20, winlen=0.025, winstep=0.01):
    signal = scipy.signal.lfilter([1, -0.97], 1, signal)

    frame_len = int(winlen * samplerate)
    frame_step = int(winstep * samplerate)
    signal_length = len(signal)
    num_frames = 1 + int((signal_length - frame_len) / frame_step)

    frames = np.zeros((num_frames, frame_len))
    for i in range(num_frames):
        start = i * frame_step
        frames[i] = signal[start:start + frame_len] * np.hamming(frame_len)

    mag_frames = np.absolute(np.fft.rfft(frames, n=n_fft))
    pow_frames = ((1.0 / n_fft) * (mag_frames ** 2))

    low_freq = 0
    high_freq = samplerate / 2
    hz_points = np.linspace(low_freq, high_freq, n_filters + 2)
    bins = np.floor((n_fft + 1) * hz_points / samplerate).astype(int)

    filterbank = np.zeros((n_filters, int(n_fft / 2 + 1)))
    for m in range(1, n_filters + 1):
        f_m_minus, f_m, f_m_plus = bins[m - 1], bins[m], bins[m + 1]
        for k in range(f_m_minus, f_m):
            filterbank[m - 1, k] = (k - f_m_minus) / (f_m - f_m_minus)
        for k in range(f_m, f_m_plus):
            filterbank[m - 1, k] = (f_m_plus - k) / (f_m_plus - f_m)

    feat = np.dot(pow_frames, filterbank.T)
    feat = np.where(feat == 0, np.finfo(float).eps, feat)
    log_feat = np.log(feat)

    lfcc = scipy.fftpack.dct(log_feat, type=2, axis=1, norm='ortho')[:, :n_ceps]
    return lfcc

def process_odss_odss_style(dataset_path, output_path, sr=16000):
    dataset_path = Path(dataset_path)
    output_path = Path(output_path)
    output_path.mkdir(exist_ok=True)

    class_map = {
        "natural": "bonafide",
        "fastpitch-hifigan": "spoof",
        "vits": "spoof"
    }

    total = 0
    skipped = 0

    for subfolder, label in class_map.items():
        audio_dir = dataset_path / subfolder
        out_dir_base = output_path / label / subfolder

        for wav_file in audio_dir.rglob("*.wav"):
            try:
                signal, rate = sf.read(str(wav_file))
                if rate != sr:
                    print(f"[!] Skipping {wav_file.name}: Sample rate {rate} ≠ {sr}")
                    skipped += 1
                    continue

                lfcc_feat = extract_lfcc(signal, samplerate=sr)
                rel_path = wav_file.relative_to(audio_dir).with_suffix(".npy")
                out_file = out_dir_base / rel_path
                out_file.parent.mkdir(parents=True, exist_ok=True)

                np.save(out_file, lfcc_feat)
                total += 1
                print(f"[✓] Saved LFCC: {out_file}")
            except Exception as e:
                print(f"[X] Failed {wav_file.name}: {e}")
                skipped += 1

process_odss_odss_style(
    dataset_path=r"C:\Users\DaysPC\Documents\Datasets\odss",
    output_path=r"C:\Users\DaysPC\Documents\Datasets\odss_lfcc"
)