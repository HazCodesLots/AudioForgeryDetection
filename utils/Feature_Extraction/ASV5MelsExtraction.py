import torch
print(torch.__version__)
print(torch.cuda.is_available())


import os
import tarfile
import librosa
import numpy as np
import torch
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import soundfile as sf
from typing import Tuple, Optional

class MelSpectrogramExtractor:
    """Extract and cache mel-spectrograms for ASVspoof5 training."""

    def __init__(self, 
                 sample_rate: int = 16000,
                 n_fft: int = 1024,
                 hop_length: int = 512,
                 n_mels: int = 128,
                 f_min: float = 50,
                 f_max: float = 7500,
                 power: float = 2.0,
                 cache_dir: str = "mel_cache"):

        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.hop_length = hop_length
        self.n_mels = n_mels
        self.f_min = f_min
        self.f_max = f_max
        self.power = power
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(exist_ok=True)

    def extract_audio(self, audio_path: str) -> Optional[np.ndarray]:
        """Load and resample audio to target sample rate."""
        try:
            audio, sr = librosa.load(audio_path, sr=self.sample_rate, mono=True)
            if len(audio) == 0:
                print(f"Warning: Empty audio file {audio_path}")
                return None
            return audio
        except Exception as e:
            print(f"Error loading {audio_path}: {e}")
            return None

    def compute_mel_spectrogram(self, audio: np.ndarray) -> np.ndarray:

        mel_spec = librosa.feature.melspectrogram(
            y=audio,
            sr=self.sample_rate,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            n_mels=self.n_mels,
            fmin=self.f_min,
            fmax=self.f_max,
            power=self.power,
            norm="slaney",
            htk=True
        )

        mel_spec_db = librosa.power_to_db(mel_spec, ref=np.max)

        return mel_spec_db.astype(np.float32)

    def process_single_file(self, 
                           audio_path: str, 
                           output_file: Optional[str] = None) -> Optional[np.ndarray]:
        """Process single audio file and optionally cache."""
        audio = self.extract_audio(audio_path)
        if audio is None:
            return None

        mel_spec = self.compute_mel_spectrogram(audio)

        if output_file:
            np.save(output_file, mel_spec)

        return mel_spec

    def extract_tar_archive(self, tar_path: str, extract_dir: str) -> None:
        """Extract tar.gz archive to directory."""
        try:
            with tarfile.open(tar_path, 'r:*') as tar:
                tar.extractall(path=extract_dir)
            print(f"✓ Extracted {tar_path}")
        except Exception as e:
            print(f"✗ Error extracting {tar_path}: {e}")


class ASVSpoof5DatasetPreprocessor:
    """Complete preprocessing pipeline for ASVspoof5 tar archives."""

    def __init__(self, 
                 data_root: str = "N:\\ASV5",
                 mel_extractor: Optional[MelSpectrogramExtractor] = None,
                 n_workers: int = 4):

        self.data_root = Path(data_root)
        self.mel_extractor = mel_extractor or MelSpectrogramExtractor()
        self.n_workers = n_workers
        self.extract_dir = self.data_root / "extracted"
        self.mel_cache_dir = self.data_root / "mel_specs"

        self.extract_dir.mkdir(exist_ok=True)
        self.mel_cache_dir.mkdir(exist_ok=True)

    def extract_all_tars(self, tar_files: list) -> None:

        print(f"Extracting {len(tar_files)} tar archives with {self.n_workers} workers...")

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {
                executor.submit(
                    self.mel_extractor.extract_tar_archive, 
                    tar_file, 
                    self.extract_dir
                ): tar_file for tar_file in tar_files
            }

            for future in as_completed(futures):
                try:
                    future.result()
                except Exception as e:
                    print(f"Error in extraction task: {e}")

    def process_audio_files(self) -> dict:

        audio_files = list(self.extract_dir.rglob("*.flac"))

        if not audio_files:
            print(f"✗ No FLAC files found in {self.extract_dir}")
            return {}

        print(f"Found {len(audio_files)} FLAC files. Processing...")

        stats = {
            'total': len(audio_files),
            'processed': 0,
            'failed': 0,
            'mel_specs': []
        }

        with ThreadPoolExecutor(max_workers=self.n_workers) as executor:
            futures = {}

            for audio_file in audio_files:

                rel_path = audio_file.relative_to(self.extract_dir)
                output_file = self.mel_cache_dir / rel_path.with_suffix(".npy")
                output_file.parent.mkdir(parents=True, exist_ok=True)

                future = executor.submit(
                    self.mel_extractor.process_single_file,
                    str(audio_file),
                    str(output_file)
                )
                futures[future] = (audio_file, output_file)

            for future in tqdm(as_completed(futures), total=len(futures)):
                audio_file, output_file = futures[future]
                try:
                    mel_spec = future.result()
                    if mel_spec is not None:
                        stats['processed'] += 1
                        stats['mel_specs'].append({
                            'audio_path': str(audio_file),
                            'mel_spec_path': str(output_file),
                            'shape': mel_spec.shape
                        })
                except Exception as e:
                    print(f"\n✗ Error processing {audio_file}: {e}")
                    stats['failed'] += 1

        return stats



if __name__ == "__main__":

    tar_files = [
        "N:\\ASV5\\flac_T_aa.tar",
        "N:\\ASV5\\flac_T_ab.tar",
        "N:\\ASV5\\flac_T_ac.tar",
        "N:\\ASV5\\flac_T_ad.tar",
        "N:\\ASV5\\flac_T_ae.tar"
    ]

    mel_extractor = MelSpectrogramExtractor(
        sample_rate=16000,
        n_fft=1024,
        hop_length=512,
        n_mels=128,
        f_min=50,
        f_max=7500,
        power=2.0,
        cache_dir="N:\\ASV5\\mel_cache"
    )


    preprocessor = ASVSpoof5DatasetPreprocessor(
        data_root="N:\\ASV5",
        mel_extractor=mel_extractor,
        n_workers=4
    )

    preprocessor.extract_all_tars(tar_files)

    stats = preprocessor.process_audio_files()

    print("\n" + "="*60)
    print("PREPROCESSING COMPLETE")
    print("="*60)
    print(f"Total files: {stats['total']}")
    print(f"Successfully processed: {stats['processed']}")
    print(f"Failed: {stats['failed']}")
    print(f"Mel-specs cached at: N:\\ASV5\\mel_cache")



    import json
    metadata = {
        'sample_rate': mel_extractor.sample_rate,
        'n_fft': mel_extractor.n_fft,
        'hop_length': mel_extractor.hop_length,
        'n_mels': mel_extractor.n_mels,
        'f_min': mel_extractor.f_min,
        'f_max': mel_extractor.f_max,
        'power': mel_extractor.power,
        'files_processed': stats['processed'],
        'mel_specs': stats['mel_specs']
    }

    with open("N:\\ASV5\\preprocessing_metadata.json", 'w') as f:
        json.dump(metadata, f, indent=2)

    print(f"Metadata saved to: N:\\ASV5\\preprocessing_metadata.json")




import tarfile
from pathlib import Path
from tqdm import tqdm

def extract_tar(tar_path, extract_dir):
    """Extract tar file with progress."""
    with tarfile.open(tar_path, 'r:*') as tar:
        members = tar.getmembers()
        with tqdm(total=len(members), desc=f"Extracting {tar_path.name}") as pbar:
            for member in members:
                tar.extract(member, path=extract_dir)
                pbar.update()
    print(f"✓ {tar_path.name} extracted")

def extract_all_tar(tar_files, extract_dir):
    """Extract all tars with parallel processing."""
    extract_dir.mkdir(parents=True, exist_ok=True)

    for tar_file in tar_files:
        extract_tar(tar_file, extract_dir)

    return extract_dir

if __name__ == "__main__":
    root = Path("N:/ASV5")
    extract_dir = root / "extracted_dev"

    tar_files = [
        root / "flac_D_aa.tar",
        root / "flac_D_ab.tar",
        root / "flac_D_ac.tar"
    ]

    extract_all_tar(tar_files, extract_dir)

    audio_files = list(extract_dir.rglob("*.flac"))
    print(f"Extracted {len(audio_files)} audio files.")



import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed
import json

class GPUMelExtractor:

    def __init__(self, 
                 sample_rate=16000,
                 n_fft=1024,
                 hop_length=512,
                 n_mels=128,
                 f_min=50,
                 f_max=7500,
                 device='cuda'):

        self.sample_rate = sample_rate
        self.device = device

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=sample_rate,
            n_fft=n_fft,
            hop_length=hop_length,
            n_mels=n_mels,
            f_min=f_min,
            f_max=f_max,
            power=2.0,
            norm='slaney',
            mel_scale='htk'
        ).to(device)

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(
            stype='power'
        ).to(device)

    def process_file(self, audio_path, output_path):

        try:
            import soundfile as sf

            audio, sr = sf.read(audio_path)
            audio = audio.astype(np.float32)

            if len(audio) == 0:
                return False

            audio_tensor = torch.from_numpy(audio).unsqueeze(0)

            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio_tensor = resampler(audio_tensor)

            audio_tensor = audio_tensor.to(self.device)
            mel_spec = self.mel_transform(audio_tensor)
            mel_spec_db = self.amplitude_to_db(mel_spec)


            mel_spec_np = mel_spec_db.squeeze(0).cpu().numpy()
            np.save(output_path, mel_spec_np)

            return True
        except Exception as e:
            print(f"Error: {audio_path} - {e}")
            return False

def process_dev_set():

    data_root = Path("N:/ASV5")
    extract_dir = data_root / "extracted_dev"
    mel_cache_dir = data_root / "mel_cache_dev"
    extract_dir.mkdir(exist_ok=True)
    mel_cache_dir.mkdir(exist_ok=True)

    audio_files = list(extract_dir.rglob("*.flac"))
    print(f"Found {len(audio_files)} FLAC files. Processing with GPU...")

    extractor = GPUMelExtractor(device='cuda')

    stats = {'success': 0, 'failed': 0}

    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = {}

        for audio_file in audio_files:
            rel_path = audio_file.relative_to(extract_dir)
            output_file = mel_cache_dir / rel_path.with_suffix(".npy")
            output_file.parent.mkdir(parents=True, exist_ok=True)

            future = executor.submit(extractor.process_file, str(audio_file), str(output_file))
            futures[future] = audio_file

        for future in tqdm(as_completed(futures), total=len(futures)):
            if future.result():
                stats['success'] += 1
            else:
                stats['failed'] += 1

    print(f"Processed: {stats['success']}/{len(audio_files)}")
    print(f"Mel-specs at: {mel_cache_dir}")

if __name__ == "__main__":
    process_dev_set()



import torch
import torchaudio
import numpy as np
from pathlib import Path
from tqdm import tqdm
from concurrent.futures import ThreadPoolExecutor, as_completed

class GPUMelExtractor:
    def __init__(self, device='cuda'):
        self.sample_rate = 16000
        self.device = device

        self.mel_transform = torchaudio.transforms.MelSpectrogram(
            sample_rate=16000, n_fft=1024, hop_length=512, n_mels=128,
            f_min=50, f_max=7500, power=2.0, norm='slaney', mel_scale='htk'
        ).to(device)

        self.amplitude_to_db = torchaudio.transforms.AmplitudeToDB(stype='power').to(device)

    def process_file(self, audio_path, output_path):
        try:
            import soundfile as sf
            audio, sr = sf.read(audio_path)
            audio = audio.astype(np.float32)

            if len(audio) == 0:
                return False

            audio_tensor = torch.from_numpy(audio).unsqueeze(0)
            if sr != self.sample_rate:
                resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
                audio_tensor = resampler(audio_tensor)

            audio_tensor = audio_tensor.to(self.device)
            mel_spec = self.mel_transform(audio_tensor)
            mel_spec_db = self.amplitude_to_db(mel_spec)
            mel_spec_np = mel_spec_db.squeeze(0).cpu().numpy()
            np.save(output_path, mel_spec_np)

            return True
        except Exception as e:
            print(f"Error: {audio_path} - {e}")
            return False

root = Path("N:/ASV5")
extract_dir = root / "extracted_eval"
mel_cache_dir = root / "mel_cache_eval"
mel_cache_dir.mkdir(exist_ok=True)

audio_files = list(extract_dir.rglob("*.flac"))
print(f"Found {len(audio_files)} FLAC files. Processing with GPU...")

extractor = GPUMelExtractor(device='cuda')
stats = {'success': 0, 'failed': 0}

with ThreadPoolExecutor(max_workers=4) as executor:
    futures = {}
    for audio_file in audio_files:
        rel_path = audio_file.relative_to(extract_dir)
        output_file = mel_cache_dir / rel_path.with_suffix(".npy")
        output_file.parent.mkdir(parents=True, exist_ok=True)
        future = executor.submit(extractor.process_file, str(audio_file), str(output_file))
        futures[future] = audio_file

    for future in tqdm(as_completed(futures), total=len(futures)):
        if future.result():
            stats['success'] += 1
        else:
            stats['failed'] += 1

print(f"\n{'='*60}")
print("EVAL SET PREPROCESSING COMPLETE")
print(f"{'='*60}")
print(f"Processed: {stats['success']}/{len(audio_files)}")
print(f"Mel-specs at: {mel_cache_dir}")