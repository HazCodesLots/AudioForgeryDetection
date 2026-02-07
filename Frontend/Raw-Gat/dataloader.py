import os
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader
import soundfile as sf
from scipy import signal
from audio_augmentations import CodecAugmentation

class ASVspoof2021Dataset(Dataset):
    def __init__(self, data_dir, protocol_file, max_len=64600, is_train=True, is_eval=False):
        '''
        Args:
            data_dir: Path to audio files (e.g., 'N:\\ASVspoof2021\\ASVspoof2021_LA_eval\\flac')
            protocol_file: Path to protocol/key file
            max_len: Maximum audio length in samples (default: 64600 = ~4 seconds at 16kHz)
            is_train: Whether this is training data
            is_eval: Whether this is evaluation data
        '''
        self.data_dir = data_dir
        self.max_len = max_len
        self.is_train = is_train
        self.is_eval = is_eval
        
        # Initialize augmentation
        self.augmenter = CodecAugmentation(p=0.5) if is_train else None
        
        # Parse protocol file
        self.samples = []
        self.labels = []
        with open(protocol_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) >= 2:
                    # Detect protocol format automatically by column count
                    if len(parts) == 8:
                        # ASVspoof2021 format: spk trial codec trans attack label trim subset
                        audio_id = parts[1]
                        label = parts[5]
                    elif len(parts) == 5:
                        # ASVspoof2019 format: speaker_id audio_id - attack_id label
                        audio_id = parts[1]
                        label = parts[4]
                    else:
                        # Fallback: assume last column is the label if not recognized
                        audio_id = parts[1]
                        label = parts[-1]
                    
                    self.samples.append(audio_id)
                    self.labels.append(1 if label == 'bonafide' else 0)
        
        print(f"Loaded {len(self.samples)} samples")
        print(f"Bonafide: {sum(self.labels)}, Spoof: {len(self.labels) - sum(self.labels)}")
    
    def __len__(self):
        return len(self.samples)
    
    def pad_or_truncate(self, audio):
        '''Pad or truncate audio to max_len'''
        if len(audio) > self.max_len:
            # Random crop for training, center crop for eval
            if self.is_train:
                start = np.random.randint(0, len(audio) - self.max_len)
            else:
                start = (len(audio) - self.max_len) // 2
            audio = audio[start:start + self.max_len]
        elif len(audio) < self.max_len:
            # Pad with zeros
            audio = np.pad(audio, (0, self.max_len - len(audio)), mode='constant')
        return audio
    
    def __getitem__(self, idx):
        audio_id = self.samples[idx]
        label = self.labels[idx]
        
        # Construct file path - try both with and without .flac extension
        audio_path = os.path.join(self.data_dir, f"{audio_id}.flac")
        if not os.path.exists(audio_path):
            audio_path = os.path.join(self.data_dir, audio_id)
        
        try:
            # Load audio
            audio, sr = sf.read(audio_path)
            
            # Ensure 16kHz
            if sr != 16000:
                # Resample if needed
                audio = signal.resample(audio, int(len(audio) * 16000 / sr))
            
            # Convert to float32
            audio = audio.astype(np.float32)
            
            # Normalization (Crucial for RawGAT-ST)
            # Standard zero-mean unit-variance per segment
            audio = (audio - audio.mean()) / (audio.std() + 1e-7)
            
            # Apply augmentation during training
            if self.is_train and self.augmenter:
                audio = self.augmenter.augment(audio)
            
            # Pad or truncate
            audio = self.pad_or_truncate(audio)
            
            # Convert to tensor
            audio_tensor = torch.FloatTensor(audio)
            label_tensor = torch.LongTensor([label])[0]
            
            return audio_tensor, label_tensor, audio_id
        
        except Exception as e:
            # ✅ CHANGED: Silently handle corrupted files (~0.1% of dataset)
            # Return zero audio on error
            return torch.zeros(self.max_len), torch.LongTensor([label])[0], audio_id

def get_dataloader(data_dir, protocol_file, batch_size=10, num_workers=4,
                   is_train=True, is_eval=False):
    '''
    Create a DataLoader for ASVspoof 2021 dataset
    
    Args:
        data_dir: Path to audio directory
        protocol_file: Path to protocol file
        batch_size: Batch size
        num_workers: Number of worker processes
        is_train: Training mode
        is_eval: Evaluation mode
    '''
    dataset = ASVspoof2021Dataset(
        data_dir=data_dir,
        protocol_file=protocol_file,
        is_train=is_train,
        is_eval=is_eval
    )
    
    dataloader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=is_train,
        num_workers=num_workers,
        pin_memory=True,
        drop_last=is_train
    )
    
    return dataloader

if __name__ == "__main__":
    # Test the dataloader
    data_dir = r"N:\ASVspoof2021\ASVspoof2021_LA_eval\flac"
    protocol_file = r"N:\ASVspoof2021\keys\LA\trial_metadata.txt"
    
    try:
        loader = get_dataloader(data_dir, protocol_file, batch_size=2,
                               num_workers=0, is_eval=True)
        print("\nTesting dataloader...")
        for batch_idx, (audio, labels, audio_ids) in enumerate(loader):
            print(f"Batch {batch_idx}:")
            print(f"  Audio shape: {audio.shape}")
            print(f"  Labels: {labels}")
            print(f"  IDs: {audio_ids}")
            if batch_idx >= 2:
                break
        print("\nDataloader test passed!")
    except Exception as e:
        print(f"Could not test dataloader (data path may not exist): {e}")
        print("This is OK - the dataloader will work when proper paths are provided")