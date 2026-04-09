class AudioAugmentation:

    @staticmethod
    def add_gaussian_noise(audio, noise_factor=0.005):
        noise = torch.randn_like(audio) * noise_factor
        return audio + noise
    
    @staticmethod
    def pitch_shift(audio, sample_rate=16000, n_steps=None):
        if n_steps is None:
            n_steps = random.choice([-2, -1, 1, 2])
        
        audio_np = audio.numpy()
        shifted = librosa.effects.pitch_shift(audio_np, sr=sample_rate, n_steps=n_steps)
        return torch.from_numpy(shifted).float()
    
    @staticmethod
    def time_stretch(audio, rate=None):
        if rate is None:
            rate = random.uniform(0.9, 1.1)
        
        audio_np = audio.numpy()
        stretched = librosa.effects.time_stretch(audio_np, rate=rate)
        
        if len(stretched) > len(audio):
            stretched = stretched[:len(audio)]
        elif len(stretched) < len(audio):
            stretched = np.pad(stretched, (0, len(audio) - len(stretched)))
        
        return torch.from_numpy(stretched).float()
    
    @staticmethod
    def apply_random_augmentations(audio, sample_rate=16000, prob=0.5):
        augmentations = []
        
        if random.random() < prob:
            audio = AudioAugmentation.add_gaussian_noise(audio)
        
        if random.random() < prob * 0.5:
            audio = AudioAugmentation.pitch_shift(audio, sample_rate)
        
        if random.random() < prob * 0.5:
            audio = AudioAugmentation.time_stretch(audio)
        
        return audio
