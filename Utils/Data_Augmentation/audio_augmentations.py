import numpy as np
import scipy.signal as signal

class CodecAugmentation:
    """
    Applies random codec-like augmentations to audio signals.
    Optimized to minimize CPU overhead in the dataloader loop.
    """
    def __init__(self, p=0.5):
        self.p = p
        # Pre-calculate constants
        self.A = 87.6
        self.mu = 255
        self.log_A_plus_1 = 1 + np.log(self.A)
        self.log_mu_plus_1 = np.log(1 + self.mu)
        
        # Pre-calculate PSTN Butterworth filter (SOS format for stability)
        self.sos_pstn = signal.butter(6, [300, 3400], btype='bandpass', fs=16000, output='sos')

    def alaw_encode(self, x):
        x_abs = np.abs(x)
        inv_A = 1/self.A
        # Fast vectorized encoding
        y = np.where(x_abs < inv_A, 
                     (self.A * x_abs) / self.log_A_plus_1,
                     (1 + np.log(np.maximum(self.A * x_abs, 1e-10))) / self.log_A_plus_1)
        return np.sign(x) * y

    def alaw_decode(self, y):
        y_abs = np.abs(y)
        inv_log_A_plus_1 = 1 / self.log_A_plus_1
        # Fast vectorized decoding
        x = np.where(y_abs < inv_log_A_plus_1,
                     (y_abs * self.log_A_plus_1) / self.A,
                     np.exp(y_abs * self.log_A_plus_1 - 1) / self.A)
        return np.sign(y) * x

    def mulaw_encode(self, x):
        return np.sign(x) * np.log(1 + self.mu * np.abs(x)) / self.log_mu_plus_1

    def mulaw_decode(self, y):
        return np.sign(y) * (1/self.mu) * (np.power(1 + self.mu, np.abs(y)) - 1)

    def apply_alaw(self, x):
        y = self.alaw_encode(np.clip(x, -1, 1))
        # 8-bit quantization is internal to the codec feel
        y = np.round(y * 127) / 127
        return self.alaw_decode(y)

    def apply_mulaw(self, x):
        y = self.mulaw_encode(np.clip(x, -1, 1))
        y = np.round(y * 127) / 127
        return self.mulaw_decode(y)

    def apply_pstn(self, x):
        """Simulate PSTN channel using pre-calculated SOS filter"""
        return signal.sosfilt(self.sos_pstn, x)

    def apply_resample_quantize(self, x, target_sr=8000, current_sr=16000):
        # target_sr/current_sr is 8000/16000 = 1/2 or 12000/16000 = 3/4
        gcd = np.gcd(target_sr, current_sr)
        up, down = target_sr // gcd, current_sr // gcd
        
        x_down = signal.resample_poly(x, up, down)
        bits = np.random.choice([8, 10, 12])
        levels = 2**(bits-1)
        x_down = np.round(x_down * levels) / levels
        return signal.resample_poly(x_down, down, up)

    def augment(self, audio):
        if np.random.random() > self.p:
            return audio
            
        aug_type = np.random.choice(['alaw', 'mulaw', 'pstn', 'resample'])
        
        if aug_type == 'alaw':
            return self.apply_alaw(audio)
        elif aug_type == 'mulaw':
            return self.apply_mulaw(audio)
        elif aug_type == 'pstn':
            # Combine PSTN filter with a nonlinear codec
            audio = self.apply_pstn(audio)
            return self.apply_alaw(audio) if np.random.random() > 0.5 else self.apply_mulaw(audio)
        elif aug_type == 'resample':
            target_sr = np.random.choice([8000, 12000])
            return self.apply_resample_quantize(audio, target_sr=target_sr)
            
        return audio
