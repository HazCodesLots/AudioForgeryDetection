import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import os
from pathlib import Path

class LFCCExtractor(nn.Module):
    """
    Linear Frequency Cepstral Coefficients (LFCC) feature extractor
    Better than MFCC for deepfake detection as it captures linear frequency patterns.
    This version avoids torchaudio and uses native torch.stft.
    """
    def __init__(
        self,
        sample_rate=16000,
        n_fft=512,
        win_length=400,
        hop_length=160,
        n_lfcc=60,
        n_filter=60
    ):
        super().__init__()
        self.sample_rate = sample_rate
        self.n_fft = n_fft
        self.win_length = win_length
        self.hop_length = hop_length
        self.n_lfcc = n_lfcc
        self.n_filter = n_filter
        
        # Register hamming window
        self.register_buffer('window', torch.hamming_window(win_length))
        
    def forward(self, waveform):
        """
        Args:
            waveform: (batch, samples)
        Returns:
            lfcc: (batch, n_lfcc, time_frames)
        """
        # Handle (batch, 1, samples) input if necessary
        if waveform.ndim == 3:
            waveform = waveform.squeeze(1)
            
        # Compute power spectrogram using native torch.stft
        # STFT requires (batch, samples)
        stft = torch.stft(
            waveform,
            n_fft=self.n_fft,
            hop_length=self.hop_length,
            win_length=self.win_length,
            window=self.window,
            center=True,
            pad_mode='reflect',
            normalized=False,
            onesided=True,
            return_complex=True
        )
        
        # Get magnitude squared (power)
        spec = stft.abs().pow(2.0)  # (batch, freq_bins, time)
        
        # Linear filter bank (uniform spacing unlike mel-scale)
        linear_filter = self._get_linear_filterbank(spec.size(1))
        linear_filter = linear_filter.to(spec.device)
        
        # Apply filter bank
        spec = spec.transpose(1, 2)  # (batch, time, freq)
        linear_spec = torch.matmul(spec, linear_filter.T)  # (batch, time, n_filter)
        
        # Log compression
        linear_spec = torch.log(linear_spec + 1e-6)
        
        # DCT (Discrete Cosine Transform) to get cepstral coefficients
        lfcc = self._dct(linear_spec, norm='ortho')[:, :, :self.n_lfcc]
        
        return lfcc.transpose(1, 2)  # (batch, n_lfcc, time)
    
    def _get_linear_filterbank(self, n_freqs):
        """Create linear spaced filterbank"""
        filters = torch.zeros(self.n_filter, n_freqs)
        freq_bins = torch.linspace(0, n_freqs - 1, self.n_filter + 2)
        
        for i in range(self.n_filter):
            left = int(freq_bins[i])
            center = int(freq_bins[i + 1])
            right = int(freq_bins[i + 2])
            
            # Triangular filter
            for j in range(left, center):
                filters[i, j] = (j - left) / (center - left)
            for j in range(center, right):
                filters[i, j] = (right - j) / (right - center)
                
        return filters
    
    def _dct(self, x, norm=None):
        """Discrete Cosine Transform"""
        x = x.transpose(-2, -1)
        N = x.shape[-1]
        v = torch.cat([x[:, :, ::2], x[:, :, 1::2].flip(-1)], dim=-1)
        
        Vc = torch.fft.fft(v, dim=-1)
        k = -torch.arange(N, dtype=x.dtype, device=x.device) * np.pi / (2 * N)
        W_r = torch.cos(k)
        W_i = torch.sin(k)
        
        V = Vc.real * W_r - Vc.imag * W_i
        
        if norm == 'ortho':
            V[:, :, 0] /= np.sqrt(N) * 2
            V[:, :, 1:] /= np.sqrt(N / 2) * 2
            
        return V.transpose(-2, -1)
