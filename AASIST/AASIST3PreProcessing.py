
class PreEmphasis(nn.Module:):
    def __init__(self, coef=0.97):
        super(PreEmphasis, self).__init__()
        self.coef = coef
        self.register_buffer('flipped_filter', torch.FloatTensor([-self.coef, 1.0]).unsqueeze(0).unsqueeze(0))

        def forward(self, x: torch.Tensor) -> torch.Tensor:
            if x.dim() == 2:
                x = x.unsqueeze(1)
                squeeze_output = True
            else:
                squeeze_output = False

            x_padded = torch.nn.functional.pad(x, (1, 0), mode='replicate')
            x_preemphasized = torch.nn.functional.conv1d(x_padded, self.flipped_filter)

            if squeeze_output:
                x_preemphasized = x_preemphasized.squeeze(1)

            return x_preemphasized


class AudioProcessor:

    def __init__(self, sample_rate: int = 16000, max_length_seconds: float = 4.0):
        self.sample_rate = sample_rate
        self.max_length_seconds = max_length_seconds
        self.max_length_samples = int(sample_rate * max_length_seconds)

    def load_audio(self, audio_path: str) -> Tuple[torch.Tensor, int]:
        waveform, sr = torchaudio.load(audio_path)

        if waveform.shape[0] != 1:
            waveform = torch.mean(waveform, dim=0, keepdim=True)

        if sr != self.sample_rate:
            resampler = torchaudio.transforms.Resample(sr, self.sample_rate)
            waveform = resampler(waveform)

        return waveform.squeeze(0), self.sample_rate

    def pad_or_crop(self, audio: torch.Tensor, length: Optional[int] = None) -> torch.Tensor:
        if length is None:
            length = self.max_length_samples

        current_length = audio.shape[0]

        if current_length > length:
            start_idx = (current_length - length) // 2
            audio = audio[start_idx:start_idx + length]
        elif current_length < length:
            pad_amount = length - current_length
            audio = torch.nn.functional.pad(audio, (0, pad_amount), mode='constant', value=0)

        return audio

    def create_sliding_windows(self, audio: torch.Tensor, window_seconds: float = 4.0, overlap_seconds: float = 2.0) -> torch.Tensor:
        window_samples = int(self.sample_rate * window_seconds)
        stride_samples = int(self.sample_rate * (window_seconds - overlap_seconds))

        audio_length = audio.shape[0]

        if audio_length <= window_samples:
            return self.pad_or_crop(audio, window_samples).unsqueeze(0)

        num_windows = 1 + (audio_length - window_samples) // stride_samples

        if (audio_length - windows_samples) % stride_samples !=0:
            num_windows += 1

        windows = []
        for i in range(num_windows):
            start = i * stride_samples
            end = start + window_samples

            if end > audio_length:

                window = audio[start:]
                window = torch.nn.functional.pad(window, (0, window_samples - window.shape[0]), mode='constant', value=0)
            else:
                window = audio[start:end]
            
            windows.append(window)
        
        return torch.stack(windows)


class PositionalEmbedding(nn.Module):

    def __init__(self, d_model: int, max_len: int = 5000, temprature: int = 10000):
        super(PositionalEmbedding, self).__init__()

        self.d_model = d_model

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)

        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(temprature) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe)

    def forward(self, x: torch.Tensor) -> torch.Tensor:

        if x.dim() == 3:
            seq_len = x.size(1)
            return x + self.pe[:seq_len, :].unsqueeze(0)
        elif x.dim() == 2:
            seq_len = x.size(0)
            return x + self.pe[:seq_len, :]
        else:
            raise ValueError(f"Expected 2D or 3D input, got {x.dim()}D")
