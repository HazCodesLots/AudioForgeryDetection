#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import os
import numpy as np
import soundfile as sf
import torch
from torch.utils.data import Dataset, DataLoader
from typing import List


class ASVspoof2021EvalDataset(Dataset):
    def __init__(self, flac_dir: str, protocol_path: str, sample_len: int = 64000):
        self.flac_dir = flac_dir
        self.sample_len = sample_len
        self.protocol = np.load(protocol_path, allow_pickle=True)
        self.utt_ids = [utt_id for utt_id in self.protocol]

    def __len__(self):
        return len(self.utt_ids)

    def __getitem__(self, idx):
        utt_id = self.utt_ids[idx]
        file_path = os.path.join(self.flac_dir, utt_id + ".flac")
        wav, _ = sf.read(file_path)

        if len(wav) < self.sample_len:
            wav = np.pad(wav, (0, self.sample_len - len(wav)), mode='constant')
        else:
            wav = wav[:self.sample_len]

        wav_tensor = torch.FloatTensor(wav)
        return wav_tensor, 0, utt_id


def get_data_loaders(flac_dir: str, protocol_path: str, batch_size: int = 4, is_eval=True):
    dataset = ASVspoof2021EvalDataset(flac_dir, protocol_path)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=2)
    return loader

