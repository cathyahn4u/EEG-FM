import torch
import mne
import numpy as np
from torch.utils.data import Dataset
from abc import ABC, abstractmethod

# 모든 EEG 데이터셋 클래스를 위한 추상 기본 클래스
class BaseEEGDataset(Dataset, ABC):
    def __init__(self, path, target_fs=256, seq_len_sec=4, **kwargs):
        self.path = path
        self.target_fs = target_fs
        self.seq_len_sec = seq_len_sec
        self.seq_len_points = int(target_fs * seq_len_sec)
        self.standard_channels = sorted(list(set(mne.channels.make_standard_montage('standard_1005').ch_names + ['T1', 'T2', 'A1', 'A2'])))
        self.file_list, self.labels, self.subjects = self._load_data_index()
    
    @abstractmethod
    def _load_data_index(self):
        raise NotImplementedError

    def __len__(self):
        return len(self.file_list)

    @abstractmethod
    def __getitem__(self, idx):
        raise NotImplementedError

    def _segment_data(self, data):
        num_channels, total_points = data.shape
        num_segments = total_points // self.seq_len_points
        
        if num_segments == 0:
            pad_width = self.seq_len_points - total_points
            padded_data = np.pad(data, ((0, 0), (0, pad_width)), 'constant')
            return np.expand_dims(padded_data, axis=0)
            
        segmented_data = np.array([
            data[:, i*self.seq_len_points : (i+1)*self.seq_len_points]
            for i in range(num_segments)
        ])
        return segmented_data
