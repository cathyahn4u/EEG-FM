import os
import glob
import mne
import torch
import numpy as np
from data_handling.base_dataset import BaseEEGDataset
from moabb.datasets import BNCI2014_001
import warnings

warnings.filterwarnings("ignore", category=RuntimeWarning)

class BCICompetitionIV2aDataset(BaseEEGDataset):
    def __init__(self, path, **kwargs):
        self.dataset = BNCI2014_001()
        super().__init__(path, **kwargs)
        self.preprocessing_fn = None

    def _load_data_index(self):
        subjects = self.dataset.subject_list
        file_list = [f"subject_{s}" for s in subjects]
        labels = [0] * len(file_list)
        return file_list, labels, subjects

    def __getitem__(self, idx):
        subject_id = self.subjects[idx]
        raw_dict, _ = self.dataset.get_data(subjects=[subject_id])
        raw = raw_dict[subject_id]['0train']
        events, event_id = mne.events_from_annotations(raw)
        labels = events[:, -1] - 1
        raw.pick_channels(raw.ch_names[:22])

        if self.preprocessing_fn:
            if self.preprocessing_fn.__name__ == 'preprocess_for_cbramod_original':
                processed_data = self.preprocessing_fn(raw, self.target_fs)
            else:
                processed_data = self.preprocessing_fn(raw, self.target_fs, self.standard_channels)
        else:
            processed_data = standard_preprocess(raw, self.target_fs, self.standard_channels)

        epochs = mne.Epochs(mne.io.RawArray(processed_data, raw.info), events, tmin=0, tmax=4, baseline=None, preload=True)
        data = epochs.get_data(copy=False)
        labels = epochs.events[:, -1]

        return {
            "signal": torch.tensor(data, dtype=torch.float32),
            "label": torch.tensor(labels, dtype=torch.long),
            "task_name": "BCICompetitionIV2a",
            "metadata": {"subject": subject_id}
        }

class GenericEEGDataset(BaseEEGDataset):
    def __init__(self, path, task_name, task_type, file_extension='*.edf', **kwargs):
        self.task_name = task_name
        self.task_type = task_type
        self.file_extension = file_extension
        super().__init__(path, **kwargs)
        self.preprocessing_fn = None

    def _load_data_index(self):
        file_list = glob.glob(os.path.join(self.path, '**', self.file_extension), recursive=True)
        if not file_list:
            print(f"Warning: No files found for {self.task_name} at {self.path} with ext {self.file_extension}")
        return file_list, [0] * len(file_list), [0] * len(file_list)

    def __getitem__(self, idx):
        file_path = self.file_list[idx]
        try:
            if '.edf' in file_path.lower():
                raw = mne.io.read_raw_edf(file_path, preload=True, verbose=False)
            elif '.bdf' in file_path.lower():
                raw = mne.io.read_raw_bdf(file_path, preload=True, verbose=False)
            else:
                raw = mne.io.read_raw(file_path, preload=True, verbose=False)
        except Exception as e:
            print(f"Error loading {file_path}: {e}")
            return None

        proc_fn = self.preprocessing_fn or standard_preprocess
        processed_data = proc_fn(raw, self.target_fs, self.standard_channels)
        segments = self._segment_data(processed_data)
        segments = np.expand_dims(segments, axis=1)

        return {
            "signal": torch.tensor(segments, dtype=torch.float32),
            "label": torch.tensor(self.labels[idx], dtype=torch.long),
            "task_name": self.task_name,
            "metadata": {"file": os.path.basename(file_path), "task_type": self.task_type}
        }

def create_dataset_class(class_name, task_name, task_type, file_extension='*.edf'):
    def __init__(self, path, **kwargs):
        GenericEEGDataset.__init__(self, path, task_name, task_type, file_extension, **kwargs)
    return type(class_name, (GenericEEGDataset,), {"__init__": __init__})

# config의 properties를 기반으로 동적 클래스 생성
# 이 부분은 main.py나 datamodule에서 config를 읽은 후 호출될 수 있음
# 여기서는 예시로 몇 개만 정의
TUABDataset = create_dataset_class("TUABDataset", "TUAB", "Clinical")
TUEVDataset = create_dataset_class("TUEVDataset", "TUEV", "Artifact")
GISTSMRBCIDataset = create_dataset_class("GISTSMRBCIDataset", "GISTSMRBCI", "MotorImagery", file_extension='*.bdf')
# ... 기타 데이터셋 ...
