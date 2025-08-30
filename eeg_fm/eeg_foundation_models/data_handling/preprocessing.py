import mne
import numpy as np
import torch
from einops import rearrange

def standard_preprocess(raw, target_fs, standard_channels):
    """
    모든 데이터셋에 적용될 수 있는 표준 전처리 파이프라인.
    """
    eeg_channels = mne.pick_types(raw.info, eeg=True)
    raw.pick(eeg_channels)
    raw.resample(target_fs, npad='auto')
    raw.filter(l_freq=0.5, h_freq=40.0, fir_design='firwin', verbose=False)
    raw.notch_filter(freqs=np.arange(60.0, target_fs/2, 60.0), fir_design='firwin', verbose=False)
    
    missing_in_raw = list(set(raw.ch_names) - set(standard_channels))
    if missing_in_raw:
        raw.drop_channels(missing_in_raw, on_missing='ignore')

    data, _ = raw.get_data(return_times=True)
    
    ch_map = {ch: i for i, ch in enumerate(raw.ch_names)}
    standard_data = np.zeros((len(standard_channels), data.shape[1]))
    for i, ch_name in enumerate(standard_channels):
        if ch_name in ch_map:
            standard_data[i, :] = data[ch_map[ch_name], :]
            
    return standard_data

def preprocess_for_cbramod_original(raw, target_fs, **kwargs):
    """
    CBraMod 원본 GitHub의 전처리 방식을 복제한 함수.
    """
    raw.resample(250, npad='auto')
    raw.filter(l_freq=0, h_freq=40.0, fir_design='firwin', verbose=False)
    data = raw.get_data()
    data -= np.mean(data, axis=1, keepdims=True)
    data /= np.std(data, axis=1, keepdims=True)
    
    info = mne.create_info(ch_names=raw.ch_names, sfreq=250, ch_types='eeg')
    processed_raw = mne.io.RawArray(data, info, verbose=False)
    processed_raw.resample(target_fs, npad='auto')
    return processed_raw.get_data()

def preprocess_for_labram_original(signal_tensor):
    """
    LaBraM 원본 GitHub의 STFT 기반 특징 추출 방식을 복제한 함수.
    """
    x_reshaped = signal_tensor
    stft_out = torch.stft(x_reshaped, 
                          n_fft=256, 
                          hop_length=64,
                          win_length=256,
                          return_complex=True, 
                          window=torch.hann_window(256, device=signal_tensor.device))
    freq_data = torch.abs(stft_out).mean(dim=-1)
    return freq_data

PREPROCESSING_FN_MAP = {
    'standard': standard_preprocess,
    'cbramod_original': preprocess_for_cbramod_original,
}
