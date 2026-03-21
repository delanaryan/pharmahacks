import numpy as np 
import pandas as pd
from scipy.signal import  butter, filtfilt, iirnotch, resample, welch
from scipy.stats import kurtosis




def bandpass_filter(data, sfreq = 128, lowcut = 0.5, highcut = 45, fs = 128, order=4):
    '''
    Band-pass filters the data given in data 
    '''
    nyquist = 0.5 * fs
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    filtered_data = filtfilt(b, a, data, axis=1)
    return filtered_data

def notch_filter(data, fs=128, freq=60, quality_factor=30):
    '''
    Notch filters the data given in data to remove powerline noise at 50 Hz.
    '''
    b, a = iirnotch(freq / (fs / 2), quality_factor)
    filtered_data = filtfilt(b, a, data, axis=1)
    return filtered_data

def clip_amplitude(data, threshold_uv=150):
    '''
    Clips the amplitude of the data to a specified threshold to remove artifacts.
    '''
    threshold = threshold_uv * 1e-6  # Convert µV to V
    clipped_data = np.clip(data, -threshold, threshold)
    return clipped_data

def remove_movement_artifacts(data, fs, kurtosis_threshold=5, window_sec = 2,step_sec = 0.2, z_thresh):
    '''
    Removes epochs with high kurtosis, which may indicate movement artifacts.
    '''
    window_samples = int(window_sec * fs)
    step_samples = int(step_sec * fs)
    n_channels, n_points = data.shape

    rms_values, kurtosis_values = [], []
    for start in range(0, n_points - window_samples + 1, step_samples):
        end = start + window_samples
        window_data = data[:, start:end]
        rms = np.sqrt(np.mean(window_data**2))
        kurt = kurtosis(window_data.flatten())
        rms_values.append(rms)
        kurtosis_values.append(kurt)
    rms_values = np.array(rms_values)
    kurtosis_values = np.array(kurtosis_values)

    rms_threshold = np.mean(rms_values) + z_thresh * np.std(rms_values)
    
    for i, start in enumerate(range(0, n_points - window_samples + 1, step_samples)):
        if kurtosis_values[i] > kurtosis_threshold or rms_values[i] > rms_threshold:
            data[:, start:start+window_samples] = 1e-10  # Zero out the artifact window
    return data

def denoise_eeg(data, orig_sfreq=500, target_sfreq=128):
    # All filtering steps use orig_sfreq — data is still at 500 Hz here
    data = bandpass_filter(data, orig_sfreq)
    data = notch_filter(data, orig_sfreq)
    data = clip_amplitude(data)
    data = remove_movement_artifacts(data, orig_sfreq)  # orig_sfreq here too
    data = normalize_channels(data)
    data = downsample(data, orig_sfreq, target_sfreq)
    return data