#denoise.py
import numpy as np
from scipy.signal import butter, filtfilt, iirnotch, resample
from scipy.stats import kurtosis


def bandpass_filter(data, sfreq=500, lowcut=0.5, highcut=45.0, order=4):
    """
    Bandpass filter between lowcut and highcut Hz.
    Must be called at orig_sfreq (500 Hz) before downsampling.
    """
    nyquist = 0.5 * sfreq  # FIX: was using wrong 'fs' parameter
    low = lowcut / nyquist
    high = highcut / nyquist
    b, a = butter(order, [low, high], btype='band')
    return filtfilt(b, a, data, axis=1)


def notch_filter(data, sfreq=500, freq=60.0, quality_factor=30.0):
    """
    Notch filter at 60 Hz to remove North American powerline noise.
    Must be called at orig_sfreq (500 Hz) before downsampling.
    """
    b, a = iirnotch(freq / (sfreq / 2), quality_factor)  # FIX: was using wrong 'fs' parameter
    return filtfilt(b, a, data, axis=1)


def clip_amplitude(data, threshold_uv=150):
    """
    Clips amplitude to ±150 µV to suppress extreme artifact spikes.
    """
    threshold = threshold_uv * 1e-6
    return np.clip(data, -threshold, threshold)


def remove_movement_artifacts(data, sfreq=500, kurtosis_threshold=5,
                               window_sec=2, step_sec=0.2, z_thresh=0.1):
    """
    Zeroes out windows with abnormally high RMS energy or kurtosis.
    RMS catches broad movement bursts, kurtosis catches muscle artifacts.
    """
    window_samples = int(window_sec * sfreq)
    step_samples = int(step_sec * sfreq)
    n_channels, n_points = data.shape

    rms_values, kurtosis_values = [], []
    for start in range(0, n_points - window_samples + 1, step_samples):
        window_data = data[:, start:start + window_samples]
        rms_values.append(np.sqrt(np.mean(window_data ** 2)))
        kurtosis_values.append(kurtosis(window_data.flatten()))

    rms_values = np.array(rms_values)
    kurtosis_values = np.array(kurtosis_values)
    rms_threshold = np.mean(rms_values) + z_thresh * np.std(rms_values)

    n_zeroed = 0
    for i, start in enumerate(range(0, n_points - window_samples + 1, step_samples)):
        if kurtosis_values[i] > kurtosis_threshold or rms_values[i] > rms_threshold:
            data[:, start:start + window_samples] = 1e-10
            n_zeroed += 1

    print(f"    → Zeroed {n_zeroed}/{len(rms_values)} artifact windows")
    return data


def normalize_channels(data):
    """
    Z-score normalizes each channel independently.
    Essential for SVM — brings all channels to same scale.
    """
    mean = data.mean(axis=1, keepdims=True)
    std = data.std(axis=1, keepdims=True)
    std[std == 0] = 1e-10
    return (data - mean) / std


def downsample(data, orig_sfreq=500, target_sfreq=128):
    """
    Downsamples from orig_sfreq to target_sfreq.
    Must be called AFTER all filtering steps.
    """
    target_len = int(data.shape[1] * target_sfreq / orig_sfreq)
    return resample(data, target_len, axis=1)


def denoise_eeg(data, orig_sfreq=500, target_sfreq=128):
    """
    Full pipeline: bandpass -> notch -> clip -> artifacts -> normalize -> downsample
    All filtering uses orig_sfreq (500 Hz), downsampling happens last.
    """
    print(f"  Input shape: {data.shape} at {orig_sfreq} Hz")

    print(f"  [1/6] Bandpass (0.5-45 Hz)...", end=' ')
    data = bandpass_filter(data, orig_sfreq)
    print("done.")

    print(f"  [2/6] Notch (60 Hz)...", end=' ')
    data = notch_filter(data, orig_sfreq)
    print("done.")

    print(f"  [3/6] Amplitude clipping (±150 µV)...", end=' ')
    data = clip_amplitude(data)
    print("done.")

    print(f"  [4/6] Movement artifact removal...")
    data = remove_movement_artifacts(data, orig_sfreq)

    print(f"  [5/6] Normalizing channels...", end=' ')
    data = normalize_channels(data)
    print("done.")

    print(f"  [6/6] Downsampling ({orig_sfreq} Hz → {target_sfreq} Hz)...", end=' ')
    data = downsample(data, orig_sfreq, target_sfreq)
    print(f"done. Output shape: {data.shape}")

    return data


if __name__ == "__main__":
    file_path = 'training/AD/3.npy'
    print(f"Loading {file_path}...")
    data = np.load(file_path)
    data_no_noise = denoise_eeg(data)