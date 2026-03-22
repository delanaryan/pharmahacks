#proc.py
import numpy as np
import pandas as pd
from scipy.signal import welch, coherence, hilbert, butter, filtfilt
from scipy.stats import entropy as scipy_entropy


def filter_subjects(df: pd.DataFrame, drop_label: str):
    """
    Returns subject IDs relevant to the chosen classification task.
    e.g. drop_label='F' for AD vs CN task.
    """
    return df[df['label'] != drop_label]['anonymized_id']


def compute_rbp(epoch, sfreq, target_time_steps=30):
    """
    Relative Band Power across 5 frequency bands using Welch's method.
    AD shows increased delta/theta and decreased alpha/beta power.

    Args:
        epoch: (19, timepoints)
        sfreq: sampling frequency (128 Hz)
        target_time_steps: number of sub-windows (30)
    Returns:
        (30, 5, 19)
    """
    n_channels, n_points = epoch.shape
    segment_len = int(n_points / target_time_steps)
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 25), (25, 45)]
    rbp = np.zeros((target_time_steps, 5, n_channels))

    for t in range(target_time_steps):
        segment = epoch[:, t*segment_len:(t+1)*segment_len]
        freqs, psd = welch(segment, fs=sfreq, nperseg=segment_len, axis=1)
        total_power = np.sum(psd, axis=1, keepdims=True) + 1e-10

        for b_idx, (fmin, fmax) in enumerate(bands):
            idx = (freqs >= fmin) & (freqs <= fmax)
            band_power = np.sum(psd[:, idx], axis=1)
            rbp[t, b_idx, :] = band_power / total_power.flatten()

    return rbp  # (30, 5, 19)


def compute_scc(epoch, sfreq, target_time_steps=30):
    """
    Spectral Coherence Connectivity across 5 frequency bands.
    Measures inter-channel synchronization — disrupted in AD.

    Args:
        epoch: (19, timepoints)
        sfreq: sampling frequency (128 Hz)
        target_time_steps: number of sub-windows (30)
    Returns:
        (30, 5, 19)
    """
    n_channels, n_points = epoch.shape
    segment_len = int(n_points / target_time_steps)
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 25), (25, 45)]
    scc = np.zeros((target_time_steps, 5, n_channels))

    for t in range(target_time_steps):
        segment = epoch[:, t*segment_len:(t+1)*segment_len]
        freqs, _ = welch(segment, fs=sfreq, nperseg=segment_len, axis=1)

        coh_matrix = np.zeros((n_channels, n_channels, len(freqs)))
        for i in range(n_channels):
            for j in range(i, n_channels):
                _, Cxy = coherence(segment[i], segment[j],
                                   fs=sfreq, nperseg=segment_len)
                coh_matrix[i, j, :] = Cxy
                coh_matrix[j, i, :] = Cxy

        for b_idx, (fmin, fmax) in enumerate(bands):
            idx = (freqs >= fmin) & (freqs <= fmax)
            band_coh = np.sqrt(np.mean(coh_matrix[:, :, idx], axis=2))
            scc[t, b_idx, :] = np.mean(band_coh, axis=1)

    return scc  # (30, 5, 19)


def compute_hjorth(epoch, target_time_steps=30):
    """
    Hjorth parameters: Activity, Mobility, Complexity.
    AD brains show reduced signal complexity due to neuronal loss.

    Args:
        epoch: (19, timepoints)
        target_time_steps: number of sub-windows (30)
    Returns:
        (30, 3, 19)
    """
    n_channels, n_points = epoch.shape
    segment_len = n_points // target_time_steps
    hjorth = np.zeros((target_time_steps, 3, n_channels))

    for t in range(target_time_steps):
        seg = epoch[:, t*segment_len:(t+1)*segment_len]
        d1 = np.diff(seg, axis=1)
        d2 = np.diff(d1, axis=1)
        activity = np.var(seg, axis=1)
        mobility = np.sqrt(np.var(d1, axis=1) / (activity + 1e-10))
        mobility_d1 = np.sqrt(np.var(d2, axis=1) / (np.var(d1, axis=1) + 1e-10))
        complexity = mobility_d1 / (mobility + 1e-10)
        hjorth[t, 0, :] = activity
        hjorth[t, 1, :] = mobility
        hjorth[t, 2, :] = complexity

    return hjorth  # (30, 3, 19)


def compute_entropy(epoch, target_time_steps=30):
    """
    Shannon entropy per channel per window.
    AD is associated with reduced neural signal complexity.

    Args:
        epoch: (19, timepoints)
        target_time_steps: number of sub-windows (30)
    Returns:
        (30, 1, 19)
    """
    n_channels, n_points = epoch.shape
    segment_len = n_points // target_time_steps
    ent = np.zeros((target_time_steps, 1, n_channels))

    for t in range(target_time_steps):
        seg = epoch[:, t*segment_len:(t+1)*segment_len]
        for ch in range(n_channels):
            sig = np.abs(seg[ch])
            sig = sig / (sig.sum() + 1e-10)
            ent[t, 0, ch] = scipy_entropy(sig)

    return ent  # (30, 1, 19)


def compute_plv(epoch, target_time_steps=30):
    """
    Broadband Phase Locking Value per channel.
    Measures time-domain phase synchrony across all frequencies.

    Args:
        epoch: (19, timepoints)
        target_time_steps: number of sub-windows (30)
    Returns:
        (30, 1, 19)
    """
    n_channels, n_points = epoch.shape
    segment_len = n_points // target_time_steps
    analytic = hilbert(epoch, axis=1)
    phase = np.angle(analytic)
    plv = np.zeros((target_time_steps, 1, n_channels))

    for t in range(target_time_steps):
        phase_seg = phase[:, t*segment_len:(t+1)*segment_len]
        phase_diff = phase_seg[:, np.newaxis, :] - phase_seg[np.newaxis, :, :]
        plv_matrix = np.abs(np.mean(np.exp(1j * phase_diff), axis=2))
        plv[t, 0, :] = np.mean(plv_matrix, axis=1)

    return plv  # (30, 1, 19)


def compute_plv_per_band(epoch, sfreq, target_time_steps=30):
    """
    Band-specific PLV — filters into each band then computes PLV.
    Theta/alpha PLV is specifically disrupted in AD.

    Args:
        epoch: (19, timepoints)
        sfreq: sampling frequency (128 Hz)
        target_time_steps: number of sub-windows (30)
    Returns:
        (30, 5, 19)
    """
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 25), (25, 45)]
    n_channels, n_points = epoch.shape
    segment_len = n_points // target_time_steps
    plv = np.zeros((target_time_steps, 5, n_channels))

    for b_idx, (fmin, fmax) in enumerate(bands):
        nyq = sfreq / 2
        b, a = butter(4, [fmin/nyq, fmax/nyq], btype='band')
        filtered = filtfilt(b, a, epoch, axis=1)
        analytic = hilbert(filtered, axis=1)
        phase = np.angle(analytic)

        for t in range(target_time_steps):
            phase_seg = phase[:, t*segment_len:(t+1)*segment_len]
            phase_diff = phase_seg[:, np.newaxis, :] - phase_seg[np.newaxis, :, :]
            plv_matrix = np.abs(np.mean(np.exp(1j * phase_diff), axis=2))
            plv[t, b_idx, :] = np.mean(plv_matrix, axis=1)

    return plv  # (30, 5, 19)
