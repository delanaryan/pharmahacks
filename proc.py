from scipy import signal 
import numpy as np
import pandas as pd
from scipy.signal import welch
import pywt
import torch

def filter_subjects(df: pd.DataFrame, drop_label: str):
    '''
    Returns the subjects array relevant to the task chosen.

    Args:
      df: the dataframe that maps each subject to its corresponding label (A, C, or F)
      drop_label: the label that is not relevant to the task chosen. Example: if i'm doing A vs C classification, then the drop label would be 'F'
    '''
    return df[df['label'] != drop_label]['anonymized_id']

def extract_features(eeg_data, sfreq, target_time_steps=30):
    """
    Extracts RBP (Welch) and SCC (PyWavelets CWT).

    Args:
        eeg_data: (Channels, Time) - A single 30-second epoch.
        sfreq: Sampling frequency (e.g., 128).
        target_time_steps: Number of time windows (30 for DICE-net).
    Returns:
        rbp: (30, 5, 19)
        scc: (30, 5, 19)
    """

    n_channels, n_points = eeg_data.shape
    segment_len = int(n_points / target_time_steps)

    # ---------------------------------------------------------
    # 1. Relative Band Power (RBP)
    # ---------------------------------------------------------
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 25), (25, 45)]
    rbp_features = np.zeros((target_time_steps, 5, n_channels))

    for t in range(target_time_steps):
        start = t * segment_len
        end = start + segment_len
        segment = eeg_data[:, start:end]

        #========================================#
        # Notes on Welch's Method:
        # - welch returns frequencies and their corresponding power spectral density (PSD).
        # - nperseg controls the length of each segment for FFT. A common choice is
        #   256 or 512 samples, but it can be adjusted based on the segment length and desired frequency resolution.
        # - The axis=1 argument computes the PSD for each channel separately.
        #========================================#
        freqs, psd = welch(segment, fs=sfreq, nperseg=segment_len, axis=1) 

        total_power = np.sum(psd, axis=1, keepdims=True)
        total_power[total_power == 0] = 1e-10 # If total power is zero, set to small value to avoid division by zero

        #=========================================#
        # Notes on RBP Calculation:
        # - For each frequency band, we identify the indices of the frequencies that fall within that
        #   band.
        # - We sum the PSD values for those frequencies to get the band power.
        # - Finally, we normalize by the total power to get the relative band power.
        #=========================================#

        for b_idx, (fmin, fmax) in enumerate(bands): 
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            band_power = np.sum(psd[:, idx], axis=1)
            rbp_features[t, b_idx, :] = band_power / total_power.flatten()

    # ---------------------------------------------------------
    # 2. Spectral Coherence Connectivity (SCC) - Using PyWavelets
    # ---------------------------------------------------------
    morlet_freqs = np.array([2, 6, 10, 18, 35])
    wavelet_name = 'cmor1.5-1.0'  # Complex Morlet (bandwidth 1.5, center freq 1.0)

    # Convert target frequencies (Hz) to Wavelet Scales
    # Scale = (Center_Freq * Sampling_Rate) / Target_Freq
    center_freq = pywt.central_frequency(wavelet_name)
    scales = (center_freq * sfreq) / morlet_freqs

    # Pre-allocate coefficients storage: (Channels, Bands, Time)
    coeffs_all = np.zeros((n_channels, len(morlet_freqs), n_points), dtype=np.complex128)

    # Compute CWT for each channel
    # pywt.cwt returns (coeffs, freqs) where coeffs is (len(scales), len(data))
    for ch in range(n_channels):
        cwt_out, _ = pywt.cwt(eeg_data[ch], scales, wavelet_name, sampling_period=1/sfreq)
        coeffs_all[ch, :, :] = cwt_out

    # Calculate Coherence for each 1-second window
    scc_features = np.zeros((target_time_steps, 5, n_channels))

    for t in range(target_time_steps):
        start = t * segment_len
        end = start + segment_len

        for b_idx in range(len(morlet_freqs)):
            # Get coefficients for this specific band and time window
            # Shape: (n_channels, n_samples_in_window)
            seg_coeffs = coeffs_all[:, b_idx, start:end]

            # --- Vectorized Coherence Calculation ---

            # 1. Cross-Spectral Density Matrix (CSD): X * Y_conjugate
            # Result is (19, 19) matrix of summed cross-products
            csd_matrix = seg_coeffs @ seg_coeffs.conj().T

            # 2. Power Spectral Density (Diagonal of CSD)
            psd_vec = np.diag(csd_matrix).real

            # 3. Denominator: sqrt(PSD_x * PSD_y)
            denom = np.sqrt(np.outer(psd_vec, psd_vec))
            denom[denom == 0] = 1e-10 # Safe division

            # 4. Coherence: |CSD| / Denominator
            coherence_matrix = np.abs(csd_matrix) / denom

            # 5. Average Coherence (SCC) per channel
            # "SCC involves calculating spectral coherence... and averaging these values for each electrode."
            scc_val = np.mean(coherence_matrix, axis=1)

            scc_features[t, b_idx, :] = scc_val

    return rbp_features, scc_features

def generate_dataset(data_list: list, labels: list, subject_ids: list, sfreq: int = 128) -> tuple:
    """
    Args:
        data_list: List of numpy arrays, each shape (channels, time)
        labels: List or array of group labels
        subject_ids: List or array of subject identifiers. In your case, this should just be the first column of the label mapping CSV file.
        sfreq: Sampling frequency
    """
    X_rbp_list = [] # We will store the RBP features for each epoch here. Each entry will be of shape (30, 5, 19) corresponding to (time windows, frequency bands, channels).
    X_scc_list = [] # We will store the SCC features for each epoch here. Each entry will also be of shape (30, 5, 19) corresponding to (time windows, frequency bands, channels).
    y_list = [] # We will store the labels for each epoch here. Each entry will be a scalar (0 or 1) corresponding to the group label for that epoch.
    groups_list = [] # We will store the subject IDs for each epoch here. Each entry will be a scalar corresponding to the subject ID for that epoch.

    window_samples = 30 * sfreq # Window samples are 30 seconds * sampling frequency, they define the length of each epoch we will extract features from.
    step_samples = 15 * sfreq # Step size for sliding window (15 seconds * sampling frequency). This means we will have 50% overlap between consecutive windows.

    for data, label, s_id in zip(data_list, labels, subject_ids):

        n_points = data.shape[1] # Total number of time points in the EEG recording for this subject. This is used to determine how many epochs we can extract using the sliding window approach.

        # Sliding Window
        for start in range(0, n_points - window_samples + 1, step_samples):
            epoch = data[:, start : start + window_samples] # Extract a 30-second epoch of shape (channels, window_samples) from the EEG data using the current start index and the defined window length.

            rbp, scc = extract_features(epoch, sfreq) # Extract RBP and SCC features for this epoch. rbp and scc will both have shape (30, 5, 19) corresponding to (time windows, frequency bands, channels).

            X_rbp_list.append(rbp) # Append the extracted RBP features for this epoch to the X_rbp_list. Each entry in this list will be a numpy array of shape (30, 5, 19).
            X_scc_list.append(scc)
            y_list.append(label) 
            groups_list.append(s_id) # Append the subject ID for this epoch to the groups_list. Each entry in this list will be a scalar corresponding to the subject ID for that epoch.

    #=========================================#
    # Notes on Pytorch Tensor Conversion:
    # - We first convert the list of feature arrays into a single numpy array using np.array
    #   which will have shape (num_epochs, 30, 5, 19).
    # - Then we convert this numpy array into a PyTorch tensor of type float32 using
    #   torch.tensor(..., dtype=torch.float32).
    # - The resulting tensors X_rbp and X_scc will have shape (num_epochs
    #   30, 5, 19) and the label tensor y will have shape (num_epochs, 1) after unsqueezing.
    #=========================================#
    X_rbp = torch.tensor(np.array(X_rbp_list), dtype=torch.float32) # Convert the list of RBP features into a single numpy array and then into a PyTorch tensor of type float32. The resulting shape of X_rbp will be (num_epochs, 30, 5, 19) where num_epochs is the total number of epochs extracted across all subjects.
    X_scc = torch.tensor(np.array(X_scc_list), dtype=torch.float32)
    y = torch.tensor(np.array(y_list), dtype=torch.float32).unsqueeze(1)
    groups = np.array(groups_list)

    return X_rbp, X_scc, y, groups