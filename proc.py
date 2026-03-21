from scipy import signal 
import numpy as np
import pandas as pd
from scipy.signal import welch, coherence
import pywt
import torch

#TODO: Denoise function

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
    ULTRA-FAST VERSION: Uses Welch's method for both RBP and SCC.
    No more Wavelets = No more infinite loops.
    """
    n_channels, n_points = eeg_data.shape
    segment_len = int(n_points / target_time_steps) # Usually 128 samples (1 sec)
    
    # Define the 5 standard frequency bands
    bands = [(0.5, 4), (4, 8), (8, 13), (13, 25), (25, 45)]
    
    # Pre-allocate output arrays: (30 segments, 5 bands, 19 channels)
    rbp_features = np.zeros((target_time_steps, 5, n_channels))
    scc_features = np.zeros((target_time_steps, 5, n_channels))

    for t in range(target_time_steps):
        # 1. Slice the 1-second segment
        start = t * segment_len
        end = start + segment_len
        segment = eeg_data[:, start:end]

        # --- PART 1: RELATIVE BAND POWER (RBP) ---
        # Compute Power Spectral Density (PSD) for all 19 channels at once
        freqs, psd = welch(segment, fs=sfreq, nperseg=segment_len, axis=1)
        total_power = np.sum(psd, axis=1, keepdims=True) + 1e-10 # Avoid div by zero

        # --- PART 2: SPECTRAL COHERENCE CONNECTIVITY (SCC) ---
        # We build a matrix of coherence for every channel pair (19x19)
        # Note: SCC_x is the average coherence of channel X with all other channels Y
        coh_matrix_all_freqs = np.zeros((n_channels, n_channels, len(freqs)))
        
        for i in range(n_channels):
            for j in range(i, n_channels): # Only upper triangle (it's symmetric!)
                _, Cxy = coherence(segment[i], segment[j], fs=sfreq, nperseg=segment_len)
                coh_matrix_all_freqs[i, j, :] = Cxy
                coh_matrix_all_freqs[j, i, :] = Cxy

        # --- PART 3: BAND-SPECIFIC AGGREGATION ---
        for b_idx, (fmin, fmax) in enumerate(bands):
            # Find frequencies belonging to this band
            idx = np.logical_and(freqs >= fmin, freqs <= fmax)
            
            # A. RBP: Sum power in band / total power
            band_power = np.sum(psd[:, idx], axis=1)
            rbp_features[t, b_idx, :] = band_power / total_power.flatten()

            # B. SCC: Implement the equation SCC_x = 1/C * sum(sqrt(Cxy))
            # The challenge formula uses sqrt of coherence (|Sxy|/sqrt(Sxx*Syy))
            band_coh_matrix = np.sqrt(np.mean(coh_matrix_all_freqs[:, :, idx], axis=2))
            
            # Average across the channel dimension (1/C * sum)
            scc_features[t, b_idx, :] = np.mean(band_coh_matrix, axis=1)

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

    # Since we are using an SVM model, we need to flatten the 4D feature tensors (num_epochs, 30, 5, 19) into 2D tensors of shape (num_epochs, 30*5*19) so that each epoch is represented as a single feature vector.
    X_combined = torch.cat((X_rbp, X_scc), dim=2) # Concatenate RBP and SCC features along the feature dimension (dim=1). This will give us a combined feature tensor of shape (num_epochs, 30, 5, 19) since we are concatenating along the time window dimension.
    X_combined = X_combined.view(X_combined.shape[0], -1) # Flatten the combined feature tensor into shape (num_epochs, 30*5*19) so that each epoch is represented as a single feature vector.

    return X_combined, y, groups