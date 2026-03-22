import numpy as np
import matplotlib.pyplot as plt
from denoise import denoise_eeg
from proc import extract_features
import os
import seaborn as sns

# Load samples (replace with your correctly identified subjects)
ad_file = 'training/AD/3.npy'  # Alzheimer's case
cn_file = 'training/CN/10.npy'  # Healthy case

ad_raw = np.load(ad_file)
cn_raw = np.load(cn_file)
ad_denoised = denoise_eeg(ad_raw)
cn_denoised = denoise_eeg(cn_raw)

# Extract features from first 30s epoch
ad_epoch = ad_denoised[:, :30*128]
cn_epoch = cn_denoised[:, :30*128]
ad_rbp, ad_scc = extract_features(ad_epoch, 128)
cn_rbp, cn_scc = extract_features(cn_epoch, 128)

# Helper function to plot time series
def plot_eeg(ax, data, title, label):
    for i in range(3):
        ax.plot(data[i, :10*128], label=f'Channel {i+1}', alpha=0.8)
    ax.set_title(title)
    ax.set_xlabel('Time (samples at 128Hz)')
    ax.set_ylabel('Amplitude (µV)')
    ax.legend()
    ax.grid(True, alpha=0.3)

# Helper function to plot PSD
def plot_psd(ax, data, title):
    from scipy.signal import welch
    freqs, psd = welch(data[0, :], fs=128, nperseg=256)
    ax.semilogy(freqs, psd, color='black', linewidth=1.5)
    ax.set_title(title)
    ax.set_xlabel('Frequency (Hz)')
    ax.set_ylabel('Power (V²/Hz, log scale)')
    bands = [
        (0.5, 4, 'red', 'Delta'),
        (4, 8, 'blue', 'Theta'),
        (8, 13, 'green', 'Alpha'),
        (13, 25, 'orange', 'Beta'),
        (25, 45, 'purple', 'Gamma')
    ]
    for fmin, fmax, color, label in bands:
        ax.axvspan(fmin, fmax, color=color, alpha=0.1)
        ax.text((fmin + fmax)/2, max(psd)*0.1, label, ha='center', va='bottom', fontsize=8, color=color)
    ax.legend([f'PSD'], loc='upper right')
    ax.grid(True, alpha=0.3)

# Plot setup: 2 rows, 2 columns (CN vs AD comparison)
plt.style.use('seaborn-v0_8')
fig, axes = plt.subplots(2, 2, figsize=(16, 10), dpi=150)
fig.suptitle('EEG Comparison: Healthy (CN) vs. Alzheimer\'s (AD) Cases\nVisualizing Differences in Brain Wave Patterns for Classification', fontsize=18, fontweight='bold', y=0.98)

# Top row: Raw EEG comparison
plot_eeg(axes[0,0], cn_raw, 'CN: Raw EEG (Healthy)\nUnprocessed brain waves', 'CN Raw')
plot_eeg(axes[0,1], ad_raw, 'AD: Raw EEG (Alzheimer\'s)\nUnprocessed brain waves', 'AD Raw')

# Bottom row: PSD comparison
plot_psd(axes[1,0], cn_denoised, 'CN: PSD (Healthy)\nEnergy distribution across frequencies')
plot_psd(axes[1,1], ad_denoised, 'AD: PSD (Alzheimer\'s)\nEnergy distribution across frequencies')

# Add RBP comparison as text or small plots if needed
# For now, print key differences
cn_avg_rbp = np.mean(cn_rbp, axis=0)
ad_avg_rbp = np.mean(ad_rbp, axis=0)
print("CN Avg RBP (Delta-Theta-Alpha-Beta-Gamma):", np.mean(cn_avg_rbp, axis=1))
print("AD Avg RBP (Delta-Theta-Alpha-Beta-Gamma):", np.mean(ad_avg_rbp, axis=1))

plt.tight_layout()
plt.savefig('hackathon_comparison.png', bbox_inches='tight', facecolor='white', dpi=300)
plt.show()