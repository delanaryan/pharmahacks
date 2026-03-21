from scipy import signal 
import numpy as np

def SCC(epoch: np.ndarray, sfreq: float) -> np.ndarray:
    """
    Compute the SCC (Spectral Coherence Coefficient) between two signals.
    """

    n_channels = epoch.shape[1]
    scc_values = np.zeros(n_channels)

    for x in range(n_channels):
        coherence_sum = 0

        for y in range(n_channels):
            # signal.coherence computes |Sxy|^2 / (Sxx * Syy)
            # The challenge formula asks for |Sxy| / sqrt(Sxx * Syy)
            # which is the square root of the standard coherence output.
            freqs, coh = signal.coherence(epoch[:, x], epoch[:, y], fs=sfreq)
            
            # Take the mean across the frequencies of interest (e.g., Alpha 8-13Hz)
            # and take the square root to match the formula's Sxx*Syy square root
            coherence_sum += np.sqrt(np.mean(coh)) 
            
        # Average over all channels (C) as per the 1/C in the formula
        scc_values[x] = coherence_sum / n_channels

    return scc_values
