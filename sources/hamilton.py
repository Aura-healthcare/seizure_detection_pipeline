import numpy as np
from scipy.signal import find_peaks

def qrs_detector(signal: np.ndarray, freq_sampling: int) -> np.ndarray:
    peaks, _ = find_peaks(signal, freq_sampling=freq_sampling)
    return peaks
