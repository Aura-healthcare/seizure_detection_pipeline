import numpy as np
from ecgdetectors import Detectors

def qrs_detector(signal: np.ndarray, freq_sampling: int) -> np.ndarray:
    return np.array(Detectors(freq_sampling).hamilton_detector(signal))
