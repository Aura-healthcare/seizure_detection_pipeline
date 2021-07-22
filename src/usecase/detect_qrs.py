from typing import List

import pandas as pd
from biosppy.signals.ecg import ecg


def detect_qrs(df_signal: pd.DataFrame,
               sampling_frequency: int) -> List(int):
    signal = list(df_signal['signal'])
    qrs_detections = ecg(
        signal=signal, sampling_rate=sampling_frequency, show=False
        )[2]
    return [int(element) for element in qrs_detections]
