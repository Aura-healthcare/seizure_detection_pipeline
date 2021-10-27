import os
import argparse
from typing import Tuple
import sys

import pandas as pd
sys.path.append('.')

from src.domain.qrs_detector import QRSDetector
from src.infrastructure.edf_loader import EdfLoader


OUTPUT_FOLDER = 'output/rr_intervals'
METHODS = ['hamilton', 'xqrs', 'gqrs', 'swt', 'engelsee']
DEFAULT_METHOD = 'hamilton'


def write_detections_csv(detections: pd.DataFrame,
                         exam_id: str) -> str:

    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    filename = f"{exam_id}.csv"
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    detections.to_csv(filepath, sep=',', index=True)

    return filename


def detect_qrs(filename: str,
               method: str,
               exam_id: str) -> Tuple[int, str]:
    '''
    Detects QRS on a signal, and writes their frame and RR-intervals in a csv
    file.
    '''
    # Read ECG channel from EDF files
    edfloader = EdfLoader(filename)
    ecg_channel_name = edfloader.get_ecg_candidate_channel()
    start_time, end_time = edfloader.get_edf_file_interval()

    sampling_frequency, ecg_data = edfloader.ecg_channel_read(
        ecg_channel_name,
        start_time,
        end_time)

    qrs_detector = QRSDetector()
    signal = list(ecg_data['signal'])
    detected_qrs, rr_intervals = qrs_detector.get_cardiac_infos(
        signal, sampling_frequency, method)
    df_detections = ecg_data.copy()
    df_detections = df_detections.iloc[detected_qrs[:-1]]
    df_detections['frame'] = detected_qrs[:-1]
    df_detections['rr_interval'] = rr_intervals
    df_detections.drop(columns='signal', inplace=True)
    filename = write_detections_csv(df_detections, exam_id)

    return sampling_frequency, filename


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='CLI parameter input')
    parser.add_argument('--filename',
                        dest='filename',
                        required=True)
    parser.add_argument('--method',
                        dest='method',
                        required=True,
                        choices=METHODS)
    parser.add_argument('--exam-id',
                        dest='exam_id',
                        required=True)
    args = parser.parse_args()
    detect_qrs(args.filename,
               args.method,
               args.exam_id)
