import os
import argparse
from typing import Tuple, List
import sys

sys.path.append('.')

from src.domain.qrs_detector import QRSDetector
from src.infrastructure.edf_loader import EdfLoader
from src.usecase.utilities import convert_args_to_dict

OUTPUT_FOLDER = 'output/rr_intervals'
METHODS = ['hamilton', 'xqrs', 'gqrs', 'swt', 'engelsee']
DEFAULT_METHOD = 'hamilton'


def detect_qrs(filepath: str,
               method: str,
               exam_id: str,
               output_folder: str = OUTPUT_FOLDER) -> Tuple[int, str]:
    '''
    Detects QRS on a signal, and writes their frame and RR-intervals in a csv
    file.
    '''
    # Read ECG channel from EDF files
    edfloader = EdfLoader(filepath)
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

    # Export
    os.makedirs(output_folder, exist_ok=True)
    export_filename = f"{exam_id}.csv"
    export_filepath = os.path.join(output_folder, export_filename)
    df_detections.to_csv(export_filepath, sep=',', index=True)

    return sampling_frequency, export_filepath


def parse_detect_qrs_args(args_to_parse: List[str]) -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='CLI parameter input')
    parser.add_argument('--filepath',
                        dest='filepath',
                        required=True)
    parser.add_argument('--method',
                        dest='method',
                        required=True,
                        choices=METHODS)
    parser.add_argument('--exam-id',
                        dest='exam_id')
    parser.add_argument('--output-folder',
                        dest='output_folder')

    args = parser.parse_args(args_to_parse)

    return args


def parse_exam_id(filepath: str) -> str:

    exam_id = os.path.basename(filepath)

    return exam_id


if __name__ == "__main__":

    args = parse_detect_qrs_args(sys.argv[1:])
    args_dict = convert_args_to_dict(args)
    if 'exam_id' not in args_dict:
        exam_id = parse_exam_id(args_dict['filepath'])
        args_dict.update({'exam_id': exam_id})
    detect_qrs(**args_dict)
