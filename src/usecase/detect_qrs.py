"""
This script is used to detect and export RR intervals.

Copyright (C) 2021 Association Aura
SPDX-License-Identifier: GPL-3.0
"""
import os
import argparse
from typing import List
import sys

sys.path.append('.')
from src.usecase.utilities import convert_args_to_dict, generate_output_path
from src.domain.qrs_detector import QRSDetector
from src.infrastructure.edf_loader import EdfLoader

OUTPUT_FOLDER = 'output/rr_intervals'
METHODS = ['hamilton', 'xqrs', 'gqrs', 'swt', 'engelsee']
DEFAULT_METHOD = 'hamilton'


def detect_qrs(qrs_file_path: str,
               method: str,
               exam_id: str,
               output_folder: str = OUTPUT_FOLDER) -> str:
    """
    Detect QRS on a, ECG signal signal.

    From an EDF file path, detects QRS abd and writes their frame and
    RR-intervals in a csv file.

    parameters
    ----------
    qrs_file_path : str
        The path of the file including ECG signal
    method : str
        QRS detection method, to be chose between hamilton, xqrs, gqrs, swt
        and engelsee
    exam_id : str
        ID of the exam to load
    output_folder : str
        Path of the output folder

    returns
    -------
    output_file_path : str
        Path where detected qrs are stored in csv format
    """
    # Reads ECG channel from EDF files
    edfloader = EdfLoader(qrs_file_path)
    ecg_channel_name = edfloader.get_ecg_candidate_channel()
    start_time, end_time = edfloader.get_edf_file_interval()

    try:
        sampling_frequency, ecg_data = edfloader.ecg_channel_read(
            ecg_channel_name,
            start_time,
            end_time)
    except ValueError:
        print(f'There is no ECG channel in {qrs_file_path}, exiting')
        sys.exit()

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
    output_file_path = generate_output_path(
        input_file_path=exam_id,
        output_folder=output_folder,
        format='csv')

    df_detections.to_csv(output_file_path, sep=',', index=True)

    return output_file_path, sampling_frequency


def parse_detect_qrs_args(args_to_parse: List[str]) -> argparse.Namespace:
    """
    Parse arguments for adaptable input.

    parameters
    ----------
    args_to_parse : List[str]
        List of the element to parse. Should be sys.argv[1:] if args are
        inputed via CLI

    returns
    -------
    args : argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='CLI parameter input')
    parser.add_argument('--qrs-file-path',
                        dest='qrs_file_path',
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


def parse_exam_id(qrs_file_path: str) -> str:
    """
    From a qrs_file_path including an exam ID, return the ID directly.

    parameters
    ----------
    qrs_file_path : str
        The path to the qrs file

    returns
    -------
    exam_id : str
        The parsed exam id
    """
    exam_id = os.path.basename(qrs_file_path)

    return exam_id


if __name__ == "__main__":

    args = parse_detect_qrs_args(sys.argv[1:])
    args_dict = convert_args_to_dict(args)
    if 'exam_id' not in args_dict:
        exam_id = parse_exam_id(args_dict['qrs_file_path'])
        args_dict.update({'exam_id': exam_id})
    detect_qrs(**args_dict)
