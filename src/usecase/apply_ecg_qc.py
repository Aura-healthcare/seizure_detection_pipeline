"""
This script is used to consolidate hrv features dataset and annotations.

Copyright (C) 2022 Association Aura
SPDX-License-Identifier: GPL-3.0
"""
import os
from typing import Tuple, List, Union
import argparse
import ecg_qc
import pandas as pd
import fileinput

import sys
sys.path.append('.')

from src.infrastructure.edf_loader import EdfLoader
from src.usecase.utilities import convert_args_to_dict, generate_output_path

MODEL_FOLDER = 'models'
MODELS = os.listdir(MODEL_FOLDER)
OUTPUT_FOLDER = 'output/quality'

ECG_QC_MODEL = 'rfc_normalized_2s.pkl'


def parse_model(model: str) -> Tuple[str, str, int, bool]:
    """Parse ecg_qc model name to output parameters.

    Parameters
    ----------
    model : str
        Name of the model = name of the file

    Returns
    -------
    model_path : str
        Path to the model
    model_name : str
        Name of the model
    length_chunk : int
        Length of signal to apply the model on. It's an EcgQc parameter.
    is_normalized : bool
        Does the data have to be normalized ? It's an EcgQc parameter.
    """
    model_path = os.path.join(MODEL_FOLDER, model)
    model_name = model.split('.')[0]
    model_split = model_name.split('_')
    try:
        length_chunk = int(model_split[-1][:-1])
    except ValueError:
        length_chunk = 9
    is_normalized = 'normalized' in model_split

    return model_path, model_name, length_chunk, is_normalized


def initialize_df_body(signal_quality: List[float],
                       length_chunk: int,
                       start_time: Union[pd.Timestamp, int],
                       end_time: Union[pd.Timestamp, int],
                       formatting: str) -> pd.DataFrame:
    """Initialize the df_body before further operations.

    Parameters
    ----------
    signal_quality : List[float]
        The ECG signal quality, composed of 1 (clean) or 0 (noise)
    length_chunk : int
        Lenght of signal to apply the model on. It's an EcgQc parameter.
    start_time : pd.Timestamp | int
        Start time of the ECG
    end_time: pd.Timestamp | int
        End time of the ECG
    formatting: str
        Formatting of the tse_bi output ('teppe' or 'tuh')

    Returns
    -------
    df_body : pd.DataFrame
        The initialize df_body
    """
    if formatting == 'dataset':
        start_times = [start_time + pd.Timedelta(
                       seconds=length_chunk * iteration)
                       for iteration, _ in enumerate(signal_quality)]
    else:
        start_time = 0
        end_time = start_time + length_chunk * len(signal_quality)
        start_times = [start_time + length_chunk * iteration
                       for iteration, _ in enumerate(signal_quality)]

    return pd.DataFrame(data={"start": start_times,
                              "end": None,
                              "annotation": signal_quality,
                              "probability": 1}), \
        end_time


def consolidate_df_body_annotation(df_body: pd.DataFrame,
                                   end_time: Union[pd.Timestamp, int]):
    """Consolidated same annotations following each others for clarity.

    Parameters
    ----------
    df_body : pd.DataFrame
        The df_body to consolidate
    end_time: pd.Timestamp | int
        End time of the ECG

    Returns
    -------
    df_body : pd.DataFrame
        The initialize df_body
    """
    for i in df_body.index.values[1:]:
        if df_body.loc[i, 'annotation'] == df_body.loc[i - 1, 'annotation']:
            df_body.loc[i, 'end'] = 'del'
    df_body = df_body[df_body['end'] != 'del']
    df_body.reset_index(drop=True, inplace=True)

    end_times = list(df_body['start'].values[1:])
    end_times.append(end_time)

    df_body = df_body.assign(end=end_times)

    return df_body


def create_ecg_qc_df_body(signal_quality: List[float],
                          length_chunk: int,
                          start_time: Union[pd.Timestamp, int],
                          end_time: Union[pd.Timestamp, int],
                          formatting: str) -> pd.DataFrame:
    """Create the body of tse_bi, incluing signal timestamps and quality.

    Parameters
    ----------
    signal_quality : List[float]
        The ECG signal quality, composed of 1 (clean) or 0 (noise)
    length_chunk : int
        Lenght of signal to apply the model on. It's an EcgQc parameter.
    start_time : pd.Timestamp | int
        Start time of the ECG
    end_time: pd.Timestamp | int
        End time of the ECG
    formatting: str
        Formatting of the tse_bi output ('teppe' or 'tuh')

    Returns
    -------
    df_body : pd.DataFrame
        The initialize df_body
    """
    df_body, end_time = initialize_df_body(signal_quality=signal_quality,
                                           length_chunk=length_chunk,
                                           start_time=start_time,
                                           end_time=end_time,
                                           formatting=formatting)
    df_body = consolidate_df_body_annotation(df_body=df_body,
                                             end_time=end_time)
    if formatting == 'dataset':
        df_body['start'] = df_body['start'].apply(
            lambda x: pd.Timestamp.isoformat(x))
        df_body['end'] = df_body['end'].apply(
            lambda x: pd.Timestamp.isoformat(x))

    return df_body


def create_ecg_qc_df_header(formatting: str) -> pd.DataFrame:
    """Create the header of the tse_bi file.

    Parameters
    ----------
    formatting : str
        Formatting of the tse_bi: dataset or tuh

    Returns
    -------
    df_header : pd.DataFrame
        DataFrame including header for tse_bi formatting
    """
    df_header = pd.DataFrame(columns=['start',
                                      'end',
                                      'annotation',
                                      'probability'])
    df_header = df_header.append({'start': 'version = tse_v1.0.0',
                                  'end': '',
                                  'annotation': '',
                                  'probability': ''},
                                 ignore_index=True)

    if formatting == 'dataset':
        df_header = df_header.append({'start': 'annotator = ecg_qc',
                                      'end': '',
                                      'annotation': '',
                                      'probability': ''},
                                     ignore_index=True)
        df_header = df_header.append({'start': ' '.join(['date =',
                                                         str(pd.Timestamp.
                                                             today())]),
                                      'end': '',
                                      'annotation': '',
                                      'probability': ''},
                                     ignore_index=True)
    df_header = df_header.append({'start': '',
                                  'end': '',
                                  'annotation': '',
                                  'probability': ''},
                                 ignore_index=True)

    return df_header


def create_ecg_qc_tse_bi(signal_quality: List[float],
                         length_chunk: int,
                         start_time: Union[pd.Timestamp, int],
                         end_time: Union[pd.Timestamp, int],
                         formatting: str) -> pd.DataFrame:
    """Create a pd.DataFrame exportable to tse_bi format.

    Parameters
    ----------
    signal_quality : List[float]
        The ECG signal quality, composed of 1 (clean) or 0 (noise)
    length_chunk : int
        Lenght of signal to apply the model on. It's an EcgQc parameter.
    start_time : pd.Timestamp | int
        Start time of the ECG
    end_time: pd.Timestamp | int
        End time of the ECG
    formatting: str
        Formatting of the tse_bi output ('teppe' or 'tuh')

    Returns
    -------
    df_ecg_qc : pd.DataFrame
        DataFrame of signal quality in tse_bi format
    """
    df_body = create_ecg_qc_df_body(signal_quality=signal_quality,
                                    length_chunk=length_chunk,
                                    start_time=start_time,
                                    end_time=end_time,
                                    formatting=formatting)
    df_header = create_ecg_qc_df_header(formatting='dataset')

    return pd.concat([df_header, df_body])


def import_signal_data(qrs_file_path: str) -> Tuple[List[float],
                                                    pd.Timestamp,
                                                    pd.Timestamp,
                                                    int]:
    """Import relevant information from the signal file.

    parameters
    ----------
    qrs_file_path : str
        The path of the file including ECG signal

    returns
    -------
    signal : List[float]
        Signal as a list of values
    start_time : pd.Timestamp
        Start time of the file
    end_time : pd.Timestamp
        End time of the file
    sampling_frequency : int
        Frequency of the read signal
    """
    edfloader = EdfLoader(qrs_file_path)
    ecg_channel_name = edfloader.get_ecg_candidate_channel()
    start_time, end_time = edfloader.get_edf_file_interval()

    try:
        sampling_frequency, ecg_data = edfloader.ecg_channel_read(
            ecg_channel_name, start_time, end_time)

    except ValueError as e:
        raise ValueError(f'There is no ECG channel in {qrs_file_path}') from e

    signal = list(ecg_data['signal'])

    return signal, start_time, end_time, sampling_frequency


def get_signal_quality(signal: List[float],
                       model: str,
                       sampling_frequency: int) -> List[str]:
    """Get the quality of a signal thanks to an ecg_qc model.

    Parameters
    ----------
    signal : List[float]
        The ECG signal to use for quality detection
    model : str
        The model to use to predicty quality
    sampling_frequency : int
        Frequency of the read signal

    Returns
    -------
    signal_quality_label : list[str]
        A list of qualities by length chunk, either 'clean' or 'noisy'
    length_chunk : int
        Length of signal to apply the model on. It's an EcgQc parameter.
    """
    model_path, _, length_chunk, is_normalized = parse_model(model)
    algo = ecg_qc.EcgQc(model=model_path,
                        sampling_frequency=sampling_frequency,
                        normalized=is_normalized)

    n = length_chunk * sampling_frequency
    signal_subdiv = [signal[i * n:(i + 1) * n]
                     for i in range((len(signal) + n - 1) // n)]

    # Padding on last chunk if necessary
    m = len(signal_subdiv[-1])
    if m < n:
        signal_subdiv[-1] += [0 for _ in range(n - m)]
    signal_qualities = [algo.get_signal_quality(signal)
                        for signal in signal_subdiv]

    return ["clean" if signal_quality == 1 else "noisy"
            for signal_quality in signal_qualities], length_chunk


def apply_ecg_qc(qrs_file_path: str,
                 formatting: str,
                 output_folder: str = OUTPUT_FOLDER,
                 model: str = ECG_QC_MODEL) -> str:
    """Apply ecg_qc on an ECG signal to compute quality.

    parameters
    ----------
    qrs_file_path : str
        The path of the file including ECG signal
    formatting: str
        Formatting of the tse_bi output ('teppe' or 'tuh')
    output_folder : str
        Path of the output folder
    model : str
        Name of the model = name of the file

    returns
    -------
    output_file_path : str
        Path where detected qrs are stored in csv format
    Applies an ECG QC model on a signal, and writes results in a json file.
    """
    signal, start_time, end_time, sampling_frequency = import_signal_data(
        qrs_file_path=qrs_file_path)

    signal_quality_label, length_chunk = get_signal_quality(
        signal=signal,
        model=model,
        sampling_frequency=sampling_frequency)

    df_ecg_qc = create_ecg_qc_tse_bi(
        signal_quality=signal_quality_label,
        length_chunk=length_chunk,
        start_time=start_time,
        end_time=end_time,
        formatting=formatting)

    output_file_path = generate_output_path(input_file_path=qrs_file_path,
                                            output_folder=output_folder,
                                            format="tse_bi",
                                            prefix="ecg_qc")
    df_ecg_qc.to_csv(output_file_path, sep=" ", index=False, header=False)

    # Format properly to tse_bi standard
    with fileinput.FileInput(output_file_path, inplace=True) as tse_bi_file:
        for line in tse_bi_file:
            print(line.replace('"', ""), end='')

    return output_file_path


def parse_apply_ecg_qc_args(args_to_parse: List[str]) -> argparse.Namespace:
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
    parser.add_argument('--output-folder',
                        dest='output_folder')
    parser.add_argument('--model',
                        dest='model',
                        choices=MODELS)
    parser.add_argument('--formatting',
                        dest='formatting',
                        choices=['dataset', 'tuh'],
                        default='dataset')
    return parser.parse_args(args_to_parse)


if __name__ == "__main__":

    args = parse_apply_ecg_qc_args(sys.argv[1:])
    args_dict = convert_args_to_dict(args)
    apply_ecg_qc(**args_dict)
