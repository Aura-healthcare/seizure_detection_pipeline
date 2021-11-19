import os
import json
from typing import Tuple, List
import argparse
import ecg_qc
import click

import sys
sys.path.append('.')

from src.infrastructure.edf_loader import EdfLoader
from src.usecase.utilities import convert_args_to_dict

MODEL_FOLDER = 'models'
MODELS = os.listdir(MODEL_FOLDER)
OUTPUT_FOLDER = 'output/quality'

ECG_QC_MODEL = 'rfc_normalized_2s.pkl'
SAMPLING_FREQUENCY = 1000

def parse_model(model: str) -> Tuple[str, str, int, bool]:
    model_path = os.path.join(MODEL_FOLDER, model)
    model_name = model.split('.')[0]
    model_split = model_name.split('_')
    try:
        length_chunk = int(model_split[-1][:-1])
    except ValueError:
        length_chunk = 9
    is_normalized = 'normalized' in model_split
    return model_path, model_name, length_chunk, is_normalized


def write_quality_json(quality: List[int],
                       exam_id: str,
                       model_name: str,
                       output_foder: str) -> str:

    os.makedirs(output_foder, exist_ok=True)
    filename = f"{exam_id}_{model_name}.csv"
    export_filepath = os.path.join(output_foder, filename)
    with open(export_filepath, 'w') \
            as outfile:
        json.dump(quality, outfile)

    return str(export_filepath)


def apply_ecg_qc(filepath: str,
                 exam_id: str,
                 output_folder: str = OUTPUT_FOLDER,
                 sampling_frequency: int = SAMPLING_FREQUENCY,
                 model: str = ECG_QC_MODEL) -> str:
    '''
    Applies an ECG QC model on a signal, and writes results in a json file.
    '''
    edfloader = EdfLoader(filepath)
    ecg_channel_name = edfloader.get_ecg_candidate_channel()
    start_time, end_time = edfloader.get_edf_file_interval()

    sampling_frequency, ecg_data = edfloader.ecg_channel_read(
        ecg_channel_name,
        start_time,
        end_time)
    signal = list(ecg_data['signal'])

    model_path, model_name, length_chunk, is_normalized = parse_model(model)
    algo = ecg_qc.EcgQc()#sampling_frequency=sampling_frequency,
                        #model=model,
                        # normalized=True)
                        #model_path,
                        # normalized=is_normalized)

    # Preprocess signal : chunks of length_chunk seconds
    n = length_chunk * sampling_frequency
    signal_subdiv = [signal[i * n:(i + 1) * n]
                     for i in range((len(signal) + n - 1) // n)]
    # Padding on last chunk if necessary
    m = len(signal_subdiv[-1])
    if m < n:
        signal_subdiv[-1] += [0 for j in range(n - m)]
    # Apply ecg_qc on each chunk
    signal_quality = [algo.get_signal_quality(x) for x in signal_subdiv]


    export_filepath = write_quality_json(signal_quality,
                                         output_folder,
                                         exam_id,
                                         model_name)

    return export_filepath


def parse_apply_ecg_qc_args(
        args_to_parse: List[str]) -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='CLI parameter input')
    parser.add_argument('--filepath',
                        dest='filepath',
                        required=True)
    parser.add_argument('--exam-id',
                        dest='exam_id',
                        required=True)
    parser.add_argument('--output-folder',
                        dest='output_folder')
    parser.add_argument('--sampling-frequency',
                        dest='sampling_frequency')
    parser.add_argument('--model',
                        dest='model',
                        choices=MODELS)
    args = parser.parse_args(args_to_parse)

    return args


if __name__ == "__main__":
    args = parse_apply_ecg_qc_args(sys.argv[1:])
    args_dict = convert_args_to_dict(args)
    apply_ecg_qc(**args_dict)
