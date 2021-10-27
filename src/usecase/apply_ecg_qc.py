import os
import json
from typing import Tuple, List

import ecg_qc
import click

from src.infrastructure.edf_loader import EdfLoader

MODEL_FOLDER = 'models'
MODELS = os.listdir(MODEL_FOLDER)
OUTPUT_FOLDER = 'output/quality'

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


def write_quality_json(quality: List[int], exam_id: str, model_name:str) -> str:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    filename = f"{exam_id}_{model_name}.csv"
    with open(os.path.join(OUTPUT_FOLDER, filename), 'w') \
            as outfile:
        json.dump(quality, outfile)
    return filename


def apply_ecg_qc(filename: str,
                 model: str,
                 exam_id: str) -> str:
    '''
    Applies an ECG QC model on a signal, and writes results in a json file.
    '''
    edfloader = EdfLoader(filename)
    ecg_channel_name = edfloader.get_ecg_candidate_channel()
    start_time, end_time = edfloader.get_edf_file_interval()

    sampling_frequency, ecg_data = edfloader.ecg_channel_read(ecg_channel_name, start_time, end_time)
    signal = list(ecg_data['signal'])

    model_path, model_name, length_chunk, is_normalized = parse_model(model)
    algo = ecg_qc.EcgQc(sampling_frequency=sampling_frequency,
                        model=model_path, normalized=is_normalized)
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
    filename = write_quality_json(signal_quality, exam_id, model_name)
    return filename


@click.command()
@click.option('--filename', required=True)
@click.option('--model', required=True, type=click.Choice(MODELS))
@click.option('--exam-id', required=True)
def main(filename: str,
         model: str,
         exam_id: str) -> None:
    _ = apply_ecg_qc(filename, model, exam_id)


if __name__ == "__main__":
    main()
