import os
import json
from typing import Tuple, List

import ecg_qc
import click

from src.usecase.ecg_channel_read import ecg_channel_read

MODEL_FOLDER = 'models'
MODELS = os.listdir(MODEL_FOLDER)
OUTPUT_FOLDER = 'output/quality'
DEFAULT_PATH = '/home/DATA/lateppe/Recherche_ECG/'


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


def write_quality_json(quality: List[int], infos: List[str]) -> str:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    filename = f"{'_'.join(infos)}.json"
    with open(os.path.join(OUTPUT_FOLDER, filename), 'w') \
            as outfile:
        json.dump(quality, outfile)
    return filename


def apply_ecg_qc(patient: str,
                 record: str,
                 segment: str,
                 channel_name: str,
                 start_time: str,
                 end_time: str,
                 model: str,
                 infos: List[str],
                 data_path: str = DEFAULT_PATH) -> str:
    '''
    Applies an ECG QC model on a signal, and writes results in a json file.
    '''
    sampling_frequency, ecg_data, _, _ = ecg_channel_read(
        patient, record, segment, channel_name,
        start_time, end_time, data_path)
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
    extended_infos = infos + ['#', model_name]
    filename = write_quality_json(signal_quality, extended_infos)
    return filename


@click.command()
@click.option('--patient', required=True)
@click.option('--record', required=True)
@click.option('--segment', required=True)
@click.option('--channel-name', required=True)
@click.option('--start-time', required=True)
@click.option('--end-time', required=True)
@click.option('--model', required=True, type=click.Choice(MODELS))
@click.option('--infos', required=True, multiple=True)
@click.option('--data-path', required=True, default=DEFAULT_PATH)
def main(patient: str,
         record: str,
         segment: str,
         channel_name: str,
         start_time: str,
         end_time: str,
         model: str,
         infos: List[str],
         data_path: str = DEFAULT_PATH) -> None:
    _ = apply_ecg_qc(
        patient, record, segment, channel_name,
        start_time, end_time, model, list(infos), data_path)


if __name__ == "__main__":
    main()
