import os
import json
from typing import Tuple, List

import pandas as pd
import ecg_qc
import click

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


def write_quality_json(quality: List[int], infos: List[str]) -> None:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    with open(os.path.join(OUTPUT_FOLDER, f"{'_'.join(infos)}.json"), 'w') \
            as outfile:
        json.dump(quality, outfile)


@click.command()
@click.option('--ecg-data', required=True)
@click.option('--sampling-frequency', required=True, type=int)
@click.option('--model', required=True, type=click.Choice(MODELS))
@click.option('--infos', required=True, type=list)
def apply_ecg_qc(ecg_data: pd.DataFrame,
                 sampling_frequency: int,
                 model: str,
                 infos: List[str]) -> None:
    signal = list(ecg_data['signal'])
    model_path, model_name, length_chunk, is_normalized = parse_model(model)
    algo = ecg_qc.ecg_qc(sampling_frequency=sampling_frequency,
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
    extended_infos = infos + [model_name]
    write_quality_json(signal_quality, extended_infos)


if __name__ == "__main__":
    apply_ecg_qc()
