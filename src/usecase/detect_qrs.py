import os
import json
from typing import List

import click
import pandas as pd

from src.domain.qrs_detector import QRSDetector

OUTPUT_FOLDER = 'output/qrs'
METHODS = ['hamilton', 'xqrs', 'gqrs', 'swt', 'engelsee']
DEFAULT_METHOD = 'hamilton'


def write_detections_json(detections: List[int], infos: List[str]) -> None:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    with open(os.path.join(OUTPUT_FOLDER, f"{'_'.join(infos)}.json"), 'w') \
            as outfile:
        json.dump(detections, outfile)


@click.command()
@click.option('--ecg-data', required=True)
@click.option('--sampling-frequency', required=True, type=int)
@click.option('--method', required=True, type=click.Choice(METHODS),
              default=DEFAULT_METHOD)
@click.option('--infos', required=True, type=list)
def detect_qrs(ecg_data: pd.DataFrame,
               sampling_frequency: int,
               method: str,
               infos: list) -> None:
    qrs_detector = QRSDetector()
    signal = list(ecg_data['signal'])
    _, rr_intervals = qrs_detector.get_cardiac_infos(
        signal, sampling_frequency, method)
    write_detections_json(rr_intervals, infos)
