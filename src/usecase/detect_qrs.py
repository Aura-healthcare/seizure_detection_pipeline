import os
from typing import List

import click
import pandas as pd

from src.domain.qrs_detector import QRSDetector

OUTPUT_FOLDER = 'output/qrs'
METHODS = ['hamilton', 'xqrs', 'gqrs', 'swt', 'engelsee']
DEFAULT_METHOD = 'hamilton'


def write_detections_csv(detections: pd.DataFrame, infos: List[str]) -> None:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    filename = os.path.join(OUTPUT_FOLDER, f"{'_'.join(infos)}.csv")
    detections.to_csv(filename, sep=',', index=True)


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
    detected_qrs, rr_intervals = qrs_detector.get_cardiac_infos(
        signal, sampling_frequency, method)
    df_detections = ecg_data.copy()
    df_detections = df_detections.iloc[detected_qrs[:-1]]
    df_detections['frame'] = detected_qrs[:-1]
    df_detections['rr_interval'] = rr_intervals
    df_detections.drop(columns='signal', inplace=True)
    write_detections_csv(df_detections, infos)


if __name__ == "__main__":
    detect_qrs()
