import os
from typing import List, Tuple

import click
import pandas as pd

from src.domain.qrs_detector import QRSDetector
from src.usecase.ecg_channel_read import ecg_channel_read


OUTPUT_FOLDER = 'output/qrs'
METHODS = ['hamilton', 'xqrs', 'gqrs', 'swt', 'engelsee']
DEFAULT_METHOD = 'hamilton'
DEFAULT_PATH = '/home/DATA/lateppe/Recherche_ECG/'


def write_detections_csv(detections: pd.DataFrame, infos: List[str]) -> str:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    filename = f"{'_'.join(infos)}.csv"
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    detections.to_csv(filepath, sep=',', index=True)
    return filename


def detect_qrs(patient: str,
               record: str,
               segment: str,
               channel_name: str,
               start_time: str,
               end_time: str,
               method: str,
               infos: list,
               data_path: str = DEFAULT_PATH) -> Tuple[int, str]:
    sampling_frequency, ecg_data, _, _ = ecg_channel_read(
        patient, record, segment, channel_name,
        start_time, end_time, data_path)
    qrs_detector = QRSDetector()
    signal = list(ecg_data['signal'])
    detected_qrs, rr_intervals = qrs_detector.get_cardiac_infos(
        signal, sampling_frequency, method)
    df_detections = ecg_data.copy()
    df_detections = df_detections.iloc[detected_qrs[:-1]]
    df_detections['frame'] = detected_qrs[:-1]
    df_detections['rr_interval'] = rr_intervals
    df_detections.drop(columns='signal', inplace=True)
    filename = write_detections_csv(df_detections, infos)
    return sampling_frequency, filename


@click.command()
@click.option('--patient', required=True)
@click.option('--record', required=True)
@click.option('--segment', required=True)
@click.option('--channel-name', required=True)
@click.option('--start-time', required=True)
@click.option('--end-time', required=True)
@click.option('--method', required=True, type=click.Choice(METHODS),
              default=DEFAULT_METHOD)
@click.option('--infos', required=True, multiple=True)
@click.option('--data-path', required=True, default=DEFAULT_PATH)
def main(patient: str,
         record: str,
         segment: str,
         channel_name: str,
         start_time: str,
         end_time: str,
         method: str,
         infos: list,
         data_path: str = DEFAULT_PATH) -> None:
    _ = detect_qrs(
        patient, record, segment, channel_name,
        start_time, end_time, method, list(infos), data_path)


if __name__ == "__main__":
    main()
