#!/usr/bin/env python3
import os
from typing import Tuple

import pandas as pd
import click

from src.infrastructure.edf_loader import EdfLoader

DEFAULT_PATH = '/home/DATA/lateppe/Recherche_ECG/'


@click.command()
@click.option('--data-path', required=True, default=DEFAULT_PATH)
@click.option('--patient', required=True)
@click.option('--record', required=True)
@click.option('--segment', required=True)
@click.option('--channel-name', required=True)
@click.option('--start-time', required=True)
@click.option('--end-time', required=True)
def ecg_channel_read(patient: str,
                     record: str,
                     segment: str,
                     channel_name: str,
                     start_time: str,
                     end_time: str,
                     data_path: str = DEFAULT_PATH) -> \
                     Tuple[int, pd.DataFrame, pd.DataFrame, pd.DataFrame]:

    edfloader = EdfLoader(data_path, patient, record, segment)
    df_ecg = edfloader.convert_edf_to_dataframe(
        channel_name, start_time, end_time)

    sample_frequency = edfloader.sampling_frequency_hz

    annotation_file = os.path.join(
        data_path, patient, f'Annotations_EEG_{record}.csv')
    segments_file = os.path.join(
        data_path, patient, f'Segments_EEG_{record}.csv')

    df_annot = pd.read_csv(
        annotation_file, index_col=0, sep='|', encoding='latin-1')
    df_seg = pd.read_csv(segments_file, sep='|')

    return sample_frequency, df_ecg, df_annot, df_seg


if __name__ == "__main__":
    ecg_channel_read()
# fs, df_ecg, df_annot, df_seg = ecg_channel_read('/home/DATA/lateppe/Recherche_ECG/', 'PAT_2', '3', 's1', "EMG1+EMG1-", "2020-12-14 21:55:00", "2020-12-14 23:05:00")
