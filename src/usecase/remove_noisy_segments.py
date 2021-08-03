import os
import json

import pandas as pd
from typing import List
import click

RR_INTERVALS_FOLDER = 'output/frames'
CHUNKS_FOLDER = 'output/quality'
OUTPUT_FOLDER = 'output/noise_free_frames'


def write_noise_free_detections_csv(detections: pd.DataFrame,
                                    infos: List[str]) -> None:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    filename = os.path.join(OUTPUT_FOLDER, f"{'_'.join(infos)}.csv")
    detections.to_csv(filename, sep=',', index=True)


def remove_noisy_segments_from_df(df_rr_intervals: pd.DataFrame,
                                  list_noisy_segments: List[int],
                                  length_chunk: int,
                                  sampling_frequency: int) -> pd.DataFrame:
    df_noise_free_rr_intervals = df_rr_intervals.copy()
    for i in range(len(list_noisy_segments)):
        chunk_is_noisy = not list_noisy_segments[i]
        if chunk_is_noisy:
            frame_start_noise = i * sampling_frequency * length_chunk
            frame_end_noise = (i + 1) * sampling_frequency * length_chunk - 1
            df_noise_free_rr_intervals = df_noise_free_rr_intervals[
                (df_noise_free_rr_intervals['frame'] < frame_start_noise) |
                (df_noise_free_rr_intervals['frame'] > frame_end_noise)]
    return df_noise_free_rr_intervals


@click.command()
@click.option('--rr-intervals-file', required=True)
@click.option('--chunk-file', required=True)
@click.option('--length-chunk', required=True, type=int)
@click.option('--sampling-frequency', required=True, type=int)
def remove_noisy_segments(rr_intervals_file: str, chunk_file: str,
                          length_chunk: int, sampling_frequency: int) -> None:
    df_rr_intervals = pd.read_csv(
        os.path.join(RR_INTERVALS_FOLDER, rr_intervals_file),
        sep=',',
        index_col=0
    )
    with open(os.path.join(CHUNKS_FOLDER, chunk_file), 'r') as f:
        list_noisy_segments = json.load(f)
    df_noise_free_rr_intervals = remove_noisy_segments_from_df(
        df_rr_intervals, list_noisy_segments, length_chunk, sampling_frequency)
    infos = rr_intervals_file.split('.')[0]
    write_noise_free_detections_csv(df_noise_free_rr_intervals, infos)


if __name__ == '__main__':
    remove_noisy_segments()
