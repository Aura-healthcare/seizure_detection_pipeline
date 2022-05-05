"""
This script is used to compare RR intervals from two detectors to find noise.

Copyright (C) 2022 Association Aura
SPDX-License-Identifier: GPL-3.0
"""
import argparse
import pandas as pd
import itertools
import sys
import fileinput

from typing import List

sys.path.append(".")
from src.usecase.utilities import convert_args_to_dict, generate_output_path
from src.usecase.apply_ecg_qc import create_ecg_qc_tse_bi

DEFAULT_TOLERENCE_MS = 20  # about 10 ms at 256 HZ
CORRESPONDANCE_LABEL = 'clean'
NO_CORRESPONDANCE_LABEL = 'noisy'
DEFAULT_length_chunk = 2


def find_frame_correspondance(
        reference_rr_interval_cumulated: float,
        df_comparison: pd.DataFrame,
        frame_tolerence: int,
        correspondance_label: str = CORRESPONDANCE_LABEL,
        no_correspondance_label: str = NO_CORRESPONDANCE_LABEL) -> str:
    """Find if a rr as a correspondance at same frame, modulo a tolrence.

    Parameters
    ----------
    reference_rr_interval_cumulated : float
        The frame used to search comparison
    df_comparison : pd.DataFrame
        The pd.DataFrame where to search the comparison
    frame_tolerence : int
        The frame tolerence to include, in milliseconds
    correspondance_label : str
        The label to return if a correspondance is found
    no_correspondance_label : str
        The label to return if no correspondance is found

    Returns
    -------
    quality label : str
        Returns a quality label: correpsondance label if some correspondance is
        found, no_correspondance_label else
    """
    mask = ((df_comparison['rr_interval_cumulated'] > (
             reference_rr_interval_cumulated - frame_tolerence)) & (
            (df_comparison['rr_interval_cumulated'] < (
             reference_rr_interval_cumulated + frame_tolerence))))
    correspondances = df_comparison[mask]['rr_interval_cumulated'].values

    return correspondance_label if len(correspondances) > 0 \
        else no_correspondance_label


def consolidate_labels(df_reference: pd.DataFrame,
                       timestamp: pd.Timestamp,
                       length_chunk: int,
                       label_to_find: str = 'noisy',
                       other_label: str = 'clean') -> str:
    """Return labels found in fixed sized segments.

    parameters
    ----------
    df_reference : pd.dataframe
        The dataframe including timestamps and quality
    timestamp : pd.timestamp
        The timestamp used for lower filter
    length_chunk : int
        The duration in seconds for upper filter
    label_to_find : str
        If this label is found at least one, return this label
    other_label : str
        Label to return if label_to_find is not found

    returns
    -------
    consolidated_label : str
        The signal quality over the duration
    """
    correspondances = df_reference[(df_reference['timestamp'] >= timestamp) & (
                                   (df_reference['timestamp'] <= (
                                    timestamp + pd.Timedelta(
                                        seconds=length_chunk))))][
                                            'correspondance'].values
    return label_to_find if label_to_find in correspondances else other_label


def create_signal_quality_label_by_length_chunk(
        df_reference: pd.DataFrame,
        length_chunk: int = DEFAULT_length_chunk) -> List[str]:
    """Return an array of quality label by segments of fixed size.

    Parameters
    ----------
    df_reference : pd.DataFrame
        The dataframe including timestamps and quality
    length_chunk : int
        The duration in seconds for upper filter

    Returns
    -------
    signal_quality_label : np.array
        The list of signal quality labels, one label every chunk length of time
    start_time : pd.Datetime
        The start time of the first quality label
    end_time : pd.Datetme
        The end time of the last quality label
    """
    first_detected_qrs = pd.Timestamp(df_reference['timestamp'].iloc[0])
    last_detected_qrs = pd.Timestamp(df_reference['timestamp'].iloc[-1])

    duration = last_detected_qrs - first_detected_qrs
    n_chunks = int(duration.seconds / length_chunk) + 1

    chunk_timestamps = [first_detected_qrs + pd.Timedelta(
        seconds=2 * iteration)
        for iteration in range(n_chunks)]
    df_signal_quality_label = pd.DataFrame(
        data={"timestamp": chunk_timestamps})
    df_signal_quality_label['label'] = df_signal_quality_label[
        'timestamp'].apply(
            lambda x: consolidate_labels(
                df_reference=df_reference,
                timestamp=x,
                length_chunk=length_chunk))

    return df_signal_quality_label['label'].values, \
        chunk_timestamps[0], \
        chunk_timestamps[-1]


def compare_qrs_detector(reference_rr_intervals_file_path: str,
                         comparison_rr_intervals_file_path: str,
                         frame_tolerence: int,
                         output_folder: str,
                         formatting: str,
                         length_chunk: int = DEFAULT_length_chunk) -> str:
    """Compare qrs detection from two detectors to find noise.

    parameters
    ----------
    reference_rr_intervals_file_path : str
        The first RR intervals file path
    comparison_rr_intervals_file_path : str
        The second RR intervals file path
    frame_tolerence : int
        The frame tolerence to include, in milliseconds
    output_folder : str
        Path of the output folder
    formatting: str
        Formatting of the tse_bi output ('teppe' or 'tuh')
    length_chunk : int
        The duration in seconds for upper filter

    returns
    -------
    output_file_path: str
        The output of the file generated
    """
    df_reference = pd.read_csv(reference_rr_intervals_file_path)
    df_reference['timestamp'] = df_reference['timestamp'].apply(
        lambda x: pd.Timestamp(x))
    df_comparison = pd.read_csv(comparison_rr_intervals_file_path)
    df_comparison['timestamp'] = df_comparison['timestamp'].apply(
        lambda x: pd.Timestamp(x))

    for df in [df_reference, df_comparison]:
        df['rr_interval_cumulated'] = list(
            itertools.accumulate(df['rr_interval'].values, lambda a, b: a + b))
    df_reference['correspondance'] = \
        df_reference['rr_interval_cumulated'].apply(
            lambda x: find_frame_correspondance(
                reference_rr_interval_cumulated=x,
                df_comparison=df_comparison,
                frame_tolerence=frame_tolerence))

    signal_quality_label, start_time, end_time = \
        create_signal_quality_label_by_length_chunk(
            df_reference=df_reference,
            length_chunk=length_chunk)

    df_ecg_qc = create_ecg_qc_tse_bi(
        signal_quality=signal_quality_label,
        length_chunk=length_chunk,
        start_time=start_time,  # Start of first beat?
        end_time=end_time,
        formatting=formatting)

    output_file_path = generate_output_path(
        input_file_path=reference_rr_intervals_file_path,
        output_folder=output_folder,
        format="tse_bi",
        prefix="qc_qrs")
    df_ecg_qc.to_csv(output_file_path, sep=" ", index=False, header=False)

    # Format properly to tse_bi standard
    with fileinput.FileInput(output_file_path, inplace=True) as tse_bi_file:
        for line in tse_bi_file:
            print(line.replace('"', ""), end='')

    return output_file_path


def parse_compare_qrs_detectors_args(
        args_to_parse: List[str]) -> argparse.Namespace:
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
    parser = argparse.ArgumentParser(description="CLI parameter input")
    parser.add_argument("--reference-rr-intervals-file-path",
                        dest="reference_rr_intervals_file_path",
                        type=str,
                        required=True)
    parser.add_argument("--comparison-rr-intervals-file-path",
                        dest="comparison_rr_intervals_file_path",
                        type=str,
                        required=True)
    parser.add_argument("--frame-tolerence",
                        dest="frame_tolerence",
                        type=int,
                        default=DEFAULT_TOLERENCE_MS)
    parser.add_argument("--output-folder",
                        dest="output_folder")
    parser.add_argument("--formatting",
                        dest="formatting",
                        default='dataset')
    return parser.parse_args(args_to_parse)


if __name__ == "__main__":
    args = parse_compare_qrs_detectors_args(sys.argv[1:])
    args_dict = convert_args_to_dict(args)
    compare_qrs_detector(**args_dict)
