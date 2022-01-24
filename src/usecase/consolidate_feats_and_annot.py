"""
This script is used to consolidate hrv features dataset and annotations.

Copyright (C) 2021 Association Aura
SPDX-License-Identifier: GPL-3.0
"""
import argparse
from datetime import tzinfo
import numpy as np
import pandas as pd
import sys
from typing import List

from pandas.core.indexes import interval

sys.path.append('.')
from src.usecase.utilities import convert_args_to_dict, generate_output_path

OUTPUT_FOLDER = 'exports/consolidated_dataset'
WINDOW_INTERVAL = 200
SEGMENT_SIZE_TRESHOLD = 0.5


def consolidate_feats_and_annot(
        features_file_path: str,
        annotations_file_path: str,
        output_folder: str = OUTPUT_FOLDER,
        window_interval: int = WINDOW_INTERVAL,
        segment_size_treshold: float = SEGMENT_SIZE_TRESHOLD,
        crop_dataset: bool = True) -> str:
    """
    Create a pd.DataFrame form a tse_bi file.

    Parameters
    ----------
    features_file_path : str
        Path of features file. Should be a CSV.
    annotations_file_path : str
        Path of annotations file. Format should be *.tse_bi.
    output_folder : str
        Path of the output folder
    window_interval : int
        Period in milliseconds from interval start_time to create upper limit
    segment_size_treshold : float
        Proportion of labels required to return a label
    crop_dataset : bool
        If True, removes the first and last line from the dataset:
            * First Line cannot include computed fetures
            * Last line cannot include label

    Returns
    -------
    df_tse_bi : pd.DataFrame
        DataFrame including the data from input tse_bi
    """
    df_features = pd.read_csv(features_file_path)
    df_tse_bi = read_tse_bi(annotations_file_path)

    # df_features['label'] = df_features['timestamp'].apply(
    #     lambda interval_start_time: get_label_on_interval(
    #         df_tse_bi=df_tse_bi,
    #         interval_start_time=interval_start_time,
    #         window_interval=window_interval,
    #         segment_size_treshold=segment_size_treshold))
    df_features['label'] = df_features['interval_index'].apply(
        lambda interval_start_time: get_label_on_interval(
            df_tse_bi=df_tse_bi,
            interval_start_time=interval_start_time,
            window_interval=window_interval,
            segment_size_treshold=segment_size_treshold))


    if crop_dataset:
        df_features.drop(
            [df_features.index[0], df_features.index[-1]],
            inplace=True)

    input_file_generated = ''.join(
        [annotations_file_path[:-7],
         features_file_path[-9:-4],
         '.tse_bi'])

    output_file_path = generate_output_path(
        input_file_path=input_file_generated,
        output_folder=output_folder,
        format='csv',
        prefix='cons')

    df_features.to_csv(output_file_path, sep=',', index=False)
    return output_file_path


def read_tse_bi(annotations_file_path: str) -> pd.DataFrame:
    """
    Create a pd.DataFrame form a tse_bi file.

    As in tse_bi stanadard an empty line seperates the data from the file
    headers, the first empty line is used to infer where data starts.

    Parameters
    ----------
    annotations_file_path : str
        Path of annotations file. Format should be *.tse_bi.

    Returns
    -------
    df_tse_bi : pd.DataFrame
        DataFrame including the data from input tse_bi
    """
    if annotations_file_path.split('/')[-1].split('.')[-1] != 'tse_bi':
        raise ValueError(
            f'Please input a tse_bi file. Input: {annotations_file_path}')
    # La teppe format, which is slightly different
    try:
        df_search_empty_line = pd.read_csv(annotations_file_path,
                                       skip_blank_lines=False,
                                       sep=' ',
                                       header=None)
        database_format = 'teppe'
    except Exception:
        df_search_empty_line = pd.read_csv(annotations_file_path,
                                           skip_blank_lines=False,
                                           header=None)
        database_format = 'tuh'

    first_empty_line_index = df_search_empty_line[
    df_search_empty_line.isnull().all(1)].index[0]

    df_tse_bi = pd.read_csv(annotations_file_path,
                            sep=' ',
                            skiprows=first_empty_line_index,
                            header=None)

    df_tse_bi.columns = ['start', 'end', 'annotation', 'probablility']

    if database_format == 'teppe':
        df_tse_bi['start'] = df_tse_bi['start'].apply(
            lambda x: pd.Timestamp(x))
        df_tse_bi['end'] = df_tse_bi['end'].apply(
            lambda x: pd.Timestamp(x))

    else:
        df_tse_bi.loc[:, ['start', 'end']] = df_tse_bi.loc[
            :, ['start', 'end']].apply(lambda x: x * 1_000)


    return df_tse_bi


def get_label_on_interval(df_tse_bi: pd.DataFrame,
                          interval_start_time: int,
                          window_interval: int,
                          segment_size_treshold: float) -> float:
    """
    From an annotation DataFrame, select annotations on an interval.

    Parameters
    ----------
    df_tse_bi : pd.DataFrame
        pd.DataFrame including annotations
    interval_start_time : int
        Timestamp in milliseconds for lower limit
    window_interval : int
        Period in milliseconds from interval start_time to create upper limit
    segment_size_treshold : float
        Proportion of labels required to return a label

    Returns
    -------
    label_ratio : float
        Ratio of seizure other the segment selected
    """
    try:
        interval_start_time = np.datetime64(interval_start_time)
        end_marker = interval_start_time + pd.Timedelta(milliseconds=window_interval)
    except:
        end_marker = interval_start_time + window_interval/1_000 #ms
    
    df_filtered = df_tse_bi[
        (df_tse_bi['start'] < end_marker) &
        (df_tse_bi['end'] > interval_start_time)
        ]
    df_filtered.loc[:, 'start'] = df_filtered['start'].apply(
        lambda x: interval_start_time if x <= interval_start_time else x)
    df_filtered.loc[:, 'end'] = df_filtered['end'].apply(
        lambda x: end_marker if x > end_marker else x)
    # df_filtered.loc[:, 'length'] = (df_filtered['end'] - df_filtered['start']).astype('timedelta64[ms]')
    df_filtered.loc[:, 'length'] = (df_filtered['end'] - df_filtered['start'])
    df_filtered.loc[:, 'length'] = df_filtered['length'].apply(lambda x: x if x > 0 else 0)
    df_filtered = df_filtered.groupby(['annotation']).sum()['length']
    try:
        seiz_length = df_filtered['seiz']
    except KeyError:
        seiz_length = 0

    try:
        bckg_length = df_filtered['bckg']
    except KeyError:
        bckg_length = 0
    try:
        label_ratio = seiz_length / (seiz_length + bckg_length)
        label_ratio = int(round(label_ratio, 0))
    except:
        label_ratio = np.nan

#    if seiz_length + bckg_length <= window_interval * segment_size_treshold:
#        label_ratio = np.nan
#    else:
#        if bckg_length > 0:
#            label_ratio = seiz_length / (seiz_length + bckg_length)
#        else:
#            label_ratio = np.nan


    return label_ratio


def parse_consolidate_feats_and_annot_args(
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
    parser = argparse.ArgumentParser(description='CLI parameter input')
    parser.add_argument('--features-file-path',
                        dest='features_file_path',
                        required=True)
    parser.add_argument('--annotations-file-path',
                        dest='annotations_file_path',
                        required=True)
    parser.add_argument('--output-folder',
                        dest='output_folder')
    parser.add_argument('--window-interval',
                        dest='window_interval',
                        type=int)
    parser.add_argument('--segment-size-treshold',
                        dest='segment_size_treshold',
                        type=float)
    parser.add_argument('--crop-dataset',
                        dest='crop_dataset',
                        type=bool)
    args = parser.parse_args(args_to_parse)

    return args


if __name__ == '__main__':
    args = parse_consolidate_feats_and_annot_args(sys.argv[1:])
    args_dict = convert_args_to_dict(args)
    consolidate_feats_and_annot(**args_dict)
