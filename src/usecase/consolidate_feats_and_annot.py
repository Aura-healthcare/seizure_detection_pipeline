import argparse
import numpy as np
import pandas as pd
import sys
from typing import List

sys.path.append('.')
from src.usecase.utilities import convert_args_to_dict, generate_output_path

OUTPUT_FOLDER = 'exports/consolidated_dataset'
WINDOW_SIZE = 10_000
SEGMENT_SIZE_TRESHOLD = 0.9


def consolidate_feats_and_annot(
        features_file_path: str,
        annotations_file_path: str,
        output_folder: str = OUTPUT_FOLDER,
        window_interval: int = WINDOW_SIZE,
        segment_size_treshold: float = SEGMENT_SIZE_TRESHOLD,
        crop_dataset: bool = True) -> str:

    df_features = pd.read_csv(features_file_path)
    df_tse_bi = read_tse_bi(annotations_file_path)

    df_features['label'] = df_features['interval_start_time'].apply(
        lambda interval_start_time: get_label_on_interval(
            df_tse_bi=df_tse_bi,
            interval_start_time=interval_start_time,
            window_interval=window_interval,
            segment_size_treshold=segment_size_treshold))

    if crop_dataset:
        df_features.drop(
            [df_features.index[0], df_features.index[-1]],
            inplace=True)

    output_file_path = generate_output_path(
        input_file_path=annotations_file_path,
        output_folder=output_folder,
        format='csv')

    df_features.to_csv(output_file_path, sep=',', index=False)

    return output_file_path


def read_tse_bi(annotations_file_path: str) -> pd.DataFrame:

    df_tse_bi = pd.read_csv(annotations_file_path,
                            sep=' ',
                            skiprows=1,
                            header=None)

    df_tse_bi.columns = ['start', 'end', 'annotation', 'probablility']
    df_tse_bi.loc[:, ['start', 'end']] = df_tse_bi.loc[:, ['start', 'end']].\
        apply(lambda x: x * 1_000)

    return df_tse_bi


def get_label_on_interval(df_tse_bi: pd.DataFrame,
                          interval_start_time: int,
                          window_interval: int,
                          segment_size_treshold: float) -> float:

    end_marker = interval_start_time + window_interval
    df_filtered = df_tse_bi[
        (df_tse_bi['start'] <= interval_start_time)]

    df_filtered['start'] = df_filtered['start'].apply(
        lambda x: interval_start_time if x <= interval_start_time else x)
    df_filtered['end'] = df_filtered['end'].apply(
        lambda x: end_marker if x > end_marker else x)
    df_filtered['length'] = df_filtered['end'] - df_filtered['start']
    df_filtered = df_filtered.groupby(['annotation']).sum()['length']

    try:
        seiz_length = df_filtered['seiz']
    except KeyError:
        seiz_length = 0

    try:
        bckg_length = df_filtered['bckg']
    except KeyError:
        bckg_length = 0

    if seiz_length + bckg_length <= window_interval * segment_size_treshold:
        label_ratio = np.nan
    else:
        if bckg_length > 0:
            label_ratio = seiz_length / bckg_length
        else:
            label_ratio = np.nan

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
    parser.add_argument('--window-size',
                        dest='window_size',
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
