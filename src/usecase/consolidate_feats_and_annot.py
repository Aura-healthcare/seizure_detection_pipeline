"""
This script is used to consolidate hrv features dataset and annotations.

Copyright (C) 2021 Association Aura
SPDX-License-Identifier: GPL-3.0
"""
import argparse
import numpy as np
import pandas as pd
import sys
from typing import List

sys.path.append(".")
from src.usecase.utilities import convert_args_to_dict, generate_output_path

OUTPUT_FOLDER = "exports/consolidated_dataset"
WINDOW_INTERVAL = 1_000
SEGMENT_SIZE_TRESHOLD = 0.9

LABEL_TARGET = "seiz"


def consolidate_feats_and_annot(
    features_file_path: str,
    annotations_file_path: str,
    output_folder: str = OUTPUT_FOLDER,
    window_interval: int = WINDOW_INTERVAL,
    segment_size_treshold: float = SEGMENT_SIZE_TRESHOLD,
    label_target: str = LABEL_TARGET,
    crop_dataset: bool = True,
) -> str:
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
    # Delete crop_dataset ?
    # Convert tse_bi of TUH by adding the start of df_features
    df_features = pd.read_csv(features_file_path)
    df_tse_bi = read_tse_bi(annotations_file_path)

    # Dataset tse_bi format
    if df_tse_bi["start"].loc[0] > np.datetime64(1, "s"):
        df_features["label"] = df_features["timestamp"].apply(
            lambda interval_start_time: get_label_on_interval(
                df_tse_bi=df_tse_bi,
                interval_start_time=interval_start_time,
                window_interval=window_interval,
                segment_size_treshold=segment_size_treshold,
                label_target="seiz"
            )
        )

    # Tuh tse_bi format
    else:
        df_features["label"] = df_features["interval_start_time"].apply(
            lambda interval_start_time: get_label_on_interval(
                df_tse_bi=df_tse_bi,
                interval_start_time=interval_start_time,
                window_interval=window_interval,
                segment_size_treshold=segment_size_treshold,
                label_target=label_target
            )
        )

    if crop_dataset:
        df_features.drop([df_features.index[0], df_features.index[-1]],
                         inplace=True)

    input_file_generated = "".join(
        [annotations_file_path[:-7], features_file_path[-9:-4], ".tse_bi"]
    )

    output_file_path = generate_output_path(
        input_file_path=input_file_generated,
        output_folder=output_folder,
        format="csv",
        prefix="cons",
    )

    df_features.to_csv(output_file_path, sep=",", index=False)

    return output_file_path


def read_tse_bi(annotations_file_path: str) -> pd.DataFrame:
    """
    Create a pd.DataFrame form a tse_bi file.

    As in tse_bi standard an empty line seperates the data from the file
    headers, the first empty line is used to infer where data starts.

    Start and end columns are converted to numpy timestamp64. For TUH
    tse_bi files, there is no timestamp but duration from the start of the
    recording. However, it is converted in timestamp all the same as it makes
    no incompatibily with other usages. However, starting date will always be
    1970-01-01 00:00:00.

    Parameters
    ----------
    annotations_file_path : str
        Path of annotations file. Format should be *.tse_bi.

    Returns
    -------
    df_tse_bi : pd.DataFrame
        DataFrame including the data from input tse_bi
    """
    if annotations_file_path.split("/")[-1].split(".")[-1] != "tse_bi":
        raise ValueError(f"Please input a tse_bi file. Input:"
                         f"{annotations_file_path}")
    # La teppe format, which is slightly different

    with open(annotations_file_path, "r") as annotations_file:
        first_empty_line = 0
        for line in annotations_file:
            first_empty_line += 1
            if line in ["\n", "r\n", "   \n"]:
                break
    try:
        df_tse_bi = pd.read_csv(
            annotations_file_path,
            skiprows=first_empty_line,
            skip_blank_lines=False,
            sep=" ",
            header=None,
        )

    except pd.errors.EmptyDataError:
        sys.exit("tse_bi file is empty")

    df_tse_bi.columns = ["start", "end", "annotation", "probablility"]

    if df_tse_bi["start"].iloc[0] == 0:  # TUH format
        df_tse_bi.loc[:, "start"] = df_tse_bi["start"].apply(
            lambda x: np.datetime64(int(x) * 1000, "ms")
        )

        df_tse_bi.loc[:, "end"] = df_tse_bi["end"].apply(
            lambda x: np.datetime64(int(x) * 1000, "ms")
        )

    else:
        df_tse_bi.loc[:, "start"] = df_tse_bi["start"].apply(
            lambda x: np.datetime64(x))

        df_tse_bi.loc[:, "end"] = df_tse_bi["end"].apply(
            lambda x: np.datetime64(x))

    return df_tse_bi


def get_label_on_interval(
    df_tse_bi: pd.DataFrame,
    interval_start_time: int,
    window_interval: int,
    segment_size_treshold: float,
    label_target: str
) -> float:
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
    # Converting interval_start_time to np.datetime64
    try:
        interval_start_time = np.datetime64(interval_start_time)

    except ValueError:
        interval_start_time = np.datetime64(int(interval_start_time), "ms")

    # Computing end marker
    end_marker = interval_start_time + \
        pd.Timedelta(milliseconds=window_interval)

    df_tse_bi_temp = df_tse_bi.copy()

    # Filtering over intervals
    df_tse_bi_temp.loc[:, "start"] = df_tse_bi_temp["start"].apply(
        lambda x: interval_start_time if x <= interval_start_time else x
    )
    df_tse_bi_temp.loc[:, "end"] = df_tse_bi_temp["end"].apply(
        lambda x: end_marker if x > end_marker else x
    )
    df_tse_bi_temp.loc[:, "length"] = df_tse_bi_temp["end"] \
        - df_tse_bi_temp["start"]

    # Setting negative length at 0
    df_tse_bi_temp.loc[:, "length"] = df_tse_bi_temp["length"].apply(
        lambda x: x.total_seconds() if x > pd.Timedelta(0) else 0
    )

    # Checking checking the duration by seiz/bckg annotations
    df_tse_bi_temp = df_tse_bi_temp.groupby(["annotation"]).sum()["length"]

    print(df_tse_bi_temp)
    assert(len(df_tse_bi_temp.index) <= 2)
    try:
        label_other = [label for label in df_tse_bi_temp.index
                       if label != label_target][0]
    except IndexError:
        label_other = None

    # Computing the size of each class, then label
    try:
        label_target_length = df_tse_bi_temp[label_target]
    except KeyError:
        label_target_length = 0

    try:
        label_other_length = df_tse_bi_temp[label_other]
    except KeyError:
        label_other_length = 0

    try:
        label_ratio = label_target_length / (label_target_length
                                             + label_other_length)
        label_ratio = round(label_ratio, 2)
        if label_ratio > segment_size_treshold:
            label_ratio = 1

    except ValueError:
        label_ratio = np.nan

    return label_ratio


def parse_consolidate_feats_and_annot_args(
    args_to_parse: List[str],
) -> argparse.Namespace:
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
    parser.add_argument(
        "--features-file-path", dest="features_file_path", required=True
    )
    parser.add_argument(
        "--annotations-file-path", dest="annotations_file_path", required=True
    )
    parser.add_argument("--output-folder", dest="output_folder")
    parser.add_argument("--window-interval", dest="window_interval", type=int)
    parser.add_argument(
        "--segment-size-treshold", dest="segment_size_treshold", type=float
    )
    parser.add_argument("--crop-dataset", dest="crop_dataset", type=bool)
    parser.add_argument("--label-target ", dest="label_target", type=str)
    args = parser.parse_args(args_to_parse)

    return args


if __name__ == "__main__":
    args = parse_consolidate_feats_and_annot_args(sys.argv[1:])
    args_dict = convert_args_to_dict(args)
    consolidate_feats_and_annot(**args_dict)
