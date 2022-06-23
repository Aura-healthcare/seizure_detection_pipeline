"""
This script is used to create a consolidated Machine Learning dataset.

Copyright (C) 2021 Association Aura
SPDX-License-Identifier: GPL-3.0
"""
import argparse
import pandas as pd
import glob
import os
import sys
from typing import List

sys.path.append(".")
from src.usecase.utilities import convert_args_to_dict
from src.usecase.compute_hrvanalysis_features import FEATURES_KEY_TO_INDEX

ML_DATASET_OUTPUT_FOLDER = "exports/ml_dataset"


def create_ml_dataset(
    input_folder: str, output_folder: str = ML_DATASET_OUTPUT_FOLDER
) -> str:
    """
    Merge individual dataset in a single large dataset.

    Parameters
    ----------
    input_folder : str
        Path of folder including datasets to merge
    output_folder : str
        Path of the output folder

    Returns
    -------
    output_file_path :
        Output file of the consolidated dataset
    """
    consolidated_datasets = glob.glob(f"{input_folder}/**/*.csv", recursive=True)
    df_consolidated = pd.read_csv(consolidated_datasets[0])

    for consolidated_dataset in consolidated_datasets[1:]:
        df_temp = pd.read_csv(consolidated_dataset)
        df_consolidated = df_consolidated.append(df_temp)

    df_consolidated.reset_index(drop=True, inplace=True)
    os.makedirs(output_folder, exist_ok=True)
    output_file_path = f"{output_folder}/df_ml.csv"
    df_consolidated.to_csv(output_file_path, index=False)

    print(f"Size of output dataset: {df_consolidated.shape[0]}")

    return output_file_path


def parse_create_ml_dataset_args(args_to_parse: List[str]) -> argparse.Namespace:
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
        "--input-folder",
        dest="input_folder",
        help="input for for features to consolidate",
    )
    parser.add_argument(
        "--output-folder", dest="output_folder", help="output folder for ml dataset"
    )
    args = parser.parse_args(args_to_parse)

    return args


if __name__ == "__main__":
    args = parse_create_ml_dataset_args(sys.argv[1:])
    args_dict = convert_args_to_dict(args)
    create_ml_dataset(**args_dict)
