import argparse
import pandas as pd
import glob
import os
import sys
from typing import List

sys.path.append('.')
from src.usecase.utilities import convert_args_to_dict

ML_DATASET_OUTPUT_FOLDER = 'exports/ml_dataset'


def create_ml_dataset(input_folder: str,
                      output_folder: str = ML_DATASET_OUTPUT_FOLDER):

    consolidated_datasets = glob.glob(f'{input_folder}/**/*.csv',
                                      recursive=True)
    df_consolidated = pd.DataFrame(columns=["interval_index",
                                            "interval_start_time",
                                            "mean_nni",
                                            "sdnn",
                                            "sdsd",
                                            "nni_50",
                                            "pnni_50",
                                            "nni_20",
                                            "pnni_20",
                                            "rmssd",
                                            "median_nni",
                                            "range_nni",
                                            "cvsd", "cvnni",
                                            "mean_hr",
                                            "max_hr",
                                            "min_hr",
                                            "std_hr",
                                            "lf",
                                            "hf",
                                            "vlf",
                                            "lf_hf_ratio",
                                            "csi",
                                            "cvi",
                                            "Modified_csi",
                                            "sampen",
                                            "sd1",
                                            "sd2",
                                            "ratio_sd2_sd1",
                                            "label"])

    for consolidated_dataset in consolidated_datasets:
        df_temp = pd.read_csv(consolidated_dataset)
        df_consolidated = df_consolidated.append(df_temp)

    df_consolidated.reset_index(drop=True, inplace=True)
    os.makedirs(output_folder, exist_ok=True)
    df_consolidated.to_csv(f'{output_folder}/df_ml.csv', index=False)

    print(f'Size of output dataset: {df_consolidated.shape[0]}')


def parse_create_ml_dataset_args(
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
    parser.add_argument('--input-folder',
                        dest='input_folder',
                        help='input for for features to consolidate')
    parser.add_argument('--output-folder',
                        dest='output_folder',
                        help='output folder for ml dataset')
    args = parser.parse_args(args_to_parse)

    return args


if __name__ == '__main__':
    args = parse_create_ml_dataset_args(sys.argv[1:])
    args_dict = convert_args_to_dict(args)
    create_ml_dataset(**args_dict)
