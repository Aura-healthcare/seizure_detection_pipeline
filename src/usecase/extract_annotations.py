"""
This script is used to extract annotations from a tse_bi file.

copyright (c) 2021 association aura
spdx-license-identifier: gpl-3.0
"""
import argparse
import json
import sys
from typing import List
sys.path.append('.')
from src.usecase.utilities import convert_args_to_dict, generate_output_path

SEIZURE_TAG = "seiz"
BACKGROUND_TAG = "bckg"


def extract_annotations(annotations_file_path: str,
                        output_folder: str,
                        seizure_tag: str = SEIZURE_TAG,
                        background_tag: str = BACKGROUND_TAG) -> str:
    """
    From a tse_bi file, export a json including binary seizure labels.

    From an annotations file path, output folder and binary tags for seizure or
    background activities, extract annotations by nature with start and
    end timestamps and export them in json format.

    parameters
    ----------
    annotations_file_path : str
        The path of annotations file
    output_folder : str
        Path of the output folder
    seizure_tag : str
        Tag of seizure tag, for example "seiz"
    background_tag : str
        Tag of background tag, for example "seiz"

    returns
    -------
    output_file_path : str
        Path where json file is saved
    """
    if not annotations_file_path.endswith('.tse_bi'):
        raise ValueError(f'*.tse_bi required, input: {annotations_file_path}')
        exit()

    background_intervals = []
    seizure_intervals = []

    with open(annotations_file_path, "r") as f:
        for line in f:
            tokens = line.split(" ")
            if(len(tokens) == 4):
                # Mutiplication by 1000 for output in milliseconds
                if tokens[2] == seizure_tag:
                    seizure_intervals.append([float(tokens[0]) * 1_000,
                                              float(tokens[1])] * 1_000)
                elif tokens[2] == background_tag:
                    background_intervals.append([float(tokens[0]) * 1_000,
                                                 float(tokens[1]) * 1_000])

    data = {"background": background_intervals,
            "seizure": seizure_intervals}

    output_file_path = generate_output_path(
        input_file_path=annotations_file_path,
        output_folder=output_folder,
        format='json')

    with open(output_file_path, "w") as out_f:
        json.dump(data, out_f)

    return output_file_path


def parse_extract_annotations_args(
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
    parser.add_argument('--annotations-file-path',
                        dest='annotations_file_path',
                        required=True)
    parser.add_argument('--output-folder',
                        dest='output_folder',
                        required=True)
    parser.add_argument('--seizure-tag',
                        dest='seizure_tag')
    parser.add_argument('--background-tag',
                        dest='background_tag')
    args = parser.parse_args(args_to_parse)

    return args


if __name__ == '__main__':

    args = parse_extract_annotations_args(sys.argv[1:])
    args_dict = convert_args_to_dict(args)
    extract_annotations(**args_dict)
