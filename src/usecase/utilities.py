"""
This file includes utilities useful for other scripts.

copyright (c) 2021 association aura
spdx-license-identifier: gpl-3.0
"""
import argparse
import os


def convert_args_to_dict(args: argparse.Namespace) -> dict:
    """
    Convert argparse arguments into a dictionnary.

    From an argparse Namespace, create a dictionnary with only inputed CLI
    arguments. Allows to use argparse with default values in functions.

    parameters
    ----------
    args : argparse.Namespace
        Arguments to parse

    returns
    -------
    args_dict : dict
        Dictionnary with only inputed arguments
    """
    args_dict = {
        argument[0]: argument[1]
        for argument
        in args._get_kwargs()
        if argument[1] is not None}

    return args_dict


def generate_output_path(input_file_path: str,
                         output_folder: str,
                         format: str) -> str:
    """
    Generate an output path.

    From an input file path, output folder and format, create a output file
    path with coherent naming.

    parameters
    ----------
    input_file_path : str
        The path of original file used for transformation before export
    output_folder : str
        Path of the output folder
    format: str
        Format of the output file, typically, csv or json

    returns
    -------
    output_file_path : str
        Path where to save the output file
    """
    os.makedirs(output_folder, exist_ok=True)
    output_file_parsing = input_file_path.split('/')[-1].split('.')[0]
    output_filename = f'{output_file_parsing}.{format}'
    output_file_path = os.path.join(output_folder, output_filename)

    return output_file_path
