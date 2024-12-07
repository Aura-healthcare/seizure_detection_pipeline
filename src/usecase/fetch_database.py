"""
Create the list of files to process in Airflow DAG.

After searching all candidates files matching a patter, create a csv
(df_candidates.csv) that lists all the files to process in the Airflow DAG.

copyright (c) 2022 association aura
spdx-license-identifier: gpl-3.0
"""
import argparse
import os
import re
import sys
import glob
from typing import List, Tuple

import pandas as pd

sys.path.append('.')
from src.usecase.utilities import convert_args_to_dict

EXPORT_FOLDER = "output/feats-v0_6"

DATA_FILE_PATTERN = "*.edf"
ANNOTATION_FILE_PATTERN = "*.tse_bi"

TUH_PATIENT_PATTERN = r".+\/(.+)_.+_.+\..+"
TUH_EXAM_PATTERN = r".+\/(.+)\..+"

DATASET_PATIENT_PATTERN = r"[P][A][T][_][0-9]*"
DATASET_EXAM_PATTERN = r"[E][E][G][_][0-9]*"


def create_df_from_file_pattern(
        data_folder_path: str,
        file_pattern: str,
        file_label: str,
        patient_pattern: str,
        exam_pattern: str) -> pd.DataFrame:
    """
    Create a pd.DataFrame listing all files according to a pattern.

    Create a dataset listing recursively all files in a folder matching
    a pattern, then extracting meta-information patient and exam ids according
    to patterns.

    parameters
    ----------
    data_folder_path : str
        The path the folder where files are searched
    file_pattern : str
        Pattern to search to list files candidates
    file_label : str
        Label to describe which king of file is searched
    patient_pattern : str
        Pattern to search to extract the patient id
    exam_pattern : str
        Pattern to search to extraction exam id

    returns
    -------
    df : pd.DataFrame
        A pd.DataFrame listing the paths to matcehd files, exam_id and
        patient_id
    """
    df = pd.DataFrame(columns=[f'{file_label}_file_path',
                               'exam_id',
                               'patient_id'])
    file_paths = glob.glob(
        os.path.join(data_folder_path,
                     f'**/{file_pattern}'),
        recursive=True)

    for file_path in file_paths:

        patient_str = re.search(patient_pattern, file_path)
        try:
            # TUH parsing
            patient_id = patient_str.group(1)
        except IndexError:
            # Dataset Parsing
            patient_id = patient_str.group(0)

        exam_str = re.search(exam_pattern, file_path)
        try:
            exam_id = exam_str.group(1)
        except IndexError:
            # Teppe
            exam_id = exam_str.group(0)

        df = df.append({
            f"{file_label}_file_path": file_path.strip(),
            "exam_id": exam_id,
            "patient_id": patient_id}, ignore_index=True)

    return df


def fetch_database(
        data_folder_path: str,
        export_folder: str = EXPORT_FOLDER,
        data_file_pattern: str = DATA_FILE_PATTERN,
        annotations_file_pattern: str = ANNOTATION_FILE_PATTERN,
        patient_pattern: str = TUH_PATIENT_PATTERN,
        exam_pattern: str = TUH_EXAM_PATTERN) -> None:
    """
    Create a csv of files to process in DAG Airflow.

    Create a dataset consolidating data files and corresponding annotations
    according to some patterns. It also extract the exam and patient ids.

    parameters
    ----------
    data_folder_path : str
        The path the folder where files are searched
    export_folder : str
        The folder where the dataframe is saved
    data_file_pattern : str
        Regex Pattern to match to search for data candidates
    annotations_file_pattern : str
        Regex Pattern to match to search for annotations candidates
    patient_pattern : str
        Regex Pattern to match to search for patient id
    exam_pattern : str
        Regex Pattern to match to search for exam id

    returns
    -------
    df : pd.DataFrame
        A pd.DataFrame listing the paths to matcehd files, exam_id and
        patient_id
    """
    df_data = create_df_from_file_pattern(
        data_folder_path=data_folder_path,
        file_pattern=data_file_pattern,
        file_label='data',
        patient_pattern=patient_pattern,
        exam_pattern=exam_pattern)

    df_annotations = create_df_from_file_pattern(
        data_folder_path=data_folder_path,
        file_pattern=annotations_file_pattern,
        file_label='annotations',
        patient_pattern=patient_pattern,
        exam_pattern=exam_pattern)

    os.makedirs(export_folder, exist_ok=True)
    df_candidates = df_data.merge(df_annotations,
                                  how='outer',
                                  on=['exam_id', 'patient_id'])

    output = os.path.join(export_folder, 'df_candidates.csv')
    df_candidates.to_csv(output,
                         index=False,
                         encoding="utf-8")


def infer_database(
        data_folder_path: str,
        pattern_to_match: str = DATASET_PATIENT_PATTERN,
        dataset_file_pattern: str = DATA_FILE_PATTERN) -> Tuple[str, str]:
    """
    Automatically infer if database is DATASET OR TUH.

    From a folder, checks if data file can be parsed according to dataset
    patient pattern. If the condition is true, returns patient and exam pattern
    for dataset, else for TUH.

    parameters
    ----------
    data_folder_path : str
        The path the folder where files are searched
    pattern_to_match : str
        Regex Pattern to match to search for patient id
    dataset_file_pattern : str
        Regex Pattern to match to search for data candidates

    returns
    -------
    patient_pattern : str
        Pattern to search to extract the patient id
    exam_pattern : str
        Pattern to search to extraction exam id
    """
    file_paths = glob.glob(
        os.path.join(data_folder_path,
                     f'**/{dataset_file_pattern}'),
        recursive=True)

    for file_path in file_paths:
        patient_str = re.search(pattern_to_match, file_path)
        if patient_str is not None:
            return DATASET_PATIENT_PATTERN, DATASET_EXAM_PATTERN

    return TUH_PATIENT_PATTERN, TUH_EXAM_PATTERN


def parse_fetch_database_args(args_to_parse: List[str]) -> argparse.Namespace:
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
    parser.add_argument('--data-folder-path',
                        dest='data_folder_path',
                        type=str,
                        required=True)
    parser.add_argument('--export-folder',
                        dest='export_folder',
                        type=str)
    parser.add_argument('--infer-database',
                        dest='infer_database',
                        action='store_true')
    parser.add_argument('--patient-pattern',
                        dest='patient_pattern',
                        type=str)
    parser.add_argument('--exam-pattern',
                        dest='exam_pattern',
                        type=str)
    parser.set_defaults(infer_database=False)
    args = parser.parse_args(args_to_parse)

    return args


if __name__ == "__main__":

    args = parse_fetch_database_args(sys.argv[1:])
    args_dict = convert_args_to_dict(args)
    if args_dict.pop('infer_database'):
        args_dict['patient_pattern'], args_dict['exam_pattern'] = \
            infer_database(data_folder_path=args_dict['data_folder_path'])
    fetch_database(**args_dict)
