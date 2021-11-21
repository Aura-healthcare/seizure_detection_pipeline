import argparse
import os
import re
import sys
import subprocess
from typing import Tuple, List

import pandas as pd

sys.path.append('.')
from src.usecase.utilities import convert_args_to_dict

# TUH database example
TUH_DATA_FILE_PATTERN = "*.edf"
TUH_ANNOTATIONS_FILE_PATTERN = "*.tse_bi"
TUH_PATIENT_PATTERN = ".+\\/(.+)_.+_.+\\..+"
TUH_EXAM_PATTERN = ".+\\/(.+)\\..+"
TUH_ANNOTATOR_PATTERN = ""
EXPORT_FOLDER = "output/db"

DB = 'tuh'

def write_database(export_folder: str,
                   df_data: pd.DataFrame,
                   df_annotations: pd.DataFrame) -> Tuple[str, str, str]:

    os.makedirs(export_folder, exist_ok=True)
    data_path = f'{export_folder}/df_data.csv'
    annotation_path = f'{export_folder}/df_annotations.csv'
    candidates_path = f'{export_folder}/df_candidates.csv'

    df_data.to_csv(data_path, index=False, encoding="utf-8")
    df_annotations.to_csv(annotation_path, index=False, encoding="utf-8")

    df_data.columns = ['edf_file_path'] + list(df_data.columns[1:])
    df_annotations.columns = ['annotations_file_path'] + list(
        df_annotations.columns[1:])
    df_candidates = df_data.merge(df_annotations,
                                  how='outer',
                                  on=['exam_id', 'patient_id'])
    # annotations: parsing fichier - s*.edf + tse_bi

    df_candidates.drop(columns='annotations_file_path', inplace=True)
    # To improve
    df_candidates.drop(columns='patient_id', inplace=True)
    df_candidates.drop(columns='annotator_id', inplace=True)
    # Remove non NA
    df_candidates['annotations_file_path'] = df_candidates[
        'edf_file_path'].apply(lambda x: parse_tse_bi(x, db=DB))
    df_candidates = df_candidates.dropna()
    df_candidates.to_csv(candidates_path, index=False, encoding="utf-8")

    return data_path, annotation_path, candidates_path


def parse_tse_bi(x,
                 db: str = 'tuh'):
    try:
        if db == 'tuh':
            split_limit = re.search('[.][e][d][f]', x).start()
            tse_bi_file_path = '.'.join([x[:split_limit], 'tse_bi'])
        else:
            split_limit = re.search('[_][s]\w*[.][e][d][f]', x).start()
            tse_bi_file_path = '.'.join([x[:split_limit], 'tse_bi'])
    except:
        tse_bi_file_path = None

    return tse_bi_file_path


def fetch_database(
        data_folder_path: str,
        export_folder: str = EXPORT_FOLDER,
        data_file_pattern: str = TUH_DATA_FILE_PATTERN,
        patient_pattern: str = TUH_PATIENT_PATTERN,
        exam_pattern: str = TUH_EXAM_PATTERN,
        annotations_file_pattern: str = TUH_ANNOTATIONS_FILE_PATTERN
        ) -> Tuple[str, str, str]:

    # Creating pd.DataFrame with edf path/exam_id/patient_id
    df_data = pd.DataFrame(columns=["data_file_path",
                                    "exam_id",
                                    "patient_id"])

    data_call = subprocess.Popen(('find',
                                  '-L',
                                  data_folder_path,
                                  '-type',
                                  'f',
                                  '-iname',
                                  data_file_pattern),
                                 stdout=subprocess.PIPE)

    for line in iter(data_call.stdout.readline, b""):
        patient_id = ""
        exam_id = ""

        patient_str = re.search(patient_pattern, line.decode("utf-8"))
        if patient_str is not None and len(patient_str.groups()) == 1:
            patient_id = patient_str.group(1)
        exam_str = re.search(exam_pattern, line.decode("utf-8"))
        if exam_str is not None and len(exam_str.groups()) == 1:
            exam_id = exam_str.group(1)

        df_data = df_data.append({
         "data_file_path": line.decode("utf-8").strip(),
         "exam_id":  exam_id,
         "patient_id": patient_id}, ignore_index=True)

    # Creating pd.DataFrame with annotation file associated with edf
    df_annotations = pd.DataFrame(columns=["data_file_path",
                                           "exam_id",
                                           "patient_id",
                                           "annotator_id"])
    annotations_call = subprocess.Popen(('find',
                                         '-L',
                                         data_folder_path,
                                         '-type',
                                         'f',
                                         '-iname',
                                         annotations_file_pattern),
                                        stdout=subprocess.PIPE)

    for line in iter(annotations_call.stdout.readline, b""):
        patient_id = ""
        exam_id = ""

        patient_str = re.search(patient_pattern, line.decode("utf-8"))
        if patient_str is not None and len(patient_str.groups()) == 1:
            patient_id = patient_str.group(1)

        exam_str = re.search(exam_pattern, line.decode("utf-8"))
        if exam_str is not None and len(exam_str.groups()) == 1:
            exam_id = exam_str.group(1)

        df_annotations = df_annotations.append({
         "data_file_path": line.decode("utf-8").strip(),
         "exam_id":  exam_id,
         "patient_id": patient_id,
         "annotator_id": ""}, ignore_index=True)

    data_path, annotation_path, candidates_path = write_database(
        export_folder,
        df_data,
        df_annotations)

    return data_path, annotation_path, candidates_path


def parse_fetch_database_args(args_to_parse: List[str]) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description='CLI parameter input')
    parser.add_argument('--data-folder-path',
                        dest='data_folder_path',
                        required=True)
    parser.add_argument('--export-folder',
                        dest='export_folder')
    parser.add_argument('--data-file-pattern',
                        dest='data_file_pattern')
    parser.add_argument('--patient-pattern',
                        dest='patient_pattern')
    parser.add_argument('--exam-pattern',
                        dest='exam_pattern')
    parser.add_argument('--annotations-file-pattern',
                        dest='annotations_file_pattern')
    args = parser.parse_args(args_to_parse)

    return args


if __name__ == "__main__":

    args = parse_fetch_database_args(sys.argv[1:])
    args_dict = convert_args_to_dict(args)
    fetch_database(**args_dict)
