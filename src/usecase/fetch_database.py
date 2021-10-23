import click
import pandas as pd
import subprocess
import re
from typing import Tuple
import os

# TUH database example
TUH_DATA_FILE_PATTERN = "*.edf"
TUH_ANNOTATIONS_FILE_PATTERN = "*.tse_bi"
TUH_PATIENT_PATTERN = ".+\\/(.+)_.+_.+\\..+"
TUH_EXAM_PATTERN = ".+\\/(.+)\\..+"
TUH_ANNOTATOR_PATTERN = ""

def fetch_database(data_folder_path: str,
         export_folder: str,
         data_file_pattern: str = TUH_DATA_FILE_PATTERN,
         patient_pattern: str = TUH_PATIENT_PATTERN,
         exam_pattern: str = TUH_EXAM_PATTERN,
         annotations_file_pattern: str = TUH_ANNOTATIONS_FILE_PATTERN,
         annotator_pattern: str = TUH_ANNOTATOR_PATTERN) -> Tuple[str, str]:

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
                                 data_file_pattern), stdout=subprocess.PIPE)

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
                                         annotations_file_pattern), stdout=subprocess.PIPE)

    for line in iter(annotations_call.stdout.readline, b""):
        patient_id = ""
        exam_id = ""
        # annotator = ""

        patient_str = re.search(patient_pattern, line.decode("utf-8"))
        if patient_str is not None and len(patient_str.groups()) == 1:
            patient_id = patient_str.group(1)

        exam_str = re.search(exam_pattern, line.decode("utf-8"))
        if exam_str is not None and len(exam_str.groups()) == 1:
            exam_id = exam_str.group(1)

        # annotator_str = re.search(annotator_pattern, line.decode("utf-8"))
        # if annotator_str is not None and len(annotator_str.groups()) == 1:
        #     annotator_id = annotator_str.group(1)

        df_annotations = df_annotations.append({
         "data_file_path": line.decode("utf-8").strip(),
         "exam_id":  exam_id,
         "patient_id": patient_id,
         "annotator_id": ""}, ignore_index=True)

    if export_folder is not None:
        df_data.to_csv(f'{export_folder}/df_data.csv',
                       index=False)
        df_annotations.to_csv(f'{export_folder}/df_annotations.csv',
                              index=False)

        df_data.columns = ['edf_file_path'] + list(df_data.columns[1:])
        df_annotations.columns = ['annotations_file_path'] + list(
            df_annotations.columns[1:])
        df_candidates = df_data.merge(df_annotations,
                                      how='outer',
                                      on=['exam_id', 'patient_id'])

        df_candidates.to_csv(f'{export_folder}/df_candidates.csv',
                             index=False)
    print({"candidates": f'{export_folder}/df_candidates.csv', "annotations": f'{export_folder}/df_annotations.csv'})
    return {"candidates": f'{export_folder}/df_candidates.csv',
            "data": f'{export_folder}/df_data.csv',
            "annotations": f'{export_folder}/df_annotations.csv'}


@click.command()
@click.option('--data-folder-path', required=True)
@click.option('--export-folder', default=None, required=False)
@click.option('--data-file-pattern', default=TUH_DATA_FILE_PATTERN)
@click.option('--patient-pattern', default=TUH_PATIENT_PATTERN)
@click.option('--exam-pattern', default=TUH_EXAM_PATTERN)
@click.option('--annotations-file-pattern',
              default=TUH_ANNOTATIONS_FILE_PATTERN)
@click.option('--annotator-pattern', default=TUH_ANNOTATOR_PATTERN)

def main(data_folder_path: str,
         export_folder: str,
         data_file_pattern: str,
         patient_pattern: str,
         exam_pattern: str,
         annotations_file_pattern: str,
         annotator_pattern: str) -> None:
    _ = fetch_database(data_folder_path, export_folder, data_file_pattern, patient_pattern, exam_pattern, annotations_file_pattern, annotator_pattern)



if __name__ == "__main__":
    main()
