"""
This script is used to test fetch_datase script.

Copyright (C) 2022 Association Aura
SPDX-License-Identifier: GPL-3.0
"""
import argparse
import pandas as pd
import pytest
from src.usecase.fetch_database import \
    ANNOTATION_FILE_PATTERN,\
    parse_fetch_database_args,\
    infer_database,\
    create_df_from_file_pattern,\
    fetch_database,\
    TUH_EXAM_PATTERN,\
    TUH_PATIENT_PATTERN,\
    DATASET_EXAM_PATTERN,\
    DATASET_PATIENT_PATTERN,\
    DATA_FILE_PATTERN

from pathlib import Path
import shutil
import os
from src.usecase.utilities import convert_args_to_dict

OUTPUT_FOLDER = 'tests/output/features'

#  TUH_FEATURES_FILE_PATH = 'data/test_data/tuh_feats_00009578_s006_t001.csv'
TUH_FEATURES_FILE_PATH = 'data/test_data/tuh_feats_00004671_s007_t000.csv'
TUH_ANNOTATIONS_FILE_PATH = \
    'data/tuh/dev/01_tcp_ar/046/00004671/' \
    's007_2012_08_04/00004671_s007_t000.tse_bi'
#  'data/tuh/dev/01_tcp_ar/002/00009578/00009578_s006_t001.tse_bi'
TUH_ANNOTATIONS_FILE_PATH_INCORRECT = \
    'data/test_data/tuh_rr_00007633_s003_t007.csv'

DATASET_FEATUERS_FILE_PATH = \
    'data/test_data/dataset_feats_PAT_0_Annotations_EEG_0_s2.csv'
DATASET_ANNOTATIONS_FILE_PATH = \
    'data/dataset/PAT_0/PAT_0_Annotations_EEG_0.tse_bi'

TUH_DATA_FOLDER = 'data/tuh'
EXPORT_FOLDER = 'tests/output'

DATASET_DATA_SOURCE = Path('data/dataset')
PATH_TEST_TEMP = Path('tests/temp')


@pytest.fixture
def create_mock_dataset_sample(path_data_source: Path = DATASET_DATA_SOURCE,
                               path_test: Path = PATH_TEST_TEMP):
    """Initialize mock dataset tree structure sample."""
    path_data = os.path.join(path_test, 'dataset')

    if os.path.exists(path_test):
        shutil.rmtree(path_test)
    shutil.copytree(path_data_source,
                    path_data)
    os.mknod(os.path.join(path_data,
                          'PAT_0',
                          'EEG_0_s1.edf'))

    yield

    if os.path.exists(path_test):
        shutil.rmtree(path_test)


def test_parse_fetch_database_args_with_infer_database():
    """Test the fetch_database parsing with inferance of database."""
    # With inference
    bash_command = (f'python3 src/usecase/fetch_database.py '
                    f'--data-folder-path {TUH_DATA_FOLDER} '
                    f'--export-folder {EXPORT_FOLDER} '
                    f'--infer-database '
                    f'--patient-pattern {TUH_PATIENT_PATTERN} '
                    f'--exam-pattern {TUH_EXAM_PATTERN}')

    args_to_parse = bash_command.split()[2:]
    parser = parse_fetch_database_args(args_to_parse)
    parser_dict = convert_args_to_dict(parser)

    correct_parser = argparse.Namespace(
        data_folder_path=TUH_DATA_FOLDER,
        export_folder=EXPORT_FOLDER,
        infer_database=True,
        patient_pattern=TUH_PATIENT_PATTERN,
        exam_pattern=TUH_EXAM_PATTERN)
    correct_parser_dict = convert_args_to_dict(correct_parser)

    assert(parser_dict == correct_parser_dict)


def test_parse_fetch_database_args_without_infer_database():
    """Test the fetch_database parsing without inferance of database."""
    bash_command = (f'python3 src/usecase/fetch_database.py '
                    f'--data-folder-path {TUH_DATA_FOLDER} '
                    f'--export-folder {EXPORT_FOLDER} '
                    f'--patient-pattern {TUH_PATIENT_PATTERN} '
                    f'--exam-pattern {TUH_EXAM_PATTERN}')

    args_to_parse = bash_command.split()[2:]
    parser = parse_fetch_database_args(args_to_parse)
    parser_dict = convert_args_to_dict(parser)

    correct_parser = argparse.Namespace(
        data_folder_path=TUH_DATA_FOLDER,
        export_folder=EXPORT_FOLDER,
        infer_database=False,
        patient_pattern=TUH_PATIENT_PATTERN,
        exam_pattern=TUH_EXAM_PATTERN)
    correct_parser_dict = convert_args_to_dict(correct_parser)

    assert(parser_dict == correct_parser_dict)


def test_infer_database_tuh():
    """Infer database is tuh."""
    assert(infer_database(
        data_folder_path=TUH_DATA_FOLDER,
        pattern_to_match=DATASET_PATIENT_PATTERN,
        dataset_file_pattern=DATA_FILE_PATTERN) == (
            TUH_PATIENT_PATTERN, TUH_EXAM_PATTERN))


def test_infer_database_dataset(create_mock_dataset_sample):
    """Infer database is dataset."""
    assert(infer_database(
        data_folder_path=PATH_TEST_TEMP,
        pattern_to_match=DATASET_PATIENT_PATTERN,
        dataset_file_pattern=DATA_FILE_PATTERN) == (
            DATASET_PATIENT_PATTERN, DATASET_EXAM_PATTERN))


def test_dataset_create_df_from_file_pattern(create_mock_dataset_sample):
    """Test function create_df_from_file_pattern with dataset."""
    df = create_df_from_file_pattern(
        data_folder_path=PATH_TEST_TEMP,
        file_pattern=DATA_FILE_PATTERN,
        file_label='data',
        patient_pattern=DATASET_PATIENT_PATTERN,
        exam_pattern=DATASET_EXAM_PATTERN)
    df_target = pd.DataFrame(
        data={'data_file_path': ['tests/temp/dataset/PAT_0/EEG_0_s1.edf'],
              'exam_id': ['EEG_0'],
              'patient_id': ['PAT_0']})
    assert(df.shape == (1, 3))
    assert(df.equals(df_target))


def test_tuh_create_df_from_file_pattern():
    """Test function create_df_from_file_pattern with TUH."""
    df = create_df_from_file_pattern(
        data_folder_path=TUH_DATA_FOLDER,
        file_pattern=DATA_FILE_PATTERN,
        file_label='data',
        patient_pattern=TUH_PATIENT_PATTERN,
        exam_pattern=TUH_EXAM_PATTERN)

    df_target = pd.DataFrame(
        data={'data_file_path': ['data/tuh/dev/01_tcp_ar/002/00009578/'
                                 '00009578_s002_t001.edf',
                                 'data/tuh/dev/01_tcp_ar/002/00009578/'
                                 '00009578_s006_t001.edf',
                                 'data/tuh/dev/01_tcp_ar/076/00007633/'
                                 's003_2013_07_09/00007633_s003_t007.edf',
                                 'data/tuh/dev/01_tcp_ar/046/00004671/'
                                 's007_2012_08_04/00004671_s007_t000.edf'],
              'exam_id': ['00009578_s002_t001',
                          '00009578_s006_t001',
                          '00007633_s003_t007',
                          '00004671_s007_t000'],
              'patient_id': ['00009578',
                             '00009578',
                             '00007633',
                             '00004671']})
    assert(df.equals(df_target))


def test_dataset_fetch_database(create_mock_dataset_sample):
    """Test function fech_database with dataset."""
    fetch_database(data_folder_path=PATH_TEST_TEMP,
                   export_folder=PATH_TEST_TEMP,
                   data_file_pattern=DATA_FILE_PATTERN,
                   annotations_file_pattern=ANNOTATION_FILE_PATTERN,
                   patient_pattern=DATASET_PATIENT_PATTERN,
                   exam_pattern=DATASET_EXAM_PATTERN)
    df = pd.read_csv(os.path.join(PATH_TEST_TEMP, 'df_candidates.csv'))
    df_target = pd.DataFrame(
        data={'data_file_path': ['tests/temp/dataset/PAT_0/EEG_0_s1.edf'],
              'exam_id': ['EEG_0'],
              'patient_id': ['PAT_0'],
              'annotations_file_path': [
                  'tests/temp/dataset/PAT_0/PAT_0_Annotations_EEG_0.tse_bi']})
    print(df.head())
    print(df_target.head())
    assert(df.equals(df_target))


def test_tuh_fetch_database():
    """Test function fech_database with TUH."""
    fetch_database(data_folder_path=TUH_DATA_FOLDER,
                   export_folder=EXPORT_FOLDER,
                   data_file_pattern=DATA_FILE_PATTERN,
                   annotations_file_pattern=ANNOTATION_FILE_PATTERN,
                   patient_pattern=TUH_PATIENT_PATTERN,
                   exam_pattern=TUH_EXAM_PATTERN)
    df = pd.read_csv(os.path.join(EXPORT_FOLDER, 'df_candidates.csv'))
    df_target = pd.DataFrame(
        data={'data_file_path': ['data/tuh/dev/01_tcp_ar/002/00009578/'
                                 '00009578_s002_t001.edf',
                                 'data/tuh/dev/01_tcp_ar/002/00009578/'
                                 '00009578_s006_t001.edf',
                                 'data/tuh/dev/01_tcp_ar/076/00007633/'
                                 's003_2013_07_09/00007633_s003_t007.edf',
                                 'data/tuh/dev/01_tcp_ar/046/00004671/'
                                 's007_2012_08_04/00004671_s007_t000.edf'],
              'exam_id': ['00009578_s002_t001',
                          '00009578_s006_t001',
                          '00007633_s003_t007',
                          '00004671_s007_t000'],
              'patient_id': [9578,
                             9578,
                             7633,
                             4671],
              'annotations_file_path': [
                  'data/tuh/dev/01_tcp_ar/002/00009578/'
                  '00009578_s002_t001.tse_bi',
                  'data/tuh/dev/01_tcp_ar/002/00009578/'
                  '00009578_s006_t001.tse_bi',
                  'data/tuh/dev/01_tcp_ar/076/00007633/'
                  's003_2013_07_09/00007633_s003_t007.tse_bi',
                  'data/tuh/dev/01_tcp_ar/046/00004671/'
                  's007_2012_08_04/00004671_s007_t000.tse_bi']})
    assert(df.equals(df_target))
