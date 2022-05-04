"""
This script is used to test fetch_datase script.

Copyright (C) 2022 Association Aura
SPDX-License-Identifier: GPL-3.0
"""
import argparse
import os
import pandas as pd
import pytest
from src.usecase.apply_ecg_qc import \
    parse_model,\
    import_signal_data,\
    apply_ecg_qc,\
    parse_apply_ecg_qc_args,\
    ECG_QC_MODEL

from src.usecase.consolidate_feats_and_annot import read_tse_bi
from src.usecase.utilities import convert_args_to_dict

OUTPUT_FOLDER = 'tests/output/'

TUH_FEATURES_FILE_PATH = 'data/test_data/tuh_feats_00004671_s007_t000.csv'
TUH_ANNOTATIONS_FILE_PATH = \
    'data/tuh/dev/01_tcp_ar/046/00004671/' \
    's007_2012_08_04/00004671_s007_t000.tse_bi'
TUH_ANNOTATIONS_FILE_PATH_INCORRECT = \
    'data/test_data/tuh_rr_00007633_s003_t007.csv'

DATASET_FEATUERS_FILE_PATH = \
    'data/test_data/dataset_feats_PAT_0_Annotations_EEG_0_s2.csv'
DATASET_ANNOTATIONS_FILE_PATH = \
    'data/dataset/PAT_0/PAT_0_Annotations_EEG_0.tse_bi'

TUH_DATA_FOLDER = 'data/tuh'

TEST_TUH_EDF_FILENAME = \
    'data/tuh/dev/01_tcp_ar/002/00009578/00009578_s002_t001.edf'
TEST_TUH_EXAM_ID = '1234536'

FORMATTING_TUH = 'tuh'
FORMATTING_DATASET = 'dataset'

EDF_FILE_WIHTOUT_ECG_CHANNEL = 'data/test_data/ST7051J0-PSG.edf'


def test_parse_model_returns_correct_parameters():
    """From the model name, test the correct output."""
    assert(parse_model(ECG_QC_MODEL) ==
           ('models/rfc_normalized_2s.pkl',
            'rfc_normalized_2s',
            2,
            True))
    assert(parse_model('dtc_7s.pkl') ==
           ('models/dtc_7s.pkl',
            'dtc_7s',
            7,
            False))


def test_parse_model_ValueError_returns_length_chunk_9():
    """Test the ValueError handling."""
    assert(parse_model('non-standard-name.pkl') ==
           ('models/non-standard-name.pkl',
            'non-standard-name',
            9,
            False))


def test_import_signal_data():
    """Assert the correct import of signal data from edf file."""
    with pytest.raises(ValueError):
        import_signal_data(EDF_FILE_WIHTOUT_ECG_CHANNEL)


def test_apply_ecg_qc_tuh():
    """Assert full pipeline to create a tse_bi for dataset format."""
    output_file_path = apply_ecg_qc(qrs_file_path=TEST_TUH_EDF_FILENAME,
                                    formatting=FORMATTING_TUH,
                                    output_folder=OUTPUT_FOLDER,
                                    model=ECG_QC_MODEL)
    assert(output_file_path == 'tests/output/ecg_qc_00009578_s002_t001.tse_bi')
    df_tse_bi_tuh = read_tse_bi(output_file_path)
    df_target_tuh = pd.DataFrame(
        data={'start': [pd.Timestamp(0)],
              'end': [pd.Timestamp(18_000_000_000)],
              'annotation': ['noisy'],
              'probability': [1]})
    assert(df_tse_bi_tuh.equals(df_target_tuh))

    if os.path.isfile(output_file_path):
        os.remove(output_file_path)


def test_apply_ecg_qc_dataset():
    """Assert full pipeline to create a tse_bi for tuh format."""
    output_file_path = apply_ecg_qc(qrs_file_path=TEST_TUH_EDF_FILENAME,
                                    formatting=FORMATTING_DATASET,
                                    output_folder=OUTPUT_FOLDER,
                                    model=ECG_QC_MODEL)
    assert(output_file_path == 'tests/output/ecg_qc_00009578_s002_t001.tse_bi')
    df_tse_bi_tuh = read_tse_bi(output_file_path)
    df_target_tuh = pd.DataFrame(
        data={'start': [pd.Timestamp('2013-02-28 16:03:12')],
              'end': [pd.Timestamp('2013-02-28 16:03:29')],
              'annotation': ['noisy'],
              'probability': [1]})
    assert(df_tse_bi_tuh.equals(df_target_tuh))

    if os.path.isfile(output_file_path):
        os.remove(output_file_path)


def test_parse_apply_ec_gqc():
    """Test the parsing of arguments."""
    bash_command = (f'python3 src/usecase/detect_qrs.py '
                    f'--qrs-file-path {TEST_TUH_EDF_FILENAME} '
                    f'--output-folder {OUTPUT_FOLDER} '
                    f'--model {ECG_QC_MODEL} '
                    f'--formatting {FORMATTING_TUH}')
    args_to_parse = bash_command.split()[2:]
    parser = parse_apply_ecg_qc_args(args_to_parse)
    parser_dict = convert_args_to_dict(parser)

    correct_parser = argparse.Namespace(
        qrs_file_path=TEST_TUH_EDF_FILENAME,
        output_folder=OUTPUT_FOLDER,
        model=ECG_QC_MODEL,
        formatting=FORMATTING_TUH)
    correct_parser_dict = convert_args_to_dict(correct_parser)

    assert(parser_dict == correct_parser_dict)
