"""
This script is used to test consolidate_feats_and_annot.py script.

Copyright (C) 2022 Association Aura
SPDX-License-Identifier: GPL-3.0
"""
import argparse
import os
import pandas as pd
import pytest
from src.usecase.consolidate_feats_and_annot import \
    consolidate_feats_and_annot,\
    get_label_on_interval,\
    read_tse_bi,\
    parse_consolidate_feats_and_annot_args
from src.usecase.utilities import convert_args_to_dict

OUTPUT_FOLDER = 'tests/output/features'

TUH_FEATURES_FILE_PATH = 'data/test_data/tuh_feats_00004671_s007_t000.csv'
TUH_ANNOTATIONS_FILE_PATH = \
    'data/tuh/dev/01_tcp_ar/046/00004671/' \
    's007_2012_08_04/00004671_s007_t000.tse_bi'
TUH_ANNOTATIONS_FILE_PATH_INCORRECT = \
    'data/test_data/tuh_rr_00007633_s003_t007.csv'
TUH_ANNOTATIONS_FILE_PATH_EMPTY = \
    'data/test_data/tuh_empty.tse_bi'


DATASET_FEATUERS_FILE_PATH = \
    'data/test_data/dataset_feats_PAT_0_Annotations_EEG_0_s2.csv'
DATASET_ANNOTATIONS_FILE_PATH = \
    'data/dataset/PAT_0/PAT_0_Annotations_EEG_0.tse_bi'

WINDOW_INTERVAL = 1_000
SEGMENT_SIZE_TRESHOLD = 0.9
CROPPED_DATASET = True
TUH_NO_BCKG_TSE_BI_PATH = \
    'data/test_data/tuh_no_bckg_00009578_s006_t001.tse_bi'

LABEL_TARGET = 'seiz'


def generate_consolidated_features_and_annot(features_file_path: str,
                                             annotations_file_path: str,
                                             cropped: bool) -> str:
    """Reusable function for consolidation generation."""
    return consolidate_feats_and_annot(
        features_file_path=features_file_path,
        annotations_file_path=annotations_file_path,
        output_folder=OUTPUT_FOLDER,
        window_interval=WINDOW_INTERVAL,
        segment_size_treshold=SEGMENT_SIZE_TRESHOLD,
        label_target=LABEL_TARGET,
        crop_dataset=cropped)


@pytest.fixture
def tuh_consolidated_cropped(
        features_file_path: str = TUH_FEATURES_FILE_PATH,
        annotations_file_path: str = TUH_ANNOTATIONS_FILE_PATH,
        cropped: bool = True):
    """Generate a cropped consolidated csv for tuh."""
    return generate_consolidated_features_and_annot(
        features_file_path=features_file_path,
        annotations_file_path=annotations_file_path,
        cropped=cropped)


@pytest.fixture
def tuh_consolidated_uncropped(
       features_file_path: str = TUH_FEATURES_FILE_PATH,
        annotations_file_path: str = TUH_ANNOTATIONS_FILE_PATH,
        cropped: bool = False):
    """Generate an uncropped consolidated csv for tuh."""
    return generate_consolidated_features_and_annot(
        features_file_path=features_file_path,
        annotations_file_path=annotations_file_path,
        cropped=cropped)


@pytest.fixture
def dataset_consolidated_uncropped(
        features_file_path: str = DATASET_FEATUERS_FILE_PATH,
        annotations_file_path: str = DATASET_ANNOTATIONS_FILE_PATH,
        cropped: bool = False):
    """Generate an uncropped consolidated csv for dataset."""
    return generate_consolidated_features_and_annot(
        features_file_path=features_file_path,
        annotations_file_path=annotations_file_path,
        cropped=cropped)


def test_tuh_consolidate_feats_and_annot_cropped(
        tuh_consolidated_cropped):
    """Test the shape of the csv is cropped for tuh."""
    df_features = pd.read_csv(TUH_FEATURES_FILE_PATH)
    df_export = pd.read_csv(tuh_consolidated_cropped)
    assert(df_export.shape == (df_features.shape[0]-2, df_features.shape[1]+1))

    if os.path.isfile(tuh_consolidated_cropped):
        os.remove(tuh_consolidated_cropped)


def test_tuh_consolidate_feats_and_annot_uncropped(
        tuh_consolidated_uncropped):
    """Test the shape of the csv is uncropped for tuh."""
    df_features = pd.read_csv(TUH_FEATURES_FILE_PATH)
    df_export = pd.read_csv(tuh_consolidated_uncropped)
    assert(df_export.shape == (df_features.shape[0], df_features.shape[1]+1))

    if os.path.isfile(tuh_consolidated_uncropped):
        os.remove(tuh_consolidated_uncropped)


def test_dataset_consolidate_feats_and_annot_uncropped(
        dataset_consolidated_uncropped):
    """Test the shape of the csv is uncropped for dataset."""
    df_features = pd.read_csv(DATASET_FEATUERS_FILE_PATH)
    df_export = pd.read_csv(dataset_consolidated_uncropped)
    assert(df_export.shape == (df_features.shape[0], df_features.shape[1]+1))

    if os.path.isfile(dataset_consolidated_uncropped):
        os.remove(dataset_consolidated_uncropped)


def test_tuh_parse_consolidate_feats_and_annot_args():
    """Test CLI usage of the script."""
    bash_command = (f'python3 src/usecase/detect_qrs.py '
                    f'--features-file-path {TUH_FEATURES_FILE_PATH} '
                    f'--annotations-file-path {TUH_ANNOTATIONS_FILE_PATH} '
                    f'--output-folder {OUTPUT_FOLDER} '
                    f'--window-interval {WINDOW_INTERVAL} '
                    f'--segment-size-treshold {SEGMENT_SIZE_TRESHOLD} '
                    f'--crop-dataset {CROPPED_DATASET}')
    args_to_parse = bash_command.split()[2:]
    parser = parse_consolidate_feats_and_annot_args(args_to_parse)
    parser_dict = convert_args_to_dict(parser)

    correct_parser = argparse.Namespace(
        features_file_path=TUH_FEATURES_FILE_PATH,
        annotations_file_path=TUH_ANNOTATIONS_FILE_PATH,
        output_folder=OUTPUT_FOLDER,
        window_interval=WINDOW_INTERVAL,
        segment_size_treshold=SEGMENT_SIZE_TRESHOLD,
        crop_dataset=CROPPED_DATASET)

    correct_parser_dict = convert_args_to_dict(correct_parser)

    assert(parser_dict == correct_parser_dict)


def test_tuh_get_label_on_interval():
    """Test label fetching."""
    intervals_to_test = [
        {'interval_start_time': 20_000, 'label': 0},
        {'interval_start_time': 50_000, 'label': 1},
        {'interval_start_time': 118_000, 'label': 0},
        {'interval_start_time': 168_000, 'label': 1},
        {'interval_start_time': 175_000, 'label': 0},
        {'interval_start_time': 200_000, 'label': 1}
    ]

    for interval_to_test in intervals_to_test:

        assert(get_label_on_interval(
            df_tse_bi=read_tse_bi(TUH_ANNOTATIONS_FILE_PATH),
            interval_start_time=interval_to_test['interval_start_time'],
            window_interval=WINDOW_INTERVAL,
            segment_size_treshold=SEGMENT_SIZE_TRESHOLD,
            label_target=LABEL_TARGET)
            == interval_to_test['label'])


def test_dataset_get_label_on_interval():
    """Test interval fetching."""
    intervals_to_test = [
        {'interval_start_time': '2017-01-10T14:30:05', 'label': 0},
        {'interval_start_time': '2017-01-10T14:32:30', 'label': 1},
        {'interval_start_time': '2017-01-10T14:32:41', 'label': 0},
        {'interval_start_time': '2017-01-10T14:38:45', 'label': 1},
        {'interval_start_time': '2017-01-10T14:38:55', 'label': 0},
    ]

    for interval_to_test in intervals_to_test:

        assert(get_label_on_interval(
            df_tse_bi=read_tse_bi(DATASET_ANNOTATIONS_FILE_PATH),
            interval_start_time=interval_to_test['interval_start_time'],
            window_interval=WINDOW_INTERVAL,
            segment_size_treshold=SEGMENT_SIZE_TRESHOLD,
            label_target=LABEL_TARGET)
            == interval_to_test['label'])


def test_input_read_tse_bi_test():
    """Test tse_bi files are correctly read."""
    df_tse_bi_tuh = read_tse_bi(TUH_ANNOTATIONS_FILE_PATH)
    df_target_tuh = pd.DataFrame(
        data={
            'start': [pd.Timestamp(0.0000 * 1_000_000_000),
                      pd.Timestamp(46.2734 * 1_000_000_000),
                      pd.Timestamp(52.0742 * 1_000_000_000),
                      pd.Timestamp(119.1406 * 1_000_000_000),
                      pd.Timestamp(125.8398 * 1_000_000_000)],
            'end': [pd.Timestamp(46.2734 * 1_000_000_000),
                    pd.Timestamp(52.0742 * 1_000_000_000),
                    pd.Timestamp(119.1406 * 1_000_000_000),
                    pd.Timestamp(125.8398 * 1_000_000_000),
                    pd.Timestamp(166.8203 * 1_000_000_000)],
            'annotation': ['bckg',
                           'seiz',
                           'bckg',
                           'seiz',
                           'bckg'],
            'probability': [1,
                            1,
                            1,
                            1,
                            1]})

    assert(df_tse_bi_tuh.iloc[:5].equals(df_target_tuh))

    df_tse_bi_dataset = read_tse_bi(DATASET_ANNOTATIONS_FILE_PATH)
    df_target_dataset = pd.DataFrame(
        data={
            'start': [pd.Timestamp('2017-01-10T14:30:00.441406'),
                      pd.Timestamp('2017-01-10T14:32:23.062500'),
                      pd.Timestamp('2017-01-10T14:32:40.015625'),
                      pd.Timestamp('2017-01-10T14:38:44.406250'),
                      pd.Timestamp('2017-01-10T14:38:50.468750')],
            'end': [pd.Timestamp('2017-01-10T14:32:23.062500'),
                    pd.Timestamp('2017-01-10T14:32:40.015625'),
                    pd.Timestamp('2017-01-10T14:38:44.406250'),
                    pd.Timestamp('2017-01-10T14:38:50.468750'),
                    pd.Timestamp('2017-01-10T14:40:42.832031')],
            'annotation': ['bckg',
                           'seiz',
                           'bckg',
                           'seiz',
                           'bckg'],
            'probability': [1,
                            1,
                            1,
                            1,
                            1]})
    assert(df_tse_bi_dataset.equals(df_target_dataset))

    # Change for better
    with pytest.raises(ValueError):
        read_tse_bi(TUH_ANNOTATIONS_FILE_PATH_INCORRECT)

    with pytest.raises(SystemExit) as e:
        read_tse_bi(TUH_ANNOTATIONS_FILE_PATH_EMPTY)
    assert e.type == SystemExit
    assert e.value.code == 'tse_bi file is empty'
