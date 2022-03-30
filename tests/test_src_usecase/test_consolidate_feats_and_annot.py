import argparse
import pandas as pd
import pytest
from src.usecase.consolidate_feats_and_annot import \
    consolidate_feats_and_annot,\
    get_label_on_interval,\
    read_tse_bi,\
    parse_consolidate_feats_and_annot_args
from src.usecase.utilities import convert_args_to_dict

OUTPUT_FOLDER = 'tests/output/features'

TUH_FEATURES_FILE_PATH = 'data/test_data/tuh_feats_00009578_s006_t001.csv'
TUH_ANNOTATIONS_FILE_PATH = \
    'data/tuh/dev/01_tcp_ar/002/00009578/00009578_s006_t001.tse_bi'
TUH_ANNOTATIONS_FILE_PATH_INCORRECT = \
    'data/test_data/tuh_rr_00007633_s003_t007.csv'

DATASET_FEATUERS_FILE_PATH = \
    'data/test_data/dataset_feats_PAT_0_Annotations_EEG_0_s2.csv'
DATASET_ANNOTATIONS_FILE_PATH = \
    'data/dataset/PAT_0/PAT_0_Annotations_EEG_0.tse_bi'

WINDOW_INTERVAL = 10_000
SEGMENT_SIZE_TRESHOLD = 0.9
CROPPED_DATASET = True
TUH_NO_BCKG_TSE_BI_PATH = \
    'data/test_data/tuh_no_bckg_00009578_s006_t001.tse_bi'


def generate_consolidated_features_and_annot(features_file_path: str,
                                             annotations_file_path: str,
                                             cropped: bool):
    returned_path = consolidate_feats_and_annot(
        features_file_path=features_file_path,
        annotations_file_path=annotations_file_path,
        output_folder=OUTPUT_FOLDER,
        window_interval=WINDOW_INTERVAL,
        segment_size_treshold=SEGMENT_SIZE_TRESHOLD,
        crop_dataset=cropped)

    return returned_path


@pytest.fixture
def tuh_consolidated_cropped_dataset(
        features_file_path: str = TUH_FEATURES_FILE_PATH,
        annotations_file_path: str = TUH_ANNOTATIONS_FILE_PATH,
        cropped: bool = True):
    returned_path = generate_consolidated_features_and_annot(
        features_file_path=features_file_path,
        annotations_file_path=annotations_file_path,
        cropped=cropped)

    return returned_path


@pytest.fixture
def tuh_consolidated_uncropped_dataset(
        features_file_path: str = TUH_FEATURES_FILE_PATH,
        annotations_file_path: str = TUH_ANNOTATIONS_FILE_PATH,
        cropped: bool = False):
    returned_path = generate_consolidated_features_and_annot(
        features_file_path=features_file_path,
        annotations_file_path=annotations_file_path,
        cropped=cropped)
    return returned_path


@pytest.fixture
def dataset_consolidated_uncropped_dataset(
        features_file_path: str = DATASET_FEATUERS_FILE_PATH,
        annotations_file_path: str = DATASET_ANNOTATIONS_FILE_PATH,
        cropped: bool = False):
    returned_path = generate_consolidated_features_and_annot(
        features_file_path=features_file_path,
        annotations_file_path=annotations_file_path,
        cropped=cropped)
    return returned_path


def test_tuh_consolidate_feats_and_annot_cropped(
        tuh_consolidated_cropped_dataset):

    returned_path = tuh_consolidated_cropped_dataset
    assert(type(returned_path) == str)

    df_features = pd.read_csv(TUH_FEATURES_FILE_PATH)
    df_export = pd.read_csv(returned_path)
    assert(df_export.shape == (df_features.shape[0]-2, df_features.shape[1]+1))


def test_tuh_consolidate_feats_and_annot_uncropped(
        tuh_consolidated_uncropped_dataset):

    returned_path = tuh_consolidated_uncropped_dataset
    assert(type(returned_path) == str)

    df_features = pd.read_csv(TUH_FEATURES_FILE_PATH)
    df_export = pd.read_csv(returned_path)
    assert(df_export.shape == (df_features.shape[0], df_features.shape[1]+1))


def test_dataset_consolidate_feats_and_annot_uncropped(
        dataset_consolidated_uncropped_dataset):

    returned_path = dataset_consolidated_uncropped_dataset
    assert(type(returned_path) == str)

    df_features = pd.read_csv(DATASET_FEATUERS_FILE_PATH)
    df_export = pd.read_csv(returned_path)
    assert(df_export.shape == (df_features.shape[0], df_features.shape[1]+1))


def test_get_label_on_interval_no_bckg_exception():
    df_tse_bi = read_tse_bi(TUH_NO_BCKG_TSE_BI_PATH)
    try:
        get_label_on_interval(
            df_tse_bi=df_tse_bi,
            interval_start_time=0,
            window_interval=WINDOW_INTERVAL,
            segment_size_treshold=SEGMENT_SIZE_TRESHOLD)
    except KeyError:
        assert True


def test_tuh_parse_consolidate_feats_and_annot_args():

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


def test_get_label_on_interval():
    pass


def test_input_read_tse_bi_test():

    read_tse_bi(TUH_ANNOTATIONS_FILE_PATH)
    assert True

    read_tse_bi(DATASET_ANNOTATIONS_FILE_PATH)
    assert True

    try:
        read_tse_bi(TUH_ANNOTATIONS_FILE_PATH_INCORRECT)
    except ValueError:
        assert True
