import argparse
import os
import numpy as np
import pandas as pd
import pytest
from src.usecase.compute_hrvanalysis_features import \
    SLIDING_WINDOW, SHORT_WINDOW, MEDIUM_WINDOW, LARGE_WINDOW,\
    compute_hrvanalysis_features,\
    parse_compute_hrvanalysis_features_args,\
    compute_features,\
    FEATURES_KEY_TO_INDEX
from src.usecase.utilities import convert_args_to_dict

OUTPUT_FOLDER = 'tests/output/features'
RR_INTERVAL_FILE_PATH = 'data/test_data/rr_00007633_s003_t007.csv'

@pytest.fixture
def test_features_computer():

    df_rr_intervals = pd.read_csv(
        os.path.join(RR_INTERVAL_FILE_PATH),
        index_col=0)
    rr_intervals = df_rr_intervals['rr_interval'].values
    rr_timestamps = np.cumsum(rr_intervals)

    features_computer = compute_features(
        rr_timestamps=rr_timestamps,
        rr_intervals=rr_intervals,
        features_key_to_index=FEATURES_KEY_TO_INDEX,
        sliding_window=SLIDING_WINDOW,
        short_window=SHORT_WINDOW,
        medium_window=MEDIUM_WINDOW,
        large_window=LARGE_WINDOW)

    return features_computer


def test_compute_hrvanalysis_features_return_str():
    result = compute_hrvanalysis_features(
        rr_intervals_file_path=RR_INTERVAL_FILE_PATH,
        output_folder=OUTPUT_FOLDER)
    assert(type(result) == str)


def test_tuh_parse_detect_qrs_args():

    bash_command = (f'python3 src/usecase/detect_qrs.py '
                    f'--rr-intervals-file-path {RR_INTERVAL_FILE_PATH} '
                    f'--output-folder {OUTPUT_FOLDER} '
                    f'--short-window {SHORT_WINDOW} '
                    f'--sliding-window {SLIDING_WINDOW} '
                    f'--medium-window {MEDIUM_WINDOW} '
                    f'--large-window {LARGE_WINDOW} ')
    args_to_parse = bash_command.split()[2:]
    parser = parse_compute_hrvanalysis_features_args(args_to_parse)
    parser_dict = convert_args_to_dict(parser)

    correct_parser = argparse.Namespace(
       rr_intervals_file_path=RR_INTERVAL_FILE_PATH,
       output_folder=OUTPUT_FOLDER,
       short_window=SHORT_WINDOW,
       sliding_window=SLIDING_WINDOW,
       medium_window=MEDIUM_WINDOW,
       large_window=LARGE_WINDOW)
    correct_parser_dict = convert_args_to_dict(correct_parser)

    assert(parser_dict == correct_parser_dict)


def test_get_rr_intervals_on_window_value_error(test_features_computer):
    # test with unreasonable index

    try:
        test_features_computer.get_rr_intervals_on_window(
            index=-1,
            size='short')
    except ValueError:
        assert True

    try:
        test_features_computer.compute_time_domain_features(
            index=0,
            clean_rrs=['a'])

    except Exception:
        assert True
