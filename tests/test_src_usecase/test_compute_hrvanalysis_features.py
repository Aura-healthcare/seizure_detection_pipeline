import pandas as pd

from src.usecase.compute_hrvanalysis_features import compute_hrvanalysis_features

OUTPUT_FOLDER = 'tests/output/features'

def test_compute_hrvanalysis_features_should_return_dataframe():
    rr_interval_file_path = 'tests/test_data/00009578_s002_t001.csv'
    result = compute_hrvanalysis_features(rr_intervals_file_path=rr_interval_file_path,
                                          output_folder=OUTPUT_FOLDER)
    assert(type(result) == str)


def test_compute_hrvanalysis_features_should_return_a_dataframe_with_proper_shape():
    pass
