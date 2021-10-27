import pandas as pd

from src.usecase.compute_hrvanalysis_features import compute_hrvanalysis_features

RR_INTERVAL_FILEPATH = 'tests/test_data/00009578_s002_t001.csv'

def test_compute_hrvanalysis_features_should_return_dataframe():
    # Given
    # When
    result = compute_hrvanalysis_features(RR_INTERVAL_FILEPATH)

    # Then
    assert(type(result) == str)


def test_compute_hrvanalysis_features_should_return_a_dataframe_with_proper_shape():
    pass
