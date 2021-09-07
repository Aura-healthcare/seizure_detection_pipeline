import pandas as pd

from src.usecase.compute_hrvanalysis_features import compute_hrvanalysis_features


def test_compute_hrvanalysis_features_should_return_dataframe():
    # Given
    # When
    result = compute_hrvanalysis_features()

    # Then
    assert isinstance(result, pd.DataFrame)


def test_compute_hrvanalysis_features_should_return_a_dataframe_with_proper_shape():
    pass
