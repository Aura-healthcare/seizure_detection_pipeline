import pandas as pd
import pytest
import sys
from pandas.testing import assert_frame_equal

import numpy as np

sys.path.append('.')
from src.usecase.data_processing.data_loading import get_dataset
from src.usecase.data_processing.data_cleaning import (
    impute_nan_values_by_median,
    outlier_detection, remove_outlier,
    replace_infinite_values_by_nan)
from tests.conftest import (COL_TO_DROP, DATASET_FILE_PATH, THRESHOLD)

def test_replace_infinite_values_by_nan_when_data_has_infinite_values(dataframe):
    # Given
    expected_response = False

    # When
    dataframe = replace_infinite_values_by_nan(dataframe)
    actual_response = dataframe.isin([np.inf, -np.inf]).values.any()

    # Then
    assert actual_response == expected_response


def test_replace_infinite_values_by_nan_when_data_hasnt_infinite_values(dataframe):
    # Given
    expected_response = False
    dataframe_without_nan = dataframe.replace([np.inf, -np.inf], np.nan)

    # When
    returned_dataframe = replace_infinite_values_by_nan(dataframe_without_nan)
    actual_response = dataframe_without_nan.isin([np.inf, -np.inf]).values.any()

    # Then
    assert actual_response == expected_response
    assert_frame_equal(dataframe_without_nan, returned_dataframe)


def test_impute_values_by_median_when_inf_values_exist(dataframe):
    # Given
    expected_response = False
    col_to_drop = COL_TO_DROP
    dataframe_wihtout_col = dataframe.drop(col_to_drop, 1)

    # When
    X_imputed, y = impute_nan_values_by_median(dataframe_wihtout_col)
    actual_response = pd.DataFrame(X_imputed).isna().values.any()

    # Then
    assert actual_response == expected_response
    assert dataframe.shape[0] == y.shape[0]


def test_impute_values_by_median_when_inf_values_not_exist(dataframe):
    # Given
    expected_response = False
    col_to_drop = COL_TO_DROP
    dataframe_wihtout_col = dataframe.drop(col_to_drop, 1)
    dataframe_without_infinite = replace_infinite_values_by_nan(
        dataframe_wihtout_col)

    # When
    X_imputed, y = impute_nan_values_by_median(dataframe_without_infinite)
    actual_response = pd.DataFrame(X_imputed).isna().values.any()

    # Then
    assert actual_response == expected_response
    assert dataframe_without_infinite.shape[0] == y.shape[0]


def test_outlier_detection_when_dataframe_is_imputed(dataframe):
    # Given
    expected_response = False
    col_to_drop = COL_TO_DROP
    dataframe_wihtout_col = dataframe.drop(col_to_drop, 1)
    X_imputed, y = impute_nan_values_by_median(dataframe_wihtout_col)
    dataframe_imputed = pd.DataFrame(X_imputed)

    # When
    outlier_score = outlier_detection(X_imputed)
    impute_check_df = dataframe_imputed.isna().values.any()

    # Then
    assert 'score' in outlier_score
    assert impute_check_df == expected_response


def test_outlier_detection_when_dataframe_is_not_imputed():
    # Given
    dataset_path = DATASET_FILE_PATH
    col_to_drop = COL_TO_DROP
    _, dataframe = get_dataset(dataset_path, col_to_drop)

    # When
    X = dataframe.drop(['label'], axis=1)
    y = dataframe['label']

    # Then
    with pytest.raises(Exception) as excep:
        outlier_score = outlier_detection(X)


def test_remove_outlier_given_imputed_dataframe_whithout_nan():
    # Given
    dataset_path = DATASET_FILE_PATH
    col_to_drop = COL_TO_DROP
    threshold = THRESHOLD
    _, dataframe = get_dataset(dataset_path, col_to_drop)
    X_imputed, y = impute_nan_values_by_median(dataframe)
    dataframe_imputed = pd.DataFrame(X_imputed)
    outlier_score = outlier_detection(X_imputed)

    # When
    X_result_outlier, y_result_outlier = remove_outlier(
        dataframe_imputed,
        outlier_score,
        y,
        threshold
    )

    # Then
    assert X_result_outlier.shape[0] == dataframe_imputed.shape[0]
    assert y_result_outlier.shape[0] == y.shape[0]