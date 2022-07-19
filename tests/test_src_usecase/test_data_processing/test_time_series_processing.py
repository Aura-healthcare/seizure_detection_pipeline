import pytest
import sys

sys.path.append('.')
from tests.conftest import DATASET_FILE_PATH_FEAT, LIST_TIME, LIST_FEAT, OPERATION_TYPE, LIST_PERIOD
from src.usecase.data_processing.data_loading import get_dataset
from src.usecase.data_processing.time_series_processing import compute_time_features, perform_op_on_features


def test_compute_time_features_given_list_time():
    # Given
    dataset_path = DATASET_FILE_PATH_FEAT
    col_to_drop = []
    list_time = LIST_TIME
    _, dataframe_without_cols = get_dataset(dataset_path, col_to_drop)

    # When
    dataframe_with_seasonal_informations = compute_time_features(
                                                dataframe=dataframe_without_cols,
                                                list_time=list_time
                                            )

    # Then
    assert "dayOfWeek" in dataframe_with_seasonal_informations.keys()
    assert "month" in dataframe_with_seasonal_informations.keys()
    assert "hour" in dataframe_with_seasonal_informations.keys()
    assert "minute" in dataframe_with_seasonal_informations.keys()


def test_compute_time_feature_not_given_list_time():
    # Given
    dataset_path = DATASET_FILE_PATH_FEAT
    col_to_drop = []
    list_time = []
    _, dataframe_without_cols = get_dataset(dataset_path, col_to_drop)

    # When
    dataframe_with_seasonal_informations = compute_time_features(
                                                dataframe=dataframe_without_cols,
                                                list_time=list_time
                                            )

    # Then
    assert "dayOfWeek" not in dataframe_with_seasonal_informations.keys()
    assert "month" not in dataframe_with_seasonal_informations.keys()
    assert "hour" not in dataframe_with_seasonal_informations.keys()
    assert "minute" not in dataframe_with_seasonal_informations.keys()


def test_perform_operations_on_features_given_list_features_and_list_period():
    # Given
    dataset_path = DATASET_FILE_PATH_FEAT
    col_to_drop = []
    list_feat = LIST_FEAT
    operation = OPERATION_TYPE
    list_period = LIST_PERIOD
    _, dataframe_without_cols = get_dataset(dataset_path, col_to_drop)

    # When
    df_with_operation_result = perform_op_on_features(
                                dataframe=dataframe_without_cols.copy(),
                                list_feat=list_feat,
                                list_period=list_period,
                                operation=operation
                            )
    # Then
    assert "mean_hr_30" in df_with_operation_result.keys()
    assert "mean_hr_30" not in dataframe_without_cols.keys()