import pytest
import sys
from pandas.testing import assert_frame_equal
from src.usecase.constants import TempFeaturesOperation

sys.path.append('.')
from src.usecase.data_processing.time_series_processing import (compute_time_features, create_rolling_variables_given_feature, 
                                                                diff_operation_from_features, compute_diff_for_feature,
                                                                perform_op_on_features)
from src.usecase.data_processing.data_loading import get_dataset
from tests.conftest import DATASET_FILE_PATH_FEAT, LIST_TIME, LIST_FEAT, OPERATION_TYPE, LIST_PERIOD


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
    assert "dayOfWeek" in dataframe_with_seasonal_informations.columns
    assert "month" in dataframe_with_seasonal_informations.columns
    assert "hour" in dataframe_with_seasonal_informations.columns
    assert "minute" in dataframe_with_seasonal_informations.columns


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
    assert "dayOfWeek" not in dataframe_with_seasonal_informations.columns
    assert "month" not in dataframe_with_seasonal_informations.columns
    assert "hour" not in dataframe_with_seasonal_informations.columns
    assert "minute" not in dataframe_with_seasonal_informations.columns


def test_perform_operations_on_features_given_list_features_and_list_period_check_periods_features_are_created():
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
        operation=operation
    )

    # Then
    assert "mean_hr_p30" in df_with_operation_result.keys()
    assert "mean_nni_p30" not in dataframe_without_cols.keys()


def test_diff_operation_from_features_given_list_feat_check_diff_feature_created():
    # Given
    dataset_path = DATASET_FILE_PATH_FEAT
    col_to_drop = []
    list_feat = LIST_FEAT
    _, dataframe_without_cols = get_dataset(dataset_path, col_to_drop)

    # When
    df_with_diff_features = diff_operation_from_features(
                                dataframe=dataframe_without_cols, 
                                list_feat=list_feat)

    # Then
    assert "mean_hr_diff" in df_with_diff_features.columns
    assert "mean_nni_diff" not in df_with_diff_features.columns


def test_diff_operation_from_features_not_given_list_feat_check_dataframes_are_identic():
    # Given
    dataset_path = DATASET_FILE_PATH_FEAT
    col_to_drop = []
    list_feat = []
    _, dataframe_with_cols = get_dataset(dataset_path, col_to_drop)

    # When
    df_with_diff_features = diff_operation_from_features(
                                dataframe=dataframe_with_cols, 
                                list_feat=list_feat)

    # Then
    assert_frame_equal(dataframe_with_cols, df_with_diff_features)


def test_create_rolling_variables_given_feature_and_operation_check_all_periods_created():
    # Given
    dataset_path = DATASET_FILE_PATH_FEAT
    col_to_drop = []
    feature_name = "mean_hr"
    operation = TempFeaturesOperation.mean.name
    _, df_with_rolling_variables = get_dataset(dataset_path, col_to_drop)

    # When
    create_rolling_variables_given_feature(dataframe=df_with_rolling_variables, 
                                            feature=feature_name, operation=operation) 

    # Then
    assert "mean_hr_p30" in df_with_rolling_variables.columns
    assert "mean_hr_p60" in df_with_rolling_variables.columns
    assert "mean_hr_p120" in df_with_rolling_variables.columns


def test_compute_diff_for_feature_given_featuer_check_feature_diff_created():
    # Given
    dataset_path = DATASET_FILE_PATH_FEAT
    col_to_drop = []
    feature_name = "mean_hr"
    _, df_with_diff_variables = get_dataset(dataset_path, col_to_drop)

    # When
    compute_diff_for_feature(dataframe=df_with_diff_variables, feature=feature_name)

    # Then
    assert "mean_hr_diff" in df_with_diff_variables.columns


def test_perform_op_on_features_given_two_features_check_feature_period_are_created_for_both_features():
    # Given
    dataset_path = DATASET_FILE_PATH_FEAT
    col_to_drop = []
    list_features = ["mean_hr", "mean_nni"]
    _, df_with_period_variables = get_dataset(dataset_path, col_to_drop)
    # When
    perform_op_on_features(dataframe=df_with_period_variables, list_feat=list_features, operation=TempFeaturesOperation.mean.name)

    # Then
    assert "mean_hr_p30" in df_with_period_variables.columns
    assert "mean_nni_p30" in df_with_period_variables.columns