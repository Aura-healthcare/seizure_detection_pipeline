import sys
from matplotlib.pyplot import get
import pytest
from pandas.testing import assert_frame_equal

import pandas as pd
import numpy as np
from sklearn import datasets
from sqlalchemy import column

sys.path.append('.')
DATASET_FILE_PATH = "data/test_data/test_data_feat_eng.csv"
DATASET_FILE_PATH_FEAT = "data/test_data/test_data_feat_contextual.csv"

from src.usecase.feature_engineering import (
                                                get_dataset,
                                                create_final_dataset_with_pca,
                                                impute_nan_values_by_median,
                                                outlier_detection,
                                                pca_analysis,
                                                perform_op_on_features,
                                                prepare_features,
                                                prepare_features_with_io,
                                                replace_infinite_values_by_nan,
                                                remove_outlier,
                                                computeContextFeatures,
                                                computeSeasonerFeatures,
                                                diffOperationFromFeatures
                                            )

def test_get_dataset(dataframe):
    # Given
    dataset_path = DATASET_FILE_PATH
    col_to_drop = ['timestamp', "set"]
    expected_reponse = False

    # When
    df_init, df_copy = get_dataset(dataset_path, col_to_drop)
    dataframe = dataframe.drop(col_to_drop, axis=1)

    #Then
    assert (df_init.empty == expected_reponse and df_copy.empty == expected_reponse)
    assert_frame_equal(df_copy, dataframe)


def test_replace_infinite_values_by_nan(dataframe):
    # Given
    expected_response = True

    # When
    dataframe = replace_infinite_values_by_nan(dataframe)

    #Then
    assert dataframe.isin([np.inf, -np.inf]).values.any() != expected_response


def test_impute_values_by_median_when_inf_values_exist(dataframe):
    # Given
    expected_response = True
    col_to_drop = ['timestamp', 'set']

    # When
    dataframe = dataframe.drop(col_to_drop, 1)
    X_imputed, y = impute_nan_values_by_median(dataframe)
    dataframe_imputed = pd.DataFrame(X_imputed)

    # Then
    assert dataframe_imputed.isna().values.any() != expected_response
    assert dataframe_imputed.shape[0] == y.shape[0]


def test_impute_values_by_median_when_inf_values_not_exist(dataframe):
    # Given
    expected_response = False
    col_to_drop = ['timestamp', 'set']

    # When
    dataframe = dataframe.drop(col_to_drop, 1)
    dataframe = replace_infinite_values_by_nan(dataframe)
    X_imputed, y = impute_nan_values_by_median(dataframe)
    dataframe_imputed = pd.DataFrame(X_imputed)

    # Then
    assert dataframe_imputed.isna().values.any() == expected_response
    assert dataframe_imputed.shape[0] == y.shape[0]


def test_outlier_detection_when_dataframe_is_imputed(dataframe):
    # Given
    expected_response = False
    col_to_drop = ['timestamp', 'set']

    # When
    dataframe = dataframe.drop(col_to_drop, 1)
    X_imputed, y = impute_nan_values_by_median(dataframe)
    dataframe_imputed = pd.DataFrame(X_imputed)

    outlier_score = outlier_detection(X_imputed)    

    # Then
    assert 'score' in outlier_score
    assert dataframe_imputed.isna().values.any() == expected_response


def test_outlier_detection_when_dataframe_is_not_imputed():
    # Given
    dataset_path = DATASET_FILE_PATH
    col_to_drop = ['timestamp', 'set']

    # When
    _, dataframe = get_dataset(dataset_path, col_to_drop)
    X = dataframe.drop(['label'], axis=1)
    y = dataframe['label']

    # Then
    with pytest.raises(Exception) as excep:
        outlier_score = outlier_detection(X)

def test_remove_outlier():
    # Given
    dataset_path = DATASET_FILE_PATH
    col_to_drop = ['timestamp', 'set']
    threshold = -1.5

    # When
    _, dataframe = get_dataset(dataset_path, col_to_drop)
    X_imputed, y = impute_nan_values_by_median(dataframe)
    dataframe_imputed = pd.DataFrame(X_imputed)

    outlier_score = outlier_detection(X_imputed)    

    X_result_outlier, Y_result_outlier = remove_outlier(
        dataframe_imputed,
        outlier_score,
        y,
        threshold
    )

    # Then
    assert X_result_outlier.shape[0] == dataframe_imputed.shape[0]
    assert Y_result_outlier.shape[0] == y.shape[0]


# def test_pca_analysis(dataframe):
#     # Given
#     dataset_path = DATASET_FILE_PATH
#     col_to_drop = ['timestamp', 'set']
#     objective_variance = .80
#     columns = ["mean_hr", "sdsd", "lf", "hf"]

#     # When
#     dataframe,_ = get_dataset(dataset_path, col_to_drop)
#     X_imputed, y = impute_nan_values_by_median(dataframe)
#     dataframe_imputed = pd.DataFrame(X_imputed, columns=columns)

#     outlier_score = outlier_detection(X_imputed)    

#     X_result_outlier, Y_result_outlier = remove_outlier(
#         dataframe_imputed,
#         outlier_score,
#         y
#     )

#     pca_out, principalComponents = pca_analysis(X_result_outlier, Y_result_outlier, objective_variance)

#     # Then
#     assert pca_out
#     assert principalComponents.size != 0


# def test_create_final_dataset_with_pca(dataframe):
#     # Given
#     expected_response = False
#     col_to_drop = ['timestamp', 'set']
#     objective_variance = .80
#     columns = ["mean_hr", "sdsd", "lf", "hf"]

#     # When
#     dataframe = dataframe.drop(col_to_drop, 1)
#     X_imputed, y = impute_nan_values_by_median(dataframe)
#     dataframe_imputed = pd.DataFrame(X_imputed, columns=columns)

#     outlier_score = outlier_detection(X_imputed)    

#     X_result_outlier, Y_result_outlier = remove_outlier(
#         dataframe_imputed,
#         outlier_score,
#         y
#     )

#     pca_out, principalComponents = pca_analysis(X_result_outlier, Y_result_outlier, objective_variance)

#     pca_df = create_final_dataset_with_pca(pca_out, principalComponents, Y_result_outlier)

#     # Then
#     assert pca_df.empty == expected_response
#     assert 'label' in pca_df
#     # assert pca_df['label'].shape[0] == Y_result_outlier.shape[0]


def test_computeSeasonerFeatures_when_list_time_is_given():
    # Given
    dataset_path = DATASET_FILE_PATH_FEAT
    col_to_drop = []
    list_time = ["dayOfWeek", "month", "hour", "minute"]

    # When
    _, dataframe = get_dataset(dataset_path, col_to_drop)
    dataframe_with_seasonal_informations = computeSeasonerFeatures(
                                                dataframe=dataframe,
                                                list_time=list_time
                                            )

    # Then
    assert "dayOfWeek" in dataframe_with_seasonal_informations.keys()
    assert "month" in dataframe_with_seasonal_informations.keys()
    assert "hour" in dataframe_with_seasonal_informations.keys()
    assert "minute" in dataframe_with_seasonal_informations.keys()


def test_perform_op_on_features():
    # Given
    dataset_path = DATASET_FILE_PATH_FEAT
    col_to_drop = []
    list_feat = ["mean_hr"]
    operation = "std"
    list_period = [30, 60]

    # When
    _, dataframe = get_dataset(dataset_path, col_to_drop)
    df_with_operation_result = perform_op_on_features(
                                dataframe=dataframe.copy(),
                                list_feat=list_feat,
                                list_period=list_period,
                                operation=operation
                            )
    # Then
    assert "mean_hr_30" in df_with_operation_result.keys()
    assert "mean_hr_30" not in dataframe.keys()
    

def test_computeSeasonerFeatures_when_list_time_not_given():
    # Given
    dataset_path = DATASET_FILE_PATH_FEAT
    col_to_drop = []
    list_time = []

    # When
    _, dataframe = get_dataset(dataset_path, col_to_drop)
    dataframe_with_seasonal_informations = computeSeasonerFeatures(
                                                dataframe=dataframe,
                                                list_time=list_time
                                            )

    # Then
    assert_frame_equal(dataframe, dataframe_with_seasonal_informations)

def test_computeContextFeatures_when_list_feat_is_given():
    # Given
    dataset_path = DATASET_FILE_PATH_FEAT
    col_to_drop = []
    list_feat = ["mean_hr", "mean_nni"]
    list_period = [30, 60]

    # When
    _, dataframe = get_dataset(dataset_path, col_to_drop)
    df_with_contextual_feat_time = computeContextFeatures(
                                    dataframe=dataframe.copy(),
                                    operation="mean",
                                    list_feat=list_feat,
                                    list_period=list_period
                                )
    
    # Then
    assert "mean_hr_30" in df_with_contextual_feat_time.keys()
    assert "mean_hr_60" in df_with_contextual_feat_time.keys()
    assert "mean_nni_30" in df_with_contextual_feat_time.keys()
    assert "mean_nni_60" in df_with_contextual_feat_time.keys()
    assert len(dataframe.columns.to_list()) < len(df_with_contextual_feat_time.columns.to_list())


def test_computeContextFeatures_when_list_feat_not_given():
    # Given
    dataset_path = DATASET_FILE_PATH_FEAT
    col_to_drop = []
    list_feat = []
    list_period = [30, 60]

    # When
    _, dataframe = get_dataset(dataset_path, col_to_drop)
    df_with_contextual_feat_time = computeContextFeatures(
                                    dataframe=dataframe.copy(),
                                    operation="mean",
                                    list_feat=list_feat,
                                    list_period=list_period
                                )
    
    # Then
    assert_frame_equal(dataframe, df_with_contextual_feat_time)


def test_diffOperationFromFeatures_when_list_feat_is_given():
    # Given
    dataset_path = DATASET_FILE_PATH_FEAT
    col_to_drop = []
    list_feat = ["mean_hr", "mean_nni"]

    # When
    _, dataframe = get_dataset(dataset_path, col_to_drop)
    df_with_diff_features = diffOperationFromFeatures(dataframe=dataframe, list_feat=list_feat)

    # Then
    assert "mean_hr_diff" in df_with_diff_features.keys()
    assert "mean_nni_diff" in df_with_diff_features.keys()


def test_diffOperationFromFeatures_when_list_feat_not_given():
    # Given
    dataset_path = DATASET_FILE_PATH_FEAT
    col_to_drop = []
    list_feat = []

    # When
    _, dataframe = get_dataset(dataset_path, col_to_drop)
    df_with_diff_features = diffOperationFromFeatures(dataframe=dataframe, list_feat=list_feat)

    # Then
    assert_frame_equal(dataframe, df_with_diff_features)


def test_prepare_features_with_io():
    # Given
    expected_response = False
    dataset_path = DATASET_FILE_PATH_FEAT,
    col_to_drop= ['interval_index', 'interval_start_time', 'filename']
    features_path = "data/test_data/ml.csv"

    # When
    prepare_features_with_io(
        dataset_path=dataset_path,
        col_to_drop=col_to_drop,
        features_path=features_path)
    res = pd.read_csv(features_path)

    # Then
    assert res.empty == expected_response