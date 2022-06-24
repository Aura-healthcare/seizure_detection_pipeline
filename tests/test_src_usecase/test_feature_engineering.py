import sys
from matplotlib.pyplot import get
import pytest
from pandas.testing import assert_frame_equal

import pandas as pd
import numpy as np
from sklearn import datasets
from sqlalchemy import column

sys.path.append('.')
DATASET_FILE_PATH = "/home/aura-sakhite/seizure_detection_pipeline/data/test_data/test_data_feat_eng.csv"
DATASET_FILE_PATH_FEAT = "/home/DATA/DetecTeppe-2022-04-08/ml_dataset_2022_04_08/test/df_ml_test.csv"

from src.usecase.feature_engineering import (
                                                get_dataset,
                                                create_final_dataset_with_pca,
                                                impute_nan_values_by_median,
                                                outlier_detection,
                                                pca_analysis,
                                                replace_infinite_values_by_nan,
                                                remove_outlier,
                                                createTimeDomainContextualFeatues
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
    expected_response = True
    col_to_drop = ['timestamp', 'set']

    # When
    dataframe = dataframe.drop(col_to_drop, 1)
    dataframe = replace_infinite_values_by_nan(dataframe)
    X_imputed, y = impute_nan_values_by_median(dataframe)
    dataframe_imputed = pd.DataFrame(X_imputed)

    # Then
    assert dataframe_imputed.isna().values.any() != expected_response
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

    # When
    _, dataframe = get_dataset(dataset_path, col_to_drop)
    X_imputed, y = impute_nan_values_by_median(dataframe)
    dataframe_imputed = pd.DataFrame(X_imputed)

    outlier_score = outlier_detection(X_imputed)    

    X_result_outlier, Y_result_outlier = remove_outlier(
        dataframe_imputed,
        outlier_score,
        y
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


def test_createTimeDomainContextualFeatures():
    # Given
    expected_response = False
    dataset_path = DATASET_FILE_PATH_FEAT
    col_to_drop = []
    _ , dataframe = get_dataset(dataset_path, col_to_drop)

    # When
    dataframe_time_domain = createTimeDomainContextualFeatues(dataframe, 30)
    print(dataframe_time_domain)

    # Then
    assert dataframe_time_domain.empty == expected_response
    assert len(dataframe.columns.to_list()) > len(dataframe_time_domain.columns.to_list())