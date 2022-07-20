import sys
from matplotlib.pyplot import get
import pytest
from pandas.testing import assert_frame_equal

import pandas as pd
import numpy as np
from sklearn import datasets
from sqlalchemy import column


sys.path.append('.')
from tests.conftest import DATASET_FILE_PATH 

from src.usecase.feature_engineering import (
                                                prepare_features,
                                                prepare_features_with_io,
                                                diffOperationFromFeatures
                                            )
    

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