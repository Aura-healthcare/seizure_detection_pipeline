import pytest
import os

import pandas as pd
import numpy as np

from src.usecase.feature_engineering import FeatureEngineering

DATASET_FILE_PATH = '/home/DATA/DetecTeppe-2022-06-06/ml_dataset_clean/df_ml_train.csv'
OBJECTIF_VAR = 0.80

def test_init_feature_engineering_object():
    
    featEng = FeatureEngineering(DATASET_FILE_PATH, OBJECTIF_VAR)
    assert featEng


def test_impute_nan_and_infinite_values_when_we_havent_infinite_values():

    data = {
        "mean_hr": np.random.rand(10),
        "sdsd": np.random.rand(10),
        "lf": np.random.rand(10),
        "hf": np.random.rand(10)
    }
    data["mean_hr"][0] = np.nan
    data["sdsd"][4] = np.nan
    dataframe = pd.DataFrame(data)

    featEng = FeatureEngineering(DATASET_FILE_PATH, OBJECTIF_VAR)
    X_imputed, Y = featEng.impute_nan_and_infinite_values(dataframe)

    X_imputed = pd.DataFrame(X_imputed, columns=dataframe.columns)

    assert X_imputed.isna().sum().any() != 0


def test_impute_nan_and_infinite_values_when_we_havent_nan_values():

    data = {
        "mean_hr": np.random.rand(10),
        "sdsd": np.random.rand(10),
        "lf": np.random.rand(10),
        "hf": np.random.rand(10),
        "timestamp": np.arange(10)
    }
    data["mean_hr"][0] = np.inf
    data["sdsd"][4] = np.inf
    dataframe = pd.DataFrame(data)

    featEng = FeatureEngineering(DATASET_FILE_PATH, OBJECTIF_VAR)
    X_imputed, Y = featEng.impute_nan_and_infinite_values(dataframe)

    X_imputed = pd.DataFrame(X_imputed, columns=dataframe.columns)

    assert X_imputed.isna().sum().any() != 0


def test_impute_nan_and_infinite_values_when_we_have_nan_and_infinite_values():

    featEng = FeatureEngineering(DATASET_FILE_PATH, OBJECTIF_VAR)

    data = {
        "mean_hr": np.random.rand(10),
        "sdsd": np.random.rand(10),
        "lf": np.random.rand(10),
        "hf": np.random.rand(10),
        "timestamp": np.arange(10)
    }
    data["mean_hr"][0] = np.nan
    data["sdsd"][4] = np.nan
    dataframe = pd.DataFrame(data)

    not_expected_for_imputation = None

    X_imputed, Y = featEng.impute_nan_and_infinite_values(dataframe)

    assert not_expected_for_imputation != X_imputed.any()
           

def test_outlier_detection_when_dataframe_is_imputed():
    
    featEng = FeatureEngineering(DATASET_FILE_PATH, OBJECTIF_VAR)

    X_init = featEng.dataframe_copy.drop('label', 1)
    Y_init = featEng.dataframe_copy['label']

    X_imputed, Y  = featEng.impute_nan_and_infinite_values()
    X, Y = featEng.outlier_detection(X_imputed)

    assert X_init.shape != X.shape
    assert Y_init.shape != Y.shape


def test_outlier_detection_when_dataframe_is_not_imputed():
    
    featEng = FeatureEngineering(DATASET_FILE_PATH, OBJECTIF_VAR)

    data = {
        "mean_hr": np.random.rand(10),
        "sdsd": np.random.rand(10),
        "lf": np.random.rand(10),
        "hf": np.random.rand(10),
        "timestamp": np.arange(10)
    }
    data["mean_hr"][0] = np.nan
    data["sdsd"][4] = np.nan
    dataframe = pd.DataFrame(data)

    with pytest.raises(Exception) as excep:
        X, Y = featEng.outlier_detection(dataframe)
        

def test_pca_analysis():

    featEng = FeatureEngineering(DATASET_FILE_PATH, OBJECTIF_VAR)
    
    X_imputed, Y  = featEng.impute_nan_and_infinite_values()
    X, Y = featEng.outlier_detection(X_imputed)

    pca_df = featEng.pca_analysis(X)

    assert pca_df