"""
Feature engineering process.

This script clean data before model fitting. The clean actions are defined by exploring data.
It's about imputting nan values by some strategy, detect outliers and remove them and make
some pca analysis to reduce multi-colinéarité.
"""

import string
from typing import Tuple

import numpy as np
from numpy import ndarray
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from scipy.stats import zscore


def get_dataset(dataset_path: str, col_to_drop: list) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    This function will get the dataset by using path. It will also do some
    preparation about dataset before feature engineering.

    Parameters
    ----------------
    dataset_path : str
        Path of dataset to load
    col_to_drop : list
        List of columns that we need to remove

    Returns
    -------
    Tuple of dataframes :
        Dataframe origine and dataframe clean
    """
    
    dataframe_origine: pd.DataFrame = pd.read_csv(dataset_path)
    dataframe_clean: pd.DataFrame = dataframe_origine.copy()

    dataframe_clean = dataframe_clean.sort_values(by='timestamp').reset_index(drop=True)
    dataframe_clean.drop(col_to_drop, axis=1, inplace=True)

    return dataframe_origine, dataframe_clean


def replace_infinite_values_by_nan(dataframe: pd.DataFrame) -> pd.DataFrame:
    
    """
    This is a simple function that replace infinite values by nan.
    """
    
    dataframe.replace([np.inf, -np.inf], np.nan, inplace=True, regex=True)

    return dataframe


def impute_nan_values_by_median(dataframe: pd.DataFrame) -> Tuple[np.array, pd.DataFrame]:
    
    """
    This function will replace nan values by median. 
    Because of many nan values can follow one another in an entire window, 
    we choose to impute nan values by global median in order to avoid extrems values.
    """

    if dataframe.isin([np.inf, -np.inf]).values.any():
        dataframe = replace_infinite_values_by_nan(dataframe)

    X = dataframe.drop(['label'], axis=1)
    Y = dataframe['label']

    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    return X_imputed, Y


def outlier_detection(X: np.ndarray) -> pd.DataFrame:

    """
    This function detect outlier based on local density of points.
    """

    LOF = LocalOutlierFactor()
    Y_pred = LOF.fit_predict(X)
    X_score = LOF.negative_outlier_factor_

    outlier_score = pd.DataFrame()
    outlier_score['score'] = X_score

    return outlier_score


def remove_outlier(
    X_imputed_df: pd.DataFrame,
    outlier_score: pd.DataFrame,
    Y: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    This function will remove outlier on the dataset
    """

    filt = outlier_score["score"] < -1.5
    outlier_index = outlier_score[filt].index.tolist()
    X_result_outlier = pd.DataFrame(X_imputed_df).drop(outlier_index)
    Y_result_outlier = Y.drop(outlier_index).values

    return X_result_outlier, Y_result_outlier


def pca_analysis(
    dataframe: pd.DataFrame,
    Y: pd.DataFrame,
    objective_variance: float
    ) -> Tuple[any, np.ndarray]:

    """
    This function resolve the problem of multicolinearity in dataset. Some features are too
    correlated. This can be redundant information that we need to remove.
    """
    #normalisation step
    #dataframe = zscore(dataframe)

    pca = PCA(n_components=objective_variance)
    pca_out = pca.fit(dataframe)
    principalComponents = pca.transform(dataframe)


    return pca_out, principalComponents


def create_final_dataset_with_pca(
    pca_out: any, principalComponents: np.array,
    Y: pd.DataFrame
    ) -> pd.DataFrame:

    """
    This function will create the final dataset by adding label.
    """

    #construction of dataset
    new_size_dataset = pca_out.components_.shape[1]
    list_colmuns = []
    for i in range(1, new_size_dataset):
        list_colmuns.append("PC"+str(i))

    pca_df = pd.DataFrame(principalComponents, columns=list_colmuns)

    pca_df['label'] = Y

    return pca_df


def createContextualFeatues(dataframe: pd.DataFrame, period: int) -> pd.DataFrame:

    """
    This function compute contextuals features for time domain from the dataset. The idea is we will capture
    information for 30 passed seconds, 1 passed minutes in order to take care times series aspect.
    """

    dataframe = dataframe.sort_values(by = 'timestamp').reset_index(drop=True)

    # time_features = ['mean_nni', 'sdnn', 'sdsd',
    #    'nni_50', 'pnni_50', 'nni_20', 'pnni_20', 'rmssd', 'median_nni',
    #    'range_nni', 'cvsd', 'cvnni', 'mean_hr', 'max_hr', 'min_hr', 'std_hr']
    # dataframe_time_domain: pd.DataFrame = dataframe[time_features]

    # Moving averages on different periods
    dataframe['mean_nni_'+str(period)] = dataframe['mean_nni'].rolling(window=period, min_periods=10).mean().shift(-10)
    dataframe['mean_hr_'+str(period)] = dataframe['mean_hr'].rolling(window=period, min_periods=10).mean().shift(-10)
    dataframe['max_hr_'+str(period)] = dataframe['max_hr'].rolling(window=period, min_periods=10).mean().shift(-10)
    dataframe['mean_diff'] = dataframe['mean_hr'].diff()
    dataframe['mean_nni_diff'] = dataframe['mean_nni'].diff()
    dataframe['max_hr_diff'] = dataframe['max_hr'].diff()

    # Std on differents periods
    dataframe['sdnn_'+str(period)] = dataframe['sdnn'].rolling(window=period, min_periods=10).std().shift(-10)
    dataframe['sdsd_'+str(period)] = dataframe['sdsd'].rolling(window=period, min_periods=10).std().shift(-10)
    dataframe['std_hr_'+str(period)] = dataframe['std_hr'].rolling(window=period, min_periods=10).std().shift(-10)
    
    # Min max on differents periods


    dataframe['lf_'+str(period)] = dataframe['lf'].rolling(window=period, min_periods=10).mean().shift(-10)
    dataframe['hf_'+str(period)] = dataframe['hf'].rolling(window=period, min_periods=10).mean().shift(-10)

    # Std on differents periods
    dataframe['vlf_'+str(period)] = dataframe['vlf'].rolling(window=period, min_periods=10).mean().shift(-10)
    dataframe['lf_hf_ratio_'+str(period)] = dataframe['lf_hf_ratio'].rolling(window=period, min_periods=10).mean().shift(-10)

    return dataframe.dropna()

def extract_patient_id(path):
    if path == 'output/feats-v0_6/feats_EEG_297_s1.csv':
        return 12
    return int(path.split('PAT_')[1].split('//')[0])

def extract_session_id(path):
    return int(path.split('_s')[0].split('_')[-1])

def extract_segment_id(path):
    return int(path.split('_s')[1].split('.csv')[0])