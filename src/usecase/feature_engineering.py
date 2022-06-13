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
    """
    
    dataframe: pd.DataFrame = pd.read_csv(dataset_path)
    dataframe_copy: pd.DataFrame = dataframe.copy()

    dataframe_copy = dataframe_copy.sort_values(by='timestamp').reset_index(drop=True)
    dataframe_copy.drop(col_to_drop, axis=1, inplace=True)

    return dataframe, dataframe_copy


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
    dataframe = zscore(dataframe)


    pca = PCA(n_components=objective_variance)
    pca_out = pca.fit(dataframe)
    principalComponents = pca.transform(dataframe)


    return pca_out, principalComponents


def create_final_dataset_with_pca(
    pca_out, principalComponents: np.ndarray,
    Y: pd.DataFrame
    ) -> pd.DataFrame:

    """
    This function will create the final dataset by adding label.
    """

    #construction of dataset
    new_size_dataset = pca_out.components_.shape[0]
    list_colmuns = []
    for i in range(1, new_size_dataset):
        list_colmuns.append("PC"+i)
    
    pca_df = pd.DataFrame(principalComponents, columns=list_colmuns)

    pca_df['label'] = Y

    return pca_df

