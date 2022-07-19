from typing import Tuple
import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.neighbors import LocalOutlierFactor


def replace_infinite_values_by_nan(dataframe: pd.DataFrame) -> pd.DataFrame:
    
    """
    This is a simple function that replace infinite values by nan.
    parameters
    ----------
    dataframe : pd.Dataframe
        Dataframe that will be used to draft infinite values

    returns
    -------
    dataframe: pd.Dataframe
        Dataframe without infinite values
    """
    
    dataframe.replace([np.inf, -np.inf], np.nan, inplace=True, regex=True)

    return dataframe


def impute_nan_values_by_median(dataframe: pd.DataFrame) -> Tuple[np.ndarray, pd.DataFrame]:
    
    """
    This function will replace nan values by median. 
    Because of many nan values can follow one another in an entire window, 
    we choose to impute nan values by global median in order to avoid extrems values.
    parameters
    ----------
    dataframe: pd.Dataframe
        Dataframe that will be imputed by median

    returns
    -------
    X_imputed: np.ndarray
        X matrix imputted
    y : pd.Dataframe
        y vector
    """

    if dataframe.isin([np.inf, -np.inf]).values.any():
        dataframe = replace_infinite_values_by_nan(dataframe)

    X = dataframe.drop(['label'], axis=1)
    y = dataframe['label']

    imputer = SimpleImputer(missing_values=np.nan, strategy='median')
    X_imputed = imputer.fit_transform(X)
    
    return X_imputed, y


def outlier_detection(X: np.ndarray) -> pd.DataFrame:

    """
    This function detect outlier based on local density of points.
    parameters
    ----------
    X: np.ndarray
        X imputted will be use to detect outlier
    
    returns
    -------
    outlier_score: pd.Dataframe
        every point of data has outlier score which is saved
    """

    LOF = LocalOutlierFactor()
    y_pred = LOF.fit_predict(X)
    X_score = LOF.negative_outlier_factor_

    outlier_score = pd.DataFrame()
    outlier_score['score'] = X_score

    return outlier_score


def remove_outlier(
    X_imputed_df: pd.DataFrame,
    outlier_score: pd.DataFrame,
    y: pd.DataFrame,
    threshold: float) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    This function will remove outlier on the dataset
    parameters
    ----------
    X_imputed_df: pd.DataFrame
    outlier_score: pd.DataFrame
    y: pd.DataFrame
    threshold: float
    threshold: float
        threshold is the limit for removing outliers
        
    returns
    -------
    X_result_outlier: pd.DataFrame
        Dataset without outlier points (cleaned data)
    y_result_outlier: pd.DataFrame
        y vector cleaned
    """

    filt = outlier_score["score"] < threshold
    outlier_index = outlier_score[filt].index.tolist()
    X_result_outlier = pd.DataFrame(X_imputed_df).drop(outlier_index)
    y_result_outlier = y.drop(outlier_index).values

    return X_result_outlier, y_result_outlier