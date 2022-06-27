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

    parameters
    ----------------
    dataset_path : str
        Path of dataset to load
    col_to_drop : list
        List of columns that we need to remove

    returns
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
    Y_pred = LOF.fit_predict(X)
    X_score = LOF.negative_outlier_factor_

    outlier_score = pd.DataFrame()
    outlier_score['score'] = X_score

    return outlier_score


def remove_outlier(
    X_imputed_df: pd.DataFrame,
    outlier_score: pd.DataFrame,
    y: pd.DataFrame
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:

    """
    This function will remove outlier on the dataset
    parameters
    ----------
    X_imputed_df: pd.DataFrame
    outlier_score: pd.DataFrame
    y: pd.DataFrame

    returns
    -------
    X_result_outlier: pd.DataFrame
        Dataset without outlier points (cleaned data)
    y_result_outlier: pd.DataFrame
        y vector cleaned
    """

    filt = outlier_score["score"] < -1.5
    outlier_index = outlier_score[filt].index.tolist()
    X_result_outlier = pd.DataFrame(X_imputed_df).drop(outlier_index)
    y_result_outlier = y.drop(outlier_index).values

    return X_result_outlier, y_result_outlier


def pca_analysis(
    dataframe: pd.DataFrame,
    objective_variance: float
    ) -> Tuple[any, np.ndarray]:

    """
    This function resolve the problem of multicolinearity in dataset. Some features are too
    correlated. This can be redundant information that we need to remove.
    parameters
    ----------
    dataframe: pd.DataFrame
    objective_variance: float

    returns
    -------
    pca_out: any
        pca object
    principalComponents: np.ndarray
        array of principals components 
    """
    #normalisation step
    dataframe = zscore(dataframe)

    pca = PCA(n_components=objective_variance)
    pca_out = pca.fit(dataframe)
    principalComponents = pca.transform(dataframe)


    return pca_out, principalComponents


def create_final_dataset_with_pca(
    pca_out: any, principalComponents: np.array,
    y: pd.DataFrame
    ) -> pd.DataFrame:

    """
    This function will create the final dataset by using principals components and adding label.
    parameters
    ----------
    pca_out: any
        pca object
    principalComponents: np.ndarray
        array of principals components 
    y: pd.Dataframe
        y label vector

    returns
    -------
    pca_df: pd.DataFrame
        dataframe with principals components and label
    """

    #construction of dataset
    new_size_dataset = pca_out.components_.shape[1]
    list_colmuns = []
    for i in range(1, new_size_dataset):
        list_colmuns.append("PC"+str(i))

    pca_df = pd.DataFrame(principalComponents, columns=list_colmuns)

    pca_df['label'] = y

    return pca_df


def createContextualFeatues(dataframe: pd.DataFrame) -> pd.DataFrame:

    """
    This function compute contextuals features for time domain from the dataset. The idea is we will capture
    information for 30 passed seconds,1 and 2 passed minutes in order to take care contextual aspect of data.
    parameters
    ----------
    dataframe: pd.DataFrame
        contextuals features are computed from the original dataset

    returns
    -------
    dataframe: pd.DataFrame
        dataframe with contextual features
    """
    
    dataframe.timestamp = dataframe.timestamp.apply(convert_timestamp)
    dataframe = dataframe.sort_values(by = 'timestamp').reset_index(drop=True)

    
    dataframe['mean_diff'] = dataframe['mean_hr'].diff().shift(-1)
    dataframe['mean_nni_diff'] = dataframe['mean_nni'].diff().shift(-1)
    dataframe['max_hr_diff'] = dataframe['max_hr'].diff().shift(-1)

    # time features
    dataframe['month'] = dataframe.timestamp.dt.month
    dataframe['dayOfWeek'] = dataframe.timestamp.dt.dayofweek
    dataframe['hour'] = dataframe.timestamp.dt.hour
    dataframe['minute'] = dataframe.timestamp.dt.minute
    dataframe['hour'] = dataframe.timestamp.dt.second

    for period in [30, 60, 120]:

        # Moving averages on different periods
        dataframe['mean_nni_'+str(period)] = dataframe['mean_nni'].rolling(window=period, min_periods=10).mean()
        dataframe["mean_hr_%s"%(period)] = dataframe['mean_hr'].rolling(window=period, min_periods=10).mean()
        dataframe['max_hr_%s'%(period)] = dataframe['max_hr'].rolling(window=period, min_periods=10).max()
        dataframe['min_hr_%s'%(period)] = dataframe['min_hr'].rolling(window=period, min_periods=10).min()

        # Std on differents periods
        dataframe['sdnn_%s'%(period)] = dataframe['sdnn'].rolling(window=period, min_periods=10).std()
        dataframe['sdsd_%s'%(period)] = dataframe['sdsd'].rolling(window=period, min_periods=10).std()
        dataframe['std_hr_%s'%(period)] = dataframe['std_hr'].rolling(window=period, min_periods=10).std()
        dataframe['rmssd_%s'%(period)] = dataframe['rmssd'].rolling(window=period, min_periods=10).std()

        dataframe['lf_%s'%(period)] = dataframe['lf'].rolling(window=period, min_periods=10).mean()
        dataframe['hf_%s'%(period)] = dataframe['hf'].rolling(window=period, min_periods=10).mean()

        # Std on differents periods
        dataframe['vlf_%s'%(period)] = dataframe['vlf'].rolling(window=period, min_periods=10).mean()
        dataframe['lf_hf_ratio_%s'%(period)] = dataframe['lf_hf_ratio'].rolling(window=period, min_periods=10).mean()

        #non linear
        dataframe['csi_%s'%(period)] = dataframe['csi'].rolling(window=period, min_periods=10).mean()
        dataframe['Modified_csi_%s'%(period)] = dataframe['Modified_csi'].rolling(window=period, min_periods=10).mean()
        dataframe['cvi_%s'%(period)] = dataframe['cvi'].rolling(window=period, min_periods=10).mean()
        dataframe['sd1_%s'%(period)] = dataframe['sd1'].rolling(window=period, min_periods=10).mean()
        dataframe['sd2_%s'%(period)] = dataframe['sd2'].rolling(window=period, min_periods=10).mean()

    return dataframe.fillna(-999)


def extract_patient_id(path):
    if path == 'output/feats-v0_6/feats_EEG_297_s1.csv':
        return 12
    return int(path.split('PAT_')[1].split('//')[0])

def extract_session_id(path):
    return int(path.split('_s')[0].split('_')[-1])

def extract_segment_id(path):
    return int(path.split('_s')[1].split('.csv')[0])

def convert_timestamp(timestamp):
    return pd.to_datetime(timestamp)