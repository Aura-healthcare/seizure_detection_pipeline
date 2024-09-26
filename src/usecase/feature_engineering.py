"""
Feature engineering process.

This script clean data before model fitting. The clean actions are defined by exploring data.
It's about imputting nan values by some strategy, detect outliers and remove them and make
some pca analysis to reduce multi-colinéarité.
"""

from typing import Tuple

import numpy as np
from numpy import ndarray
import pandas as pd
from sklearn.neighbors import LocalOutlierFactor
from sklearn.impute import SimpleImputer
from sklearn.decomposition import PCA
from scipy.stats import zscore


def prepare_features_with_io(dataset_path: str,
        col_to_drop: list, features_path: str) -> None:
    """
    This function prepare features by making artefact. It is used on airflow pipeline
    parameteres
    -----------
    dataset_path: str
    col_to_drop: list
    features_path: str
    return
    ------
    None
    """
    
    data = prepare_features(
                    dataset_path=dataset_path,
                    col_to_drop=col_to_drop)

    data.to_csv(features_path, index=False)


def prepare_features(
        dataset_path: str,
        col_to_drop: list) -> pd.DataFrame:
    
    """
    This function prepare feature before training model
    parameters
    ----------
    dataset_path: str
    col_to_drop: list

    return 
    ------
    dataframe: pd.DataFrame
    """

    dataframe = pd.concat(pd.read_csv(p) for p in dataset_path).drop(col_to_drop, 1)
    

    dataframe = replace_infinite_values_by_nan(dataframe=dataframe)

    list_time = ["dayOfTheWeek", "month", "hour"]
    dataframe = computeSeasonerFeatures(dataframe=dataframe, list_time=list_time)

    list_feat, list_period = ["mean_hr", "mean_nni"], [30, 60]
    dataframe = computeContextFeatures(
                    dataframe=dataframe, 
                    operation="mean",
                    list_feat=list_feat,
                    list_period=list_period
                    )

    return dataframe


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


def computeContextFeatures(
        dataframe: pd.DataFrame, 
        operation: str, 
        list_feat: list, 
        list_period: list) -> pd.DataFrame:
    """
    This function will create contextuals features from time domaine features.
    new features will be generated by doing operations like : rolling, diff, etc.
    parameters
    ----------
    dataframe: pd.DataFrame
        dataframe constructed from dataset
    operation: str
        type of operation to use on feature
    list_feat: list
        list of features that we want to use to compute contextuals time domaine features
    period: list
        list of period which is used to compute new features
    
    return
    ------
    dataframe: pd.DataFrame
        new dataframe with contextuals time domaine features
    """

    dataframe = perform_op_on_features(
            dataframe=dataframe, 
            list_feat=list_feat, 
            operation=operation,
            list_period=list_period)
        
    return dataframe


def computeSeasonerFeatures(dataframe: pd.DataFrame, list_time: list) -> pd.DataFrame:
    """
    This function compute seasoner information from timestamp. These informations
    are used to let know the model weither of seasonality informations is in time series.
    parameters
    ----------
    dataframe: pd.DataFrame
        dataframe constructed from dataset
    list_time: list
        list of time used to compute seasonal information

    return 
    ------
    dataframe: pd.DataFrame
        new dataframe with seasonal informations
    """
    dataframe.timestamp = dataframe.timestamp.apply(convert_timestamp)
    dataframe = dataframe.sort_values(by = 'timestamp').reset_index(drop=True)

    for time in list_time:
        if time == "dayOfWeek":
            dataframe['dayOfWeek'] = dataframe.timestamp.dt.month
        elif time == "month":
            dataframe['month'] = dataframe.timestamp.dt.dayofweek
        elif time == "hour":
            dataframe['hour'] = dataframe.timestamp.dt.hour
        elif time == "minute":
            dataframe['minute'] = dataframe.timestamp.dt.minute
        else:
            dataframe['second'] = dataframe.timestamp.dt.second

    return dataframe


def diffOperationFromFeatures(dataframe: pd.DataFrame, list_feat: list) -> pd.DataFrame:
    """
    This function will  compute difference from chosen features. It aims
    to know the variability of features.
    parameters
    ----------
    dataframe: pd.DataFrame
        dataframe constructed from dataset
    list_feat: list
        list of features that we want to use to compute difference operation.
    
    return
    ------
    dataframe: pd.DataFrame
        new dataframe with variability informations.
    """

    for feat in list_feat:
        dataframe[feat+'_diff'] = dataframe[feat].diff().shift(-1) 
    
    return dataframe

def perform_op_on_features(dataframe: pd.DataFrame, list_feat: list, list_period: list, operation: str) -> pd. DataFrame:
    """
    This functioncompute operation on features by iterating over list of features.
    parameters
    ----------
    dataframe: pd.DataFrame
        dataframe from dataset
    list_feat: str
        list of features that we want to use
    operation: str
        operation performed on features : mean, std, etc.
    list_period: list
        list of periods which is used to compute new features

    return
    ------
    dataframe: pd.Dataframe
    """

    for period in list_period:
        if operation == "mean":
            for feat in list_feat:
                dataframe[feat+'_%s'%(period)] = dataframe[feat].rolling(window=period, min_periods=10).mean()
        elif operation == "std":
            for feat in list_feat:
                dataframe[feat+'_%s'%(period)] = dataframe[feat].rolling(window=period, min_periods=10).std()
        else:
            for feat in list_feat:
                dataframe[feat+'_%s'%(period)] = dataframe[feat].rolling(window=period, min_periods=10).sum()
    
    return dataframe

def create_X_y_set(dataframe: pd.DataFrame, col_to_drop: list) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    This function create X matrix and y label vector.
    parameters
    ----------
    dataframe: pd.DataFrame
    col_to_drop: list

    returns
    -------
    X: pd.Dataframe
    y: pd.Dataframe
    """

    X = dataframe.drop(col_to_drop, 1)
    y = dataframe['label']

    return X, y

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