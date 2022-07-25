"""
Prepare Features process.

This script prepare data before training model. Data are loaded, cleaned before giving them to machine
learning model. Some transformation can be done by using time series aspects.
"""
import pandas as pd

from src.usecase.data_processing.time_series_processing import (compute_contextuals_features, 
                                                                compute_time_features)
from src.usecase.data_processing.data_cleaning import replace_infinite_values_by_nan
from src.usecase.data_processing.data_loading import get_dataset


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
    dataframe = compute_time_features(dataframe=dataframe, list_time=list_time)

    list_feat, list_period = ["mean_hr", "mean_nni"], [30, 60]
    dataframe = compute_contextuals_features(
                    dataframe=dataframe, 
                    operation="mean",
                    list_feat=list_feat,
                    list_period=list_period
                    )

    return dataframe


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