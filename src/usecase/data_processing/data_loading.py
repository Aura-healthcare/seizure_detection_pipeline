

from typing import Tuple

import pandas as pd


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
    
    dataframe_origine: pd.DataFrame = pd.read_csv(dataset_path, index_col=False)

    sorted_dataframe = dataframe_origine.sort_values(by='timestamp', inplace=False).reset_index(drop=True, inplace=False)
    dataframe_without_cols = sorted_dataframe.drop(col_to_drop, axis=1, inplace=False)

    return dataframe_origine, dataframe_without_cols