import pytest
import sys
from pandas.testing import assert_frame_equal
import pandas as pd

sys.path.append('.')
from tests.conftest import DATASET_FILE_PATH, DATASET_FILE_PATH_FEAT, COL_TO_DROP
from src.usecase.data_processing.data_loading import get_dataset


def test_get_dataset_given_path_col_to_drop_return_dataframe_without_col_to_drop(dataframe):
    # Given
    dataset_path = DATASET_FILE_PATH
    col_to_drop = COL_TO_DROP
    expected_reponse = False
    expected_dataframe = dataframe.drop(col_to_drop, axis=1)

    # When
    df_init, df_cleaned = get_dataset(dataset_path, col_to_drop)

    #Then
    assert (df_init.empty == expected_reponse and df_cleaned.empty == expected_reponse)
    assert_frame_equal(df_cleaned, expected_dataframe)
    assert 'timestamp' not in df_cleaned.columns
    assert 'set' not in df_cleaned.columns


def test_get_dataset_given_path_col_to_drop_return_df_origin_and_df_without_col_to_drop(dataframe):
    # Given
    dataset_path = DATASET_FILE_PATH
    col_to_drop = COL_TO_DROP
    expected_dataframe = dataframe
    expected_dataframe['timestamp'] = pd.to_datetime(expected_dataframe['timestamp'], format="%Y-%m-%d")

    # When
    df_init, df_cleaned = get_dataset(dataset_path, col_to_drop)

    #Then
    assert 'timestamp' in df_init.columns 
    assert 'set' in df_init.columns 
    assert 'timestamp' not in df_cleaned.columns 
    assert 'set' not in df_cleaned.columns
