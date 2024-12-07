import sys
import pandas as pd

sys.path.append('.')
from tests.conftest import DATASET_FILE_PATH_FEAT
from src.usecase.data_processing.prepare_features import prepare_features_with_io


def test_prepare_features_with_io_check_output_dataframe_is_created():
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

    actual_response = pd.read_csv(features_path)

    # Then
    assert actual_response.empty == expected_response