import sys

from airflow.decorators import dag, task

sys.path.append('.')
from dags.config import (
    DEFAULT_ARGS, 
    START_DATE, 
    CONCURRENCY, 
    SCHEDULE_INTERVAL,
    MODEL_PARAM,
    MLRUNS_DIR,
    TEST_DATA,
    TRACKING_URI,
    TRAIN_DATA)
from src.usecase.train_model import (train_model_with_io)
from src.usecase.feature_engineering import (
                                                replace_infinite_values_by_nan,
                                                impute_nan_values_by_median,
                                                prepare_features_with_io
                                            )


@dag(default_args=DEFAULT_ARGS, 
    start_date=START_DATE,
    schedule_interval=SCHEDULE_INTERVAL,
    concurrency=CONCURRENCY)
def train_model():

    @task
    def prepare_features_task(
            dataset_path: str,
            col_to_drop: list,
            features_path: str) -> str:
        
        prepare_features_with_io(
            dataset_path=dataset_path, 
            col_to_drop=col_to_drop, 
            features_path=features_path)
        
        return features_path
        

    @task
    def train_model_task(
            feature_tain_path: str,
            feature_test_path: str,
            tracking_uri: str = TRACKING_URI,
            model_param: dict = MODEL_PARAM,
            mlruns_dir: str = MLRUNS_DIR) -> None:
        
        train_model_with_io(feature_tain_path, feature_test_path, tracking_uri=tracking_uri, model_param=model_param, mlruns_dir=mlruns_dir)
        
    features_train_path="data/test_data/ml_train.csv"
    features_test_path="data/test_data/ml_test.csv"

    ml_train_path = prepare_features_task(
        dataset_path=TRAIN_DATA,
        col_to_drop=['interval_index', 'interval_start_time', 'filename', 'set'], 
        feature_path=features_train_path)
    
    ml_test_path = prepare_features_task(
        dataset_path=TEST_DATA,
        col_to_drop=['interval_index', 'interval_start_time', 'filename', 'set'], 
        feature_path=features_test_path)
    
    train_model_task(ml_train_path, ml_test_path, tracking_uri=TRACKING_URI, model_param=MODEL_PARAM, mlruns_dir=MLRUNS_DIR)

train_model_dag = train_model()