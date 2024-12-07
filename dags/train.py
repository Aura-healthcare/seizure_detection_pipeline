from src.usecase.data_processing.prepare_features import prepare_features_with_io
from src.usecase.train_model import (train_pipeline_with_io)
from dags.config import (
    DEFAULT_ARGS,
    START_DATE,
    CONCURRENCY,
    SCHEDULE_INTERVAL,
    MODELS_PARAM,
    MLRUNS_DIR,
    TEST_DATA,
    TRACKING_URI,
    TRAIN_DATA,
    FEATURE_TRAIN_PATH,
    FEATURE_TEST_PATH,
    COL_TO_DROP)
import sys

from airflow.decorators import dag, task

@dag(default_args=DEFAULT_ARGS,
     start_date=START_DATE,
     schedule_interval=SCHEDULE_INTERVAL,
     catchup=False,
     concurrency=CONCURRENCY)
def train_pipeline():

    @task
    def prepare_features_task(
            dataset_path: str,
            col_to_drop: list,
            feature_path: str) -> str:

        prepare_features_with_io(
            dataset_path=dataset_path,
            col_to_drop=col_to_drop,
            features_path=feature_path)

        return feature_path

    @task
    def train_model_task(
            feature_tain_path: str,
            feature_test_path: str,
            tracking_uri: str = TRACKING_URI,
            model_param: dict = MODELS_PARAM['xgboost'],
            mlruns_dir: str = MLRUNS_DIR) -> None:

        train_pipeline_with_io(feature_tain_path, feature_test_path,
                            tracking_uri=tracking_uri, model_param=model_param, mlruns_dir=mlruns_dir)

    # Orchestration
    features_train_path = FEATURE_TRAIN_PATH
    features_test_path = FEATURE_TEST_PATH

    ml_train_path = prepare_features_task(
        dataset_path=TRAIN_DATA,
        col_to_drop=COL_TO_DROP,
        feature_path=features_train_path)

    ml_test_path = prepare_features_task(
        dataset_path=TEST_DATA,
        col_to_drop=COL_TO_DROP,
        feature_path=features_test_path)

    train_model_task(feature_tain_path=ml_train_path, feature_test_path=ml_test_path, tracking_uri=TRACKING_URI,
                     model_param=MODELS_PARAM['xgboost'], mlruns_dir=MLRUNS_DIR)


train_pipeline_dag = train_pipeline()