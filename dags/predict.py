import os
import sys
from datetime import datetime, timedelta, datetime

from airflow.decorators import dag, task
from airflow.utils.dates import days_ago

sys.path.append('.')
from dags.config import (DEFAULT_ARGS, START_DATE, CONCURRENCY, SCHEDULE_INTERVAL)


@dag(default_args=DEFAULT_ARGS, 
    start_date=START_DATE,
    schedule_interval=timedelta(minutes=2),
    concurrency=CONCURRENCY)
def predict():
    @task
    def prepare_features_with_io_task() -> str:
        pass

    @task
    def predict_with_io_task(feature_path: str) -> None:
        pass

    feature_path = prepare_features_with_io_task()
    predict_with_io_task(feature_path)

predict_dag = predict()