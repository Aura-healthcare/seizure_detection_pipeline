import os
import os
import sys
from datetime import datetime as dt
import datetime
import xgboost as xgb
import numpy as np

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')

ML_DATASET_OUTPUT_FOLDER = "/opt/airflow/output"
AIRFLOW_PREFIX_TO_DATA = '/opt/airflow/data/'

TRAIN_DATA = os.path.join(AIRFLOW_PREFIX_TO_DATA, "train/df_ml_train.csv")
TEST_DATA = os.path.join(AIRFLOW_PREFIX_TO_DATA , "test/df_ml_test.csv")
FEATURE_TRAIN_PATH= os.path.join(ML_DATASET_OUTPUT_FOLDER, "ml_train.csv")
FEATURE_TEST_PATH= os.path.join(ML_DATASET_OUTPUT_FOLDER, "ml_test.csv")

COL_TO_DROP = ['interval_index', 'interval_start_time', 'set']

START_DATE = dt(2021, 8, 1)
CONCURRENCY = 4
SCHEDULE_INTERVAL = datetime.timedelta(minutes=10)
DEFAULT_ARGS = {'owner': 'airflow'}

TRACKING_URI = 'http://mlflow:5000'

# MODEL_PARAM = {
#     'model': RandomForestClassifier(),
#     'grid_parameters': {
#         'min_samples_leaf': np.arange(1, 5, 1),
#         'max_depth': np.arange(1, 7, 1),
#         'max_features': ['auto'],
#         'n_estimators': np.arange(10, 20, 2)}}

MODEL_PARAM = {
    'model': xgb.XGBClassifier(),
    'grid_parameters': {
        'nthread':[4],
        'learning_rate': [0.1, 0.01, 0.05],
        'max_depth': np.arange(3, 5, 2),
        'scale_pos_weight':[1],
        'n_estimators': np.arange(15, 25, 2),
        'missing':[-999]}
    }

MLRUNS_DIR = f'{os.getcwd()}/mlruns'