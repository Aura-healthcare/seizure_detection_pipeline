import os
from datetime import datetime
import xgboost as xgb
import numpy as np

PROJECT_FOLDER = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_FOLDER = os.path.join(PROJECT_FOLDER, 'data')

TRAIN_DATA = "/home/DATA/DetecTeppe-2022-04-08/ml_dataset_2022_04_08/train/df_ml_train.csv"
TEST_DATA = "/home/DATA/DetecTeppe-2022-04-08/ml_dataset_2022_04_08/test/df_ml_test.csv"
FEATURE_TRAIN_PATH="data/test_data/ml_train.csv"
FEATURE_TEST_PATH="data/test_data/ml_test.csv"

COL_TO_DROP = ['interval_index', 'interval_start_time', 'filename', 'set']

START_DATE = datetime(2021, 4, 22)
CONCURRENCY = 4
SCHEDULE_INTERVAL = None
DEFAULT_ARGS = {'owner': 'airflow'}

TRACKING_URI = 'http://localhost:5000'

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