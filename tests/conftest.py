from pytest import fixture
import numpy as np
import pandas as pd
import datetime
from freezegun import freeze_time

DATASET_FILE_PATH = "data/test_data/test_data_feat_eng.csv"
DATASET_FILE_PATH_FEAT = "data/test_data/test_data_feat_contextual.csv"

COL_TO_DROP = ['timestamp', 'set']

THRESHOLD = -1.5

LIST_TIME =  ["dayOfWeek", "month", "hour", "minute"]

LIST_FEAT = ["mean_hr"]
OPERATION_TYPE = "std"
LIST_PERIOD = [30, 60]


@fixture
@freeze_time("2022-06-13")
def dataframe():
    data ={
        "mean_hr": 10*[1.],
        "sdsd": 10*[1.3],
        "lf": 10*[1.6],
        "hf": 10*[1.9],
        "set": 10*["train"],
        "timestamp": 10*[datetime.date.today()],
        "label": 10*[1]
    }

    data["mean_hr"][0] = np.nan
    data["sdsd"][4] = np.nan
    data["lf"][2] = np.inf
    data["hf"][6] = np.inf

    dataframe = pd.DataFrame(data)

    return dataframe