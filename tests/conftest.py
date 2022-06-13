from pytest import fixture
import numpy as np
import pandas as pd
import datetime

@fixture
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

# Given
# When
#Then