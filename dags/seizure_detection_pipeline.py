from datetime import datetime
from typing import Tuple

import pandas as pd

from airflow.decorators import dag, task

from src.usecase.fetch_database import fetch_database
from src.usecase.detect_qrs import detect_qrs
from src.usecase.apply_ecg_qc import apply_ecg_qc
from src.usecase.remove_noisy_segments import remove_noisy_segments
from src.usecase.ecg_qc_statistical_analysis import ecg_qc_statistical_analysis

START_DATE = datetime(2021, 4, 22)
CONCURRENCY = 12
SCHEDULE_INTERVAL = None
DEFAULT_ARGS = {'owner': 'airflow'}

DICT_PARAMS_EDF_FILE = {
    "patient": "PAT_6",
    "record": "77",
    "file_path": ""
    "segment": "s1",
    "channel_name": "emg6+emg6-",
    "start_time": "2020-12-18 13:00:00",
    "end_time": "2020-12-18 14:30:00",
    "data_path": 'data'
}
INFOS = list(DICT_PARAMS_EDF_FILE.values())[:4]

DETECT_QRS_METHOD = 'hamilton'
ECG_QC_MODEL = 'rfc_normalized_2s.pkl'
LENGTH_CHUNK = 2


@dag(default_args=DEFAULT_ARGS,
     dag_id='seizure_detection_pipeline',
     description='Start the whole seizure detection pipeline',
     start_date=START_DATE,
     schedule_interval=SCHEDULE_INTERVAL,
     concurrency=CONCURRENCY)
def dag_seizure_detection_pipeline():


    @task(multiple_outputs=True)
    def t_detect_qrs(dict_params: dict) -> Tuple[int, str]:
        sampling_frequency, filename = detect_qrs(**dict_params)
        return {
            "sampling_frequency": sampling_frequency,
            "filename": filename
        }

    @task(multiple_outputs=True)
    def t_fetch_database(dict_params: dict) -> Tuple[str, str]:
        db_list_filename = fetch_database(**dict_params)
        return db_list_filename

    @task()
    def t_apply_ecg_qc(dict_params: dict) -> str:
        filename = apply_ecg_qc(**dict_params)
        return filename

    @task()
    def t_remove_noisy_segment(dict_params: dict) -> str:
        filename = remove_noisy_segments(**dict_params)
        return filename

    @task()
    def t_ecg_qc_statistical_analysis(dict_params: dict):
        ecg_qc_statistical_analysis(**dict_params)

    db_list_filename = t_fetch_database(
        {
            "data_folder_path": "data/tuh",
            "export_folder": "output/db"
        }
    )

    data_list_filename = db_list_filename["data"]
    annotation_list_filename = db_list_filename["annotations"]

    db = pd.read_csv(data_list_filename)
    for exam in db:

    #return_dict = t_detect_qrs(
    #    {
    #        **DICT_PARAMS_EDF_FILE,
    #        "method": DETECT_QRS_METHOD,
    #        "infos": INFOS
    #    }
    #)
    #sampling_frequency = return_dict["sampling_frequency"]
    #file_rr_intervals = return_dict["filename"]
    #
    #file_quality = t_apply_ecg_qc(
    #    {
    #        **DICT_PARAMS_EDF_FILE,
    #        "model": ECG_QC_MODEL,
    #        "infos": INFOS
    #    }
    #)
    #
    #file_clean_rr_intervals = t_remove_noisy_segment(
    #    {
    #        "rr_intervals_file": file_rr_intervals,
    #        "chunk_file": file_quality,
    #        "length_chunk": LENGTH_CHUNK,
    #        "sampling_frequency": sampling_frequency
    #    }
    #)
    #
    #t_ecg_qc_statistical_analysis(
    #    {
    #        "chunk_file": file_quality
    #    }
    #
    #)


dag_pipeline = dag_seizure_detection_pipeline()
