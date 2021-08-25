from datetime import datetime
import pandas as pd
from airflow.decorators import dag, task

from src.usecase.ecg_channel_read import ecg_channel_read
from src.usecase.detect_qrs import detect_qrs
from src.usecase.apply_ecg_qc import apply_ecg_qc
from src.usecase.remove_noisy_segments import remove_noisy_segments
from src.usecase.ecg_qc_statistical_analysis import ecg_qc_statistical_analysis

START_DATE = datetime(2021, 4, 22)
CONCURRENCY = 12
SCHEDULE_INTERVAL = None
DEFAULT_ARGS = {'owner': 'airflow'}

DICT_PARAMS = {
    "patient": "PAT_6",
    "record": "77",
    "segment": "s1",
    "channel_name": "emg6+emg6-",
    "start_time": "2020-12-18 13:00:00",
    "end_time": "2020-12-18 14:30:00"
}
INFOS = list(DICT_PARAMS.values())[:4]

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
    def t_ecg_channel_read(dict_params: dict) -> pd.DataFrame:
        sample_frequency, df_ecg, df_annot, df_seg = \
            ecg_channel_read(**dict_params)
        return {
            "sample_frequency": sample_frequency,
            "df_ecg": df_ecg,
            "df_annot": df_annot,
            "df_seg": df_seg
        }

    @task()
    def t_detect_qrs(dict_params: dict) -> str:
        detect_qrs(**dict_params)
        return "detect qrs done"

    @task()
    def t_apply_ecg_qc(dict_params: dict) -> str:
        apply_ecg_qc(**dict_params)
        return "apply ecg qc done"

    @task()
    def t_remove_noisy_segment(dict_params: dict, flag_detect_qrs: str,
                               flag_apply_ecg_qc: str) -> str:
        remove_noisy_segments(**dict_params)
        return "remove noisy segment done"

    @task()
    def t_ecg_qc_statistical_analysis(dict_params: dict,
                                      flag_apply_ecg_qc: str):
        ecg_qc_statistical_analysis(dict_params)

    returned_dict = t_ecg_channel_read(DICT_PARAMS)
    ecg_data = returned_dict["df_ecg"]
    sf = returned_dict["sample_frequency"]

    flag_detect_qrs = t_detect_qrs(
        {
            "ecg_data": ecg_data,
            "sampling_frequency": sf,
            "method": DETECT_QRS_METHOD,
            "infos": INFOS
        }
    )

    flag_apply_ecg_qc = t_apply_ecg_qc(
        {
            "ecg_data": ecg_data,
            "sampling_frequency": sf,
            "model": ECG_QC_MODEL,
            "infos": INFOS
        }
    )

    t_remove_noisy_segment(
        {
            "rr_interval_file":
            f"{'_'.join(INFOS)}.csv",
            "chunk_file":
            f"{'_'.join(INFOS)}_#_{ECG_QC_MODEL.split('.')[0]}.csv",
            "length_chunk": LENGTH_CHUNK,
            "sampling_frequency": sf
        },
        flag_detect_qrs,
        flag_apply_ecg_qc
    )

    t_ecg_qc_statistical_analysis(
        {
            "chunk_file":
            f"{'_'.join(INFOS)}_#_{ECG_QC_MODEL.split('.')[0]}.csv"
        },
        flag_apply_ecg_qc
    )


dag_pipeline = dag_seizure_detection_pipeline()
