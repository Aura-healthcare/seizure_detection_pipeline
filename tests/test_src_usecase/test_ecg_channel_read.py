import pytest
from src.infrastructure.edf_loader import EdfLoader
from src.usecase.ecg_channel_read import ecg_channel_read


DEFAULT_PATH = '/home/DATA/lateppe/Recherche_ECG/'
FILE_PATH = 'PL/PAT_12/EEG_156_s1.edf'
RECORD = 156
PATIENT = 'PAT_12'
SEGMENT ='Segments_EEG_{RECORD}.csv'

def test_ecg_channel_read():

    #Given
    edfloader = EdfLoader(FILE_PATH)
    ecg_channel_name = edfloader.get_ecg_candidate_channel()
    start_time, end_time = edfloader.get_edf_file_interval()

    #When
    sample_frequency, df_ecg, df_annot, df_seg = ecg_channel_read(PATIENT, RECORD, SEGMENT, ecg_channel_name, start_time, end_time)
    
    #Then
    assert(sample_frequency!=0 and not df_ecg.empty and not df_seg.empty)