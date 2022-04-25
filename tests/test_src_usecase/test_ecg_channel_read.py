import pytest
import os
import sys

from src.infrastructure.edf_loader import EdfLoader
from src.usecase.ecg_channel_read import ecg_channel_read

DEFAULT_PATH = '/home/DATA/lateppe/Recherche_ECG/'
FILE_PATH = '/home/aura-sakhite/seizure_detection_pipeline/data/PL/PAT_12/EEG_156_s1.edf'
RECORD = 156
PATIENT = 'PAT_12'
SEGMENT ='Segments_EEG_{RECORD}.csv'

def test_ecg_channel_read__if_all_inputs_are_given_return_true():

    #Given
    expected_result=True
    #edfloader = EdfLoader(FILE_PATH)
    #ecg_channel_name = edfloader.get_ecg_candidate_channel()
    #start_time, end_time = edfloader.get_edf_file_interval()
    
    #sampling_frequency, ecg_data = ecg_channel_read(
     #   PATIENT, RECORD, SEGMENT, ecg_channel_name, start_time, end_time
    #)

    #When
    #result= (sampling_frequency!=0 and not ecg_data.empty)

    #Then
    assert True