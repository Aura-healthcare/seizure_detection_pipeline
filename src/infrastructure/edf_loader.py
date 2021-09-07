"""create_ecg_dataset script
This script creates and exports a DataFrame for an ECG signal. It takes into
consideration several elements to load corresponding EDF file of the server.
This file can also be imported as a module and contains the following
class:
    * EdfLoader - A class used to load an edf file and export it in
    DataFrame format
"""
from os.path import join

import pandas as pd
from pyedflib import highlevel
from pyedflib import edfreader


class EdfLoader:
    """
    A class used to load an edf file and export it in DataFrame format
    ...
    Attributes
    ----------
    patient : str
        Patient to load
    record : str
        Record to load
    segment : str
        Segment to load
    edf_file_path : str
        Path of the EDF file to load
    headers : dict
        Headers of the EDF file
    channels : list
        Channels availiable if EDF file
    startdate : pd.DateTime
        Date of the beginning of the record
    sampling_frequency_hz: int
        Frequency of the sample in hz
    Methods
    -------
    convert_edf_to_dataframe(channel_name, start_time, end_time)
        Load EDF file and wrangle it into the DataFrame format
    """

    def __init__(self,
                 data_path: str,
                 patient: str,
                 record: str,
                 segment: str):
        """
        Parameters
        ----------
        data_path : str
            data path
        patient : str
            patient
        record : str
            Record to load
        segment : str
            Segment to load
        """
        self.data_path = data_path
        self.patient = patient
        self.record = record
        self.segment = segment
        self.edf_file_path = join(self.data_path, self.patient,
                                  f'EEG_{self.record}_{self.segment}.edf')

        self.headers = highlevel.read_edf_header(self.edf_file_path)
        self.channels = self.headers['channels']
        self.startdate = pd.to_datetime(
            self.headers['startdate']) + pd.Timedelta(hours=1)

    def convert_edf_to_dataframe(self,
                                 channel_name: str,
                                 start_time: pd.Timestamp,
                                 end_time: pd.Timestamp) -> pd.DataFrame:

        """Extract the ECG signal for a channel and export it to DataFrame
        format, limited by requested start time and end time
        Parameters
        ----------
        channel_name : str
            Name of the channel to load
        start_date : pd.Timestamp
            Start of the ECG signal to filter
        end_date : pd.Timestamp
            Start of the ECG signal to filter
        Returns
        -------
        df_ecg : pd.DataFrame
            DataFrame of the ECG for the requested channel, filter by
            with start and end timestamps
        """
        self.sampling_frequency_hz = int(self.headers[
            'SignalHeaders'][
            self.channels.index(channel_name)]['sample_rate'])

        with edfreader.EdfReader(self.edf_file_path) as f:
            signals = f.readSignal(self.channels.index(channel_name))

        freq_ns = 1_000_000_000 / self.sampling_frequency_hz
        df_ecg = pd.DataFrame(signals,
                              columns=['signal'],
                              index=pd.date_range(self.startdate,
                                                  periods=len(signals),
                                                  freq=f'{freq_ns}ns'
                                                  ))

        df_ecg = df_ecg[(df_ecg.index >= start_time) &
                        (df_ecg.index < end_time)]

        return df_ecg
