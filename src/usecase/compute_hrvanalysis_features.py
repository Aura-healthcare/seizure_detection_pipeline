"""
This script is used to compute hrvanalysis features on RR intervals.

copyright (c) 2021 association aura
spdx-license-identifier: gpl-3.0
"""
import argparse
import os
import sys
from typing import List

import numpy as np
import pandas as pd

from hrvanalysis import remove_outliers, remove_ectopic_beats, \
    interpolate_nan_values, get_time_domain_features, get_csi_cvi_features, \
    get_sampen, get_poincare_plot_features, get_frequency_domain_features

sys.path.append('.')
from src.usecase.utilities import convert_args_to_dict, generate_output_path

SLIDING_WINDOW = 1_000
SHORT_WINDOW = 10_000  # short window lasts 10 seconds - 10 000 milliseconds
MEDIUM_WINDOW = 60_000  # medium window lasts 60 secondes
LARGE_WINDOW = 150_000  # large window lasts 2 minutes 30 seconds

OUTPUT_FOLDER = 'output/features'

FEATURES_KEY_TO_INDEX = {
    'interval_index': 0,
    'interval_start_time': 1,  # in milliseconds
    'mean_nni': 2,
    'sdnn': 3,
    'sdsd': 4,
    'nni_50': 5,
    'pnni_50': 6,
    'nni_20': 7,
    'pnni_20': 8,
    'rmssd': 9,
    'median_nni': 10,
    'range_nni': 11,
    'cvsd': 12,
    'cvnni': 13,
    'mean_hr': 14,
    'max_hr': 15,
    'min_hr': 16,
    'std_hr': 17,
    'lf': 18,
    'hf': 19,
    'vlf': 20,
    'lf_hf_ratio': 21,
    'csi': 22,
    'cvi': 23,
    'Modified_csi': 24,
    'sampen': 25,
    'sd1': 26,
    'sd2': 27,
    'ratio_sd2_sd1': 28
}


class compute_features:
    """
    Compute hrvanalysis features in attribute "features".

    This class compute Heart Rate Variability features based on detected
    RR-intervals. Some features are efficent only on some time periods,
    therefore there are 3 time windows to compute when time availiable is
    long enough. A sliding time window allows to scroll through timeline to
    to compute at different start points.

    *compute* method is used to compute all possible features directly. They
    will be stored in attribute *features*.
    """

    def __init__(self,
                 rr_timestamps: np.array,
                 rr_intervals: np.array,
                 features_key_to_index: dict,
                 sliding_window: int,
                 short_window: int,
                 medium_window: int,
                 large_window: int):
        """
        Initialize all required attributes to compute Heart Rate Variation.

        Parameters
        ----------
        rr_timestamps : np.array
            Computed RR intervals timestamps as an array
        rr_intervals : np.array
            RR intervals are each timestamps as an array
        features_key_to_index : dict
            Features to index as a dictionnary. Needed to add data at the right
            position
        sliding_window : int
            Period in milliseconds that will be used to slide on the timeline
        short_window : int
            Size of the short time window, in milliseconds, which fits to
            compute time domain features
        medium_window : int
            Size of the medium time window, in milliseconds, which fits to
            compute
        large_window : int
            Size of the large time window, in milliseconds, which fits to
            compute non linear features
        """
        # Sets arguments as class attributes
        self.__dict__.update(locals())

        # Compute quantity of short intervals to iterate
        self.n_short_intervals = int(
            (self.rr_timestamps[-1] / self.sliding_window) + 1)

        # Initialize "features" variable to store results
        self.features = np.empty([self.n_short_intervals,
                                  len(self.features_key_to_index.keys())])
        self.features[:] = np.NaN

        # Computes hrv features and store them in variable "features"
        self.compute()

    def compute(self):
        """
        Compute possible all features.

        According to different threshold, compute possible features from
        hrvanalysis:
        * time domain
        * non linear
        * frequency domain
        These features are stored in class attribute "features".
        """
        for _index in range(self.n_short_intervals):

            (self.features
                [_index]
                [self.features_key_to_index
                    ["interval_index"]]
             ) = _index

            (self.features
                [_index]
                [self.features_key_to_index
                    ["interval_start_time"]]) = _index * self.sliding_window

            for _size in ['short', 'medium', 'large']:
                _window_size = self.__dict__[f'{_size}_window']
                if (_index * self.sliding_window) >= _window_size:
                    # >= ou >(intialement) ?
                    _rr_on_intervals = self.get_rr_intervals_on_window(
                        index=_index,
                        size=_size)

                    # Handling to improve
                    if len(_rr_on_intervals) == 0:
                        print("No RR intervals")
                        continue

                    _clean_rrs = self.get_clean_intervals(_rr_on_intervals)
                    if _size == 'short':
                        self.compute_time_domain_features(_index, _clean_rrs)
                    elif _size == 'medium':
                        self.compute_non_linear_features(_index, _clean_rrs)
                    elif _size == 'large':
                        self.compute_frequency_domain_features(_index,
                                                               _clean_rrs)

    def get_rr_intervals_on_window(self,
                                   index: int,
                                   size: str) -> np.array:
        """
        Filter RR interval corresponding to a time window before index.

        Parameters
        ----------
        index : int
            Index of the timeline splitted by sliding window size
        size : str
            Size of the window, choices are [small, medium, large]

        Returns
        -------
        rr_on_intervals : np.array
            RR intervals on the select time window.
        """
        starting_timestamp = index * self.sliding_window
        window = self.__dict__[f'{size}_window']
        offset = (starting_timestamp - window)
        rr_indices = np.logical_and(
            self.rr_timestamps >= offset,
            self.rr_timestamps < starting_timestamp)

        rr_on_intervals = self.rr_intervals[rr_indices]

        return rr_on_intervals

    def get_clean_intervals(self,
                            rr_intervals: np.array) -> np.array:
        """
        Clean and returns RR intervals.

        Parameters
        ----------
        rr_intervals : np.array
            RR intervals to clean as an array

        Returns
        -------
        interpolated_nn_intervals : np.array
            The same RR intervals cleaned from outliers
        """
        # Removes outliers from signal
        rr_intervals_without_outliers = remove_outliers(
            rr_intervals=rr_intervals,
            low_rri=300,
            high_rri=1800,
            verbose=False)

        # Replaces outliers nan values with linear interpolation
        interpolated_rr_intervals = interpolate_nan_values(
            rr_intervals=rr_intervals_without_outliers,
            interpolation_method="linear")

        # Removes ectopic beats from signal
        nn_intervals_list = remove_ectopic_beats(
            rr_intervals=interpolated_rr_intervals,
            method="malik",
            verbose=False)

        # Replaces ectopic beats nan values with linear interpolation
        interpolated_nn_intervals = interpolate_nan_values(
            rr_intervals=nn_intervals_list)

        return interpolated_nn_intervals

    def compute_time_domain_features(self,
                                     index: int,
                                     clean_rrs: np.array):
        """
        Compute time domain features from HRVanalysis.

        These features are meant for short window features and are added to
        features attribute.

        Parameters
        ----------
        index : int
            Index of the timeline splitted by sliding window size
        clean_rrs : np.array
            Clean RR intervals
        """
        try:
            time_domain_features = get_time_domain_features(clean_rrs)
            for key in time_domain_features.keys():
                (self.features
                    [index]
                    [self.features_key_to_index[key]]) = \
                    time_domain_features[key]

        except ZeroDivisionError:
            pass

        except Exception as e:
            print(f'Interval {str(index)} - '
                  f'computation issue on short window features {str(e)}')

    def compute_non_linear_features(self,
                                    index: int,
                                    clean_rrs: np.array):
        """
        Compute non linear features from HRVanalysis.

        These features are meant for medium window features and are added to
        features attribute.

        Parameters
        ----------
        index : int
            Index of the timeline splitted by sliding window size
        clean_rrs : np.array
            Clean RR intervals
        """
        try:
            cvi_csi_features = get_csi_cvi_features(clean_rrs)
            for _key in cvi_csi_features.keys():
                (self.features
                    [index]
                    [self.features_key_to_index[_key]]) = \
                    cvi_csi_features[_key]

            sampen = get_sampen(clean_rrs)
            (self.features
                [index]
                [self.features_key_to_index["sampen"]]) = sampen["sampen"]

            poincare_features = get_poincare_plot_features(clean_rrs)
            for key in poincare_features.keys():
                (self.features
                    [index]
                    [self.features_key_to_index[key]]) = poincare_features[key]

        except Exception as e:
            print(f'Interval {str(index)} - '
                  f'computation issue on medium window features {str(e)}')

    def compute_frequency_domain_features(self,
                                          index: int,
                                          clean_rrs: np.array):
        """
        Compute frequency domain features from HRV analysis.

        These features are meant for large window features and are added to
        features attribute.

        Parameters
        ----------
        index : int
            Index of the timeline splitted by sliding window size
        clean_rrs : np.array
            Clean RR intervals
        """
        try:
            frequency_domain_features = get_frequency_domain_features(
                nn_intervals=clean_rrs)
            for _key in frequency_domain_features.keys():
                if _key in self.features_key_to_index:
                    (self.features
                        [index]
                        [self.features_key_to_index[_key]]) = \
                        frequency_domain_features[_key]
        except Exception as e:
            print(f"Interval {str(index)} - "
                  f"computation issue on large window features {str(e)}")


def compute_hrvanalysis_features(rr_intervals_file_path: str,
                                 output_folder: str = OUTPUT_FOLDER,
                                 sliding_window: int = SLIDING_WINDOW,
                                 short_window: int = SHORT_WINDOW,
                                 medium_window: int = MEDIUM_WINDOW,
                                 large_window: int = LARGE_WINDOW,
                                 features_key_to_index: dict =
                                 FEATURES_KEY_TO_INDEX) -> str:
    """
    Compute HRV features from RR intervals.

    From a csv file including RR-intervals, computes HRVanalysis features and
    export them as a CSV.

    Parameters
    ----------
    rr_intervals_file_path: str
        Path of RR intervals csv file
    output_folder : str
        Path of the output folder
    sliding_window : int
        Period in milliseconds that will be used to slide on the timeline
    short_window : int
        Size of the short time window, in milliseconds, which fits to compute
        time domain features
    medium_window : int
        Size of the medium time window, in milliseconds, which fits to compute
    large_window : int
        Size of the large time window, in milliseconds, which fits to compute
        non linear features
    features_key_to_index : dict
        A dictionnary with the name and index of features to include. Should
        not be changed

    Returns
    -------
    output_file_path :
        Output file of computed HRVanalysis features
    """
    df_rr_intervals = pd.read_csv(
        os.path.join(rr_intervals_file_path))
    rr_intervals = df_rr_intervals['rr_interval'].values
    rr_timestamps = np.cumsum(rr_intervals)
    start_timestamp = df_rr_intervals['timestamp'].values[0]  # MODIFY

    features_computer = compute_features(
        rr_timestamps=rr_timestamps,
        rr_intervals=rr_intervals,
        features_key_to_index=features_key_to_index,
        sliding_window=sliding_window,
        short_window=short_window,
        medium_window=medium_window,
        large_window=large_window)

    df_features = pd.DataFrame(
        data=features_computer.features,
        columns=features_key_to_index)

    df_features['timestamp'] = df_features['interval_start_time'].apply(
        lambda x: pd.Timestamp(start_timestamp,
                               tz=None) + pd.Timedelta(x, unit='milliseconds'))

    # EXPORT
    output_file_path = generate_output_path(
        input_file_path=rr_intervals_file_path,
        output_folder=output_folder,
        format='csv',
        prefix='feats')

    df_features['filename'] = output_file_path

    df_features.to_csv(output_file_path, sep=',', index=False)

    return output_file_path


def parse_compute_hrvanalysis_features_args(
        args_to_parse: List[str]) -> argparse.Namespace:
    """
    Parse arguments for adaptable input.

    parameters
    ----------
    args_to_parse : List[str]
        List of the element to parse. Should be sys.argv[1:] if args are
        inputed via CLI

    returns
    -------
    args : argparse.Namespace
        Parsed arguments
    """
    parser = argparse.ArgumentParser(description='CLI parameter input')
    parser.add_argument('--rr-intervals-file-path',
                        dest='rr_intervals_file_path',
                        required=True)
    parser.add_argument('--output-folder',
                        dest='output_folder')
    parser.add_argument('--sliding-window',
                        dest='sliding_window',
                        type=int)
    parser.add_argument('--short-window',
                        dest='short_window',
                        type=int)
    parser.add_argument('--medium-window',
                        dest='medium_window',
                        type=int)
    parser.add_argument('--large-window',
                        dest='large_window',
                        type=int)
    args = parser.parse_args(args_to_parse)

    return args


if __name__ == '__main__':
    args = parse_compute_hrvanalysis_features_args(sys.argv[1:])
    args_dict = convert_args_to_dict(args)
    compute_hrvanalysis_features(**args_dict)
