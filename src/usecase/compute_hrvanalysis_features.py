import os
import argparse
import sys
from typing import List
import numpy as np
import pandas as pd

from hrvanalysis import remove_outliers, remove_ectopic_beats, \
    interpolate_nan_values, get_time_domain_features, get_csi_cvi_features, \
    get_sampen, get_poincare_plot_features, get_frequency_domain_features

sys.path.append('.')
from src.usecase.utilities import convert_args_to_dict

SHORT_WINDOW = 10000  # short window lasts 10 seconds - 10 000 milliseconds
MEDIUM_WINDOW = 60000  # medium window lasts 60 secondes
LARGE_WINDOW = 150000  # large window lasts 2 minutes 30 seconds
# REFACT & chiffres
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

    def __init__(self,
                 rr_timestamps: np.array,
                 rr_intervals: np.array,
                 features_key_to_index: dict,
                 short_window: int,
                 medium_window: int,
                 large_window: int):

        # Sets arguments as class attributes
        self.__dict__.update(locals())

        # Creates windows offsets, based on short_window size
        for size in ['short', 'medium', 'large']:
            self.__dict__.update(
                {f'{size}_window_offset': (
                    self.__dict__[f'{size}_window'] / self.short_window
                    )})

        # Compute quantity of short intervals to iterate
        self.n_short_intervals = int(
            (self.rr_timestamps[-1] + self.rr_intervals[-1])
            / self.short_window) \
            + 1

        # Initialize "features" variable to store results
        self.features = np.empty([self.n_short_intervals,
                                  len(self.features_key_to_index.keys())])
        self.features[:] = np.NaN

        # Computes hrv features and store them in variable "features"
        self.compute()

    def compute(self):

        for _index in range(self.n_short_intervals):

            (self.features
                [_index]
                [self.features_key_to_index
                    ["interval_index"]]
             ) = _index

            (self.features
                [_index]
                [self.features_key_to_index
                    ["interval_start_time"]]) = _index * self.short_window

            for _size in ['short', 'medium', 'large']:
                _window_size = self.__dict__[f'{_size}_window']
                if (_index * self.short_window) >= _window_size:
                    # >= ou >(intialement) ?
                    _rr_on_intervals = self.get_rr_intervals_on_window(
                        index=_index,
                        size=_size)
                    _clean_rrs = self.get_clean_intervals(_rr_on_intervals)
                    if _size == 'short':
                        self.compute_time_domain_features(_index, _clean_rrs)
                    elif _size == 'medium':
                        self.compute_non_linear_features(_index, _clean_rrs)
                    elif _size == 'large':
                        self.compute_time_domain_features(_index, _clean_rrs)

    def get_rr_intervals_on_window(self,
                                   index: int,
                                   size: str) -> np.array:
        """
        From an index (COMPETE) and a window size, filter rr_intervals
        on this window
        """
        window = self.__dict__[f'{size}_window']
        offset = (index - self.__dict__[f"{size}_window_offset"]) \
            * self.short_window

        rr_indices = np.logical_and(
            self.rr_timestamps >= offset,
            self.rr_timestamps < (offset + window))

        rr_on_intervals = self.rr_intervals[rr_indices]

        if len(rr_on_intervals) == 0:
            raise ValueError("No RR intervals")

        return rr_on_intervals

    def get_clean_intervals(self,
                            rr_intervals: np.array) -> np.array:

        # Removes outliers from signal
        rr_intervals_without_outliers = remove_outliers(
            rr_intervals=rr_intervals,
            low_rri=300,
            high_rri=1800)

        # Replaces outliers nan values with linear interpolation
        interpolated_rr_intervals = interpolate_nan_values(
            rr_intervals=rr_intervals_without_outliers,
            interpolation_method="linear")

        # Removes ectopic beats from signal
        nn_intervals_list = remove_ectopic_beats(
            rr_intervals=interpolated_rr_intervals,
            method="malik")
        # Replaces ectopic beats nan values with linear interpolation
        interpolated_nn_intervals = interpolate_nan_values(
           rr_intervals=nn_intervals_list)

        return interpolated_nn_intervals

    def compute_time_domain_features(self,
                                     index: int,
                                     clean_rrs: np.array):
        '''
        Computes non time domain features from HRVanalysis. These features are
        meant for short window features.
        '''
        try:
            time_domain_features = get_time_domain_features(clean_rrs)
            for key in time_domain_features.keys():
                (self.features
                    [index]
                    [self.features_key_to_index[key]]) = \
                        time_domain_features[key]

        except Exception as e:
            print(f'Interval {str(index)} - '
                  f'computation issue on short window features {str(e)}')

    def compute_non_linear_features(self,
                                    index: int,
                                    clean_rrs: np.array):
        '''
        Computes non linear features from HRVanalysis. These features are meant
        for medium window features.
        '''
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
        '''
        Computes frequency domain features from HRV analysis. These features
        are meant for large window features.
        '''
        try:
            frequency_domain_features = get_frequency_domain_features(
                clean_rrs=clean_rrs)
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
                                 short_window: int = SHORT_WINDOW,
                                 medium_window: int = MEDIUM_WINDOW,
                                 large_window: int = LARGE_WINDOW,
                                 features_key_to_index: dict =
                                 FEATURES_KEY_TO_INDEX) -> str:
    '''
    Computes features from RR-intervals (from a csv file),
    and writes them in another csv file.
    '''
    df_rr_intervals = pd.read_csv(
        os.path.join(rr_intervals_file_path),
        index_col=0)
    rr_intervals = df_rr_intervals['rr_interval'].values
    rr_timestamps = np.cumsum(rr_intervals)

    features_computer = compute_features(
        rr_timestamps=rr_timestamps,
        rr_intervals=rr_intervals,
        features_key_to_index=features_key_to_index,
        short_window=short_window,
        medium_window=medium_window,
        large_window=large_window)

    df_features = pd.DataFrame(
        data=features_computer.features,
        columns=features_key_to_index)

    # EXPORT
    os.makedirs(output_folder, exist_ok=True)
    file_parsing = rr_intervals_file_path.split('/')[-1].split('.')[0]
    filename = f'{file_parsing}.csv'
    filepath = os.path.join(output_folder, filename)
    df_features.to_csv(filepath, sep=',', index=True)

    return filepath


def parse_compute_hrvanalysis_features_args(
        args_to_parse: List[str]) -> argparse.Namespace:

    parser = argparse.ArgumentParser(description='CLI parameter input')
    parser.add_argument('--rr-intervals-file-path',
                        dest='rr_intervals_file_path',
                        required=True)
    parser.add_argument('--output-folder',
                        dest='output_folder')
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
