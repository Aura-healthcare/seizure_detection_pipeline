import json
import os

import click
import numpy as np
import pandas as pd

from hrvanalysis import remove_outliers, remove_ectopic_beats, interpolate_nan_values, get_time_domain_features, get_csi_cvi_features, get_sampen, get_poincare_plot_features, get_frequency_domain_features

SHORT_WINDOW = 10000  # short window lasts 10 seconds - 10 000 milliseconds
MEDIUM_WINDOW = 60000  # medium window lasts 60 secondes
LARGE_WINDOW = 150000  # large window lasts 2 minutes 30 seconds
FEATURES_KEY_TO_INDEX = {
    'interval_index': 0,
    'interval_start_time': 1,  # en milliseconds
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
    'ratio_sd2_sd1': 28,
    'label': 29
}

RR_INTERVALS_FOLDER = 'output/clean_rr_intervals'
OUTPUT_FOLDER = 'output/features'


def write_features_csv(features: pd.DataFrame,
                       infos: str) -> str:
    os.makedirs(OUTPUT_FOLDER, exist_ok=True)
    filename = f"{infos}.csv"
    filepath = os.path.join(OUTPUT_FOLDER, filename)
    features.to_csv(filepath, sep=',', index=True)
    return filename


def get_rr_intervals_on_window(rr_timestamps, rrs, offset, window):
    rr_indices = np.logical_and(rr_timestamps >= offset, rr_timestamps < (offset + window))
    return rrs[rr_indices]


def get_clean_intervals(rrs):
    # This remove outliers from signal
    rr_intervals_without_outliers = remove_outliers(rr_intervals=rrs,
                                                    low_rri=300, high_rri=1800)
    # This replace outliers nan values with linear interpolation
    interpolated_rr_intervals = interpolate_nan_values(rr_intervals=rr_intervals_without_outliers,
                                                       interpolation_method="linear")

    # This remove ectopic beats from signal
    nn_intervals_list = remove_ectopic_beats(rr_intervals=interpolated_rr_intervals, method="malik")
    # This replace ectopic beats nan values with linear interpolation
    interpolated_nn_intervals = interpolate_nan_values(rr_intervals=nn_intervals_list)

    return interpolated_nn_intervals


def compute_short_term_features_on_interval(features, i, rr_timestamps, rrs):
    # Adding indexes
    features[i][FEATURES_KEY_TO_INDEX["interval_index"]] = i
    features[i][FEATURES_KEY_TO_INDEX["interval_start_time"]] = i * SHORT_WINDOW

    rrs_on_interval = get_rr_intervals_on_window(rr_timestamps, rrs, i * SHORT_WINDOW, SHORT_WINDOW)
    if(len(rrs_on_interval) == 0):
        raise ValueError("No RR intervals")

    clean_rrs = get_clean_intervals(rrs_on_interval)
    time_domain_features = get_time_domain_features(clean_rrs)
    for key in time_domain_features.keys():
        features[i][FEATURES_KEY_TO_INDEX[key]] = time_domain_features[key]


def compute_medium_term_features_on_interval(features, i, rr_timestamps, rrs, medium_window_offset):
    if (i * SHORT_WINDOW) > MEDIUM_WINDOW:
        rr_on_medium_intervals = get_rr_intervals_on_window(rr_timestamps, rrs, (i - medium_window_offset) * SHORT_WINDOW, MEDIUM_WINDOW)
        clean_rrs = get_clean_intervals(rr_on_medium_intervals)

        if len(rr_on_medium_intervals) == 0:
            raise ValueError("No RR intervals")

        # Compute non linear features
        cvi_csi_features = get_csi_cvi_features(clean_rrs)
        for key in cvi_csi_features.keys():
            features[i][FEATURES_KEY_TO_INDEX[key]] = cvi_csi_features[key]

        sampen = get_sampen(clean_rrs)
        features[i][FEATURES_KEY_TO_INDEX["sampen"]] = sampen["sampen"]

        poincare_features = get_poincare_plot_features(clean_rrs)
        for key in poincare_features.keys():
            features[i][FEATURES_KEY_TO_INDEX[key]] = poincare_features[key]


def compute_long_term_features_on_interval(features, i, rr_timestamps, rrs, large_window_offset):
    if (i * SHORT_WINDOW) > LARGE_WINDOW:
        rr_on_large_intervals = get_rr_intervals_on_window(rr_timestamps, rrs, (i - large_window_offset) * SHORT_WINDOW, LARGE_WINDOW)

        if len(rr_on_large_intervals) == 0:
            raise ValueError("No RR intervals")

        clean_rrs = get_clean_intervals(rr_on_large_intervals)

        # Compute frequency domain features
        frequency_domain_features = get_frequency_domain_features(clean_rrs)
        for key in frequency_domain_features.keys():
            if key in FEATURES_KEY_TO_INDEX:
                features[i][FEATURES_KEY_TO_INDEX[key]] = frequency_domain_features[key]


def compute_labels_on_interval(features, i, background_intervals, seizure_intervals):
    short_interval_s = SHORT_WINDOW * 0.001
    interval_range = [[i * short_interval_s, (i+1) * short_interval_s]]
    intersec_interval_background = intersections(interval_range, background_intervals)
    intersec_interval_seizure = intersections(interval_range, seizure_intervals)

    sum_background = 0
    for interval in intersec_interval_background:
        sum_background += (interval[1] - interval[0])

    sum_seizure = 0
    for interval in intersec_interval_seizure:
        sum_seizure += (interval[1] - interval[0])

    ratio_background = sum_background / short_interval_s
    ratio_seizure = sum_seizure / short_interval_s

    if (ratio_background + ratio_seizure) < 0.9:
        features[i][FEATURES_KEY_TO_INDEX["label"]] = np.NaN

    else:
        features[i][FEATURES_KEY_TO_INDEX["label"]] = ratio_seizure


def get_annotations_data(annotations_filename):
    background_intervals = []
    seizure_intervals = []
    annotations_data = json.load(open(annotations_filename, 'r'))
    background_intervals = annotations_data["background"]
    seizure_intervals = annotations_data["seizure"]

    return background_intervals, seizure_intervals


def intersections(a, b):
    ranges = []
    i = j = 0
    while i < len(a) and j < len(b):
        a_left, a_right = a[i]
        b_left, b_right = b[j]

        if a_right < b_right:
            i += 1
        else:
            j += 1

        if a_right >= b_left and b_right >= a_left:
            end_pts = sorted([a_left, a_right, b_left, b_right])
            middle = [end_pts[1], end_pts[2]]
            ranges.append(middle)

    ri = 0
    while ri < len(ranges)-1:
        if ranges[ri][1] == ranges[ri+1][0]:
            ranges[ri:ri+2] = [[ranges[ri][0], ranges[ri+1][1]]]

        ri += 1

    return ranges


def compute_hrvanalysis_features(rr_intervals_file: str) -> str:
    '''
    Computes features from RR-intervals (from a csv file),
    and writes them in another csv file.
    '''
    df_rr_intervals = pd.read_csv(
        os.path.join(RR_INTERVALS_FOLDER, rr_intervals_file),
        sep=',',
        index_col=0
    )
    rr_intervals = list(df_rr_intervals['rr_interval'])
    # FAIRE DES CHOSES
    df_features = get_df_hrv_features(rr_intervals)
    infos = rr_intervals_file.split('.')[0]
    filename = write_features_csv(df_features, infos)
    return filename


@click.command()
@click.option('--rr-intervals-file', required=True)
def main(rr_intervals_file: str) -> None:
    _ = compute_hrvanalysis_features(rr_intervals_file)


if __name__ == '__main__':
    main()


parser = OptionParser()
parser.add_option("-i", "--input_file", dest="input_filename",
                  help="input file path")
parser.add_option("-o", "--output_file", dest="output_filename",
                  help="output file path")
parser.add_option("-a", "--annotations_file", dest="annotations_filename",
                  help="annotations file path")
parser.add_option("-q", "--qrs_detector", dest="qrs_detector_used",
                  help="QRS detector used - available: 1/ pan-tompkins, 2/ swt - Stationnary Wavelets tramsform, 3/ XQRS")
parser.add_option("-v", "--verbose",
                  action="store_const", const=1, dest="verbose")
(options, args) = parser.parse_args()

if not options.input_filename.endswith(".json"):
    raise ValueError("Invalid input filepath")
    exit()

if not options.output_filename.endswith(".json"):
    raise ValueError("Invalid output filepath ")
    exit()

if not options.qrs_detector_used or (options.qrs_detector_used.lower() != "gqrs" and options.qrs_detector_used.lower() != "xqrs" and options.qrs_detector_used.lower() != "hamilton" and options.qrs_detector_used.lower() != "engelsee" and options.qrs_detector_used.lower() != "swt"):
    raise ValueError("Invalid QRS Detector ")
    exit()

output_filename = options.output_filename
input_filename = options.input_filename
qrs_detector = options.qrs_detector_used.lower()
annotations_filename = options.annotations_filename
#annotations_filename = options.annotations_file

try:
    # Get QRS frames / RR intervals data
    raw_data = json.load(open(input_filename))
    background_intervals, seizure_intervals = get_annotations_data(annotations_filename)

    rrs = np.asarray(raw_data[qrs_detector]["rr_intervals"])
    rr_timestamps = np.cumsum(rrs)

    duration = rr_timestamps[-1] + rrs[-1]
    n_short_intervals = (int)(duration / SHORT_WINDOW) + 1
    medium_window_offset = MEDIUM_WINDOW / SHORT_WINDOW
    large_window_offset = LARGE_WINDOW / SHORT_WINDOW

    features =  np.empty([n_short_intervals, len(FEATURES_KEY_TO_INDEX.keys())])
    features[:] = np.NaN

    # Sequence features computations in ten seconds intervals
    for i in range(0, n_short_intervals):
        try:
            compute_labels_on_interval(features, i, background_intervals, seizure_intervals)
        except Exception as e:
            print("Interval " + str(i) + " - label computation issue - " + str(e))

        try:
            compute_short_term_features_on_interval(features, i, rr_timestamps, rrs)
        except Exception as e:
            print("Interval " + str(i) + "- computation issue on short term features " + str(e))

        try:
            compute_medium_term_features_on_interval(features, i, rr_timestamps, rrs, medium_window_offset)
        except Exception as e:
            print("Interval " + str(i) + "- computation issue on medium term features" + str(e))

        try:
            compute_long_term_features_on_interval(features, i, rr_timestamps, rrs, medium_window_offset)
        except Exception as e:
            print("Interval " + str(i) + "- computation issue on long term features" + str(e))

    keys = [key for key in FEATURES_KEY_TO_INDEX.keys()]
    data = {"keys": keys,
            "features": features.tolist()}
    json.dump(data, open(output_filename, "w"))
except Exception as e:
   print(e)