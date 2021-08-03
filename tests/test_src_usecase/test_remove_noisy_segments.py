import pandas as pd

from src.usecase.remove_noisy_segments import remove_noisy_segments_from_df


def test_second_segment_is_removed_and_no_other():
    # Given
    length_chunk = 4
    sampling_frequency = 256

    index = [
        pd.to_datetime("2021-01-01 00:00:00"),
        pd.to_datetime("2021-01-01 00:00:01"),
        pd.to_datetime("2021-01-01 00:00:03"),
        pd.to_datetime("2021-01-01 00:00:04"),
        pd.to_datetime("2021-01-01 00:00:05"),
        pd.to_datetime("2021-01-01 00:00:08"),
        pd.to_datetime("2021-01-01 00:00:10")
        ]

    data = [
        [0, 1000],
        [256, 2000],
        [768, 1000],
        [1024, 1000],
        [1280, 3000],
        [2048, 2000],
        [2560, 2000]
        ]

    df_rr_intervals = pd.DataFrame(data, columns=['frame', 'rr_interval'], index=index)

    list_noisy_segments = [1, 0, 1]

    index_expected = [
        pd.to_datetime("2021-01-01 00:00:00"),
        pd.to_datetime("2021-01-01 00:00:01"),
        pd.to_datetime("2021-01-01 00:00:03"),
        pd.to_datetime("2021-01-01 00:00:08"),
        pd.to_datetime("2021-01-01 00:00:10")
        ]

    data_expected = [
        [0, 1000],
        [256, 2000],
        [768, 1000],
        [2048, 2000],
        [2560, 2000]
        ]

    expected_result = pd.DataFrame(data_expected, columns=['frame', 'rr_interval'], index=index_expected)

    # When
    actual_result = remove_noisy_segments_from_df(df_rr_intervals, list_noisy_segments, length_chunk, sampling_frequency)

    # Then
    pd.testing.assert_frame_equal(actual_result, expected_result)
