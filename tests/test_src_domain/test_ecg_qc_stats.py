from src.domain.ecg_qc_stats import percentage_noisy_segments, length_record, noise_free_intervals_stats


def test_percentage_noisy_segments_should_return_0_when_no_noise():
    # Given
    noise_list = [1, 1, 1, 1, 1]
    # When
    result = percentage_noisy_segments(noise_list)
    # Then
    assert result == 0


def test_percentage_noisy_segments_should_return_100_when_only_noise():
    # Given
    noise_list = [0, 0, 0, 0, 0, 0, 0]
    # When
    result = percentage_noisy_segments(noise_list)
    # Then
    assert result == 100


def test_percentage_noisy_segments_should_return_correct_percentage():
    # Given
    noise_list = [1, 0, 0, 0, 1, 1, 0]
    # When
    result = percentage_noisy_segments(noise_list)
    # Then
    assert result == 57.14


def test_percentage_noisy_segments_should_return_None_when_empty_list():
    # Given
    noise_list = []
    # When
    result = percentage_noisy_segments(noise_list)
    # Then
    assert result is None


def test_length_record_should_return_correct_length():
    # Given
    noise_list = [1, 0, 1, 1, 1, 0]
    length_chunk = 4
    # When
    result = length_record(noise_list, length_chunk)
    # Then
    assert result == 24


def test_length_record_should_return_0_when_empty_list():
    # Given
    noise_list = []
    length_chunk = 4
    # When
    result = length_record(noise_list, length_chunk)
    # Then
    assert result == 0


def test_noise_free_intervals_stats_min_should_be_correct():
    # Given
    noise_list = [1, 0, 1, 1, 1, 0, 0]
    length_chunk = 4
    # When
    minimum, _, _ = noise_free_intervals_stats(noise_list, length_chunk)
    # Then
    assert minimum == 4


def test_noise_free_intervals_stats_max_should_be_correct():
    # Given
    noise_list = [1, 0, 1, 1, 1, 0, 0]
    length_chunk = 4
    # When
    _, maximum, _ = noise_free_intervals_stats(noise_list, length_chunk)
    # Then
    assert maximum == 12


def test_noise_free_intervals_stats_median_should_be_correct():
    # Given
    noise_list = [1, 0, 1, 1, 1, 0, 0, 1, 1, 1, 1, 0, 1, 1, 0]
    length_chunk = 4
    # When
    _, _, med = noise_free_intervals_stats(noise_list, length_chunk)
    # Then
    assert med == 10


def test_noise_free_intervals_stats_should_return_0_when_only_noise():
    # Given
    noise_list = [0, 0, 0, 0, 0]
    length_chunk = 4
    expected_result = (0, 0, 0)
    # When
    actual_result = noise_free_intervals_stats(noise_list, length_chunk)
    # Then
    assert actual_result == expected_result


def test_noise_free_intervals_stats_should_return_None_when_empty_list():
    # Given
    noise_list = []
    length_chunk = 4
    expected_result = (None, None, None)
    # When
    actual_result = noise_free_intervals_stats(noise_list, length_chunk)
    # Then
    assert actual_result == expected_result
