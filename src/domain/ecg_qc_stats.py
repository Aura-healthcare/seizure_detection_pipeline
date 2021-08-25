from statistics import median
from typing import Optional, Tuple
from itertools import groupby


def percentage_noisy_segments(noisy_segments: list) -> Optional[float]:
    if len(noisy_segments) == 0:
        return None
    else:
        return round((1-sum(noisy_segments)/len(noisy_segments))*100, 2)


def length_record(noisy_segments: list, length_chunk: int) -> int:
    return length_chunk*len(noisy_segments)


def noise_free_intervals_stats(noisy_segments: list, length_chunk: int) \
        -> Tuple[Optional[int], Optional[int], Optional[float]]:
    if len(noisy_segments) == 0:
        return None, None, None
    else:
        intervals = [sum(1 for _ in group)
                     for bit, group in groupby(noisy_segments) if bit]
        if len(intervals) == 0:
            return 0, 0, 0
        else:
            minimun = min(intervals)*length_chunk
            maximum = max(intervals)*length_chunk
            med = median(intervals)*length_chunk
            return minimun, maximum, med
