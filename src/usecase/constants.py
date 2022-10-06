from enum import Enum

class SeasonalFeature(Enum):
    month = "month"
    dayOfWeek = "dayOfWeek"
    hour = "hour"
    minute = "minute"
    second = "second"

class TempFeaturesOperation(Enum):
    mean = 1
    std = 2
    sum = 3


class TempFeaturesPeriod(Enum):
    """
    This class define lags for contextuals operations on features.
    p30 = period 30 seconds
    p60 = period 60 seconds
    p120 = period 120 seconds
    """
    p30 = 30
    p60 = 60
    p120 = 120