from enum import Enum

class QuantileNames(Enum):
    PERCENT_5 = 'PERCENT_5'
    PERCENT_20 = 'PERCENT_20'
    PERCENT_50 = 'PERCENT_50'
    PERCENT_80 = 'PERCENT_80'
    PERCENT_95 = 'PERCENT_95'

class QuantileVals(Enum):
    PERCENT_5 = 0.05
    PERCENT_20 = 0.2
    PERCENT_50 = 0.5
    PERCENT_80 = 0.8
    PERCENT_95 = 0.95