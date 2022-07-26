import numpy as np
import pandas as pd
from matplotlib import pyplot as plt


# Outlier Thresholds

def outlier_thresholds(dataframe, col_name, q1=0.25, q3=0.75):
    """
    Calculates a range for outliers.
    :param dataframe: Pandas.DataFrame
    :param col_name:  string
    Variable name whose outliers will be determined
    :param q1: quantile 1 int or float (optional default 0.25)
    :param q3: quantile 3 int or float (optional default 0.75)
    :return: int or float , low limit and up limit
    """

    quartile1 = dataframe[col_name].quantile(q1)

    quartile3 = dataframe[col_name].quantile(q3)

    interquantile_range = quartile3 - quartile1

    up_limit = quartile3 + 1.5 * interquantile_range

    low_limit = quartile1 - 1.5 * interquantile_range

    return low_limit, up_limit


# Checking for outliers

def check_outlier(dataframe, col_name):
    """
    It checks if there are any outliers.
    :param dataframe: Pandas.DataFrame
    :param col_name: string
    Variable name whose outliers will be check
    :return: bool
    """

    low_limit, up_limit = outlier_thresholds(dataframe, col_name)

    if dataframe[(dataframe[col_name] > up_limit) | (dataframe[col_name] < low_limit)].any(axis=None):
        return True

    else:
        return False