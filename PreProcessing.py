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