import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def check_df(dataframe, head=5):
    """
    Returns general information about the data frame.
    :param dataframe: The data frame whose information we want.
    :param head: the number of data we want to observe starting from the beginning(default head = 5)
    :return: no return
    """
    print("##################### Shape #####################")
    print(dataframe.shape)
    print("##################### Types #####################")