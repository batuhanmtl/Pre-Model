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
    print(dataframe.dtypes)
    print("##################### Head #####################")
    print("##################### Tail #####################")
    print(dataframe.tail(head))
    print("##################### NA #####################")
    print(dataframe.isnull().sum())
    print("##################### Quantiles #####################")
    print(dataframe.describe([0, 0.05, 0.50, 0.95, 0.99, 1]).T)

# Category Variable Analysis

def cat_summary(dataframe, col_name, plot=False):
    """
        It shows the ratio and frequencies of the categorical variables in the variable with each other.
        :param dataframe: Data set(Pandas.DataFrame)
        :param col_name: The variable name-string- you want to see the frequency and ratios of.
        :param plot: bool
        :return: no return
        """

    if dataframe[col_name].dtypes == "bool":
        dataframe[col_name] = dataframe[col_name].astype(int)

        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)