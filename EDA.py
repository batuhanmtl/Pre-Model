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

    else:
        print(pd.DataFrame({col_name: dataframe[col_name].value_counts(),
                            "Ratio": 100 * dataframe[col_name].value_counts() / len(dataframe)}))
        print("##########################################")

        if plot:
            sns.countplot(x=dataframe[col_name], data=dataframe)
            plt.show(block=True)


# Numerical Variable Analysis

def num_summary(dataframe, numerical_col, plot=False):
    """
        Shows quantiles information about numeric variable data.
        :param plot: bool , for chart ratio
        :param dataframe: Data set(Pandas.DataFrame)
        :param numerical_col: Numerical column name -string-
        :return: no return
    """
    quantiles = [0.05, 0.10, 0.20, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 0.95, 0.99]
    print(dataframe[numerical_col].describe(quantiles).T)

    if plot:
        dataframe[numerical_col].hist()
        plt.xlabel(numerical_col)
        plt.title(numerical_col)
        plt.show(block=True)


# Capturing Categorical and Numeric Variables and Generalizing Operations


def grab_col_names(dataframe, cat_th=10, car_th=20):
    """
    It gives the names of categorical, numerical and categorical but cardinal variables in the data set.
    :param dataframe: is the dataframe whose variable names are to be retrieved.
    :param cat_th: (optional default=10)class threshold for numeric but categorical variables. int or float
    :param car_th: (optional default=20) class threshold for categorical but cardinal variables. int or float
    :return: cat_cols: list
         Categorical variable list
     num_cols: list
         Numeric variable list
     cat_but_car: list
         Categorical view cardinal variable list

    notes:
        cat_cols + num_cols + cat_but_car = total number of variables
        num_but_cat is inside cat_cols.

    """