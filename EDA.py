import numpy as np
import pandas as pd
import seaborn as sns
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

    # cat_cols, cat_but_car
    cat_cols = [col for col in dataframe.columns if dataframe[col].dtypes == "O"]
    num_but_cat = [col for col in dataframe.columns if
                   dataframe[col].nunique() < cat_th and dataframe[col].dtypes != "O"]
    cat_but_car = [col for col in dataframe.columns if
                   dataframe[col].nunique() > car_th and dataframe[col].dtypes == "O"]
    cat_cols = cat_cols + num_but_cat
    cat_cols = [col for col in cat_cols if col not in cat_but_car]

    # num_cols
    num_cols = [col for col in dataframe.columns if dataframe[col].dtypes != "O"]
    num_cols = [col for col in num_cols if col not in num_but_cat]

    print(f"Observations: {dataframe.shape[0]}")
    print(f"Variables: {dataframe.shape[1]}")
    print(f'cat_cols: {len(cat_cols)}')
    print(f'num_cols: {len(num_cols)}')
    print(f'cat_but_car: {len(cat_but_car)}')
    print(f'num_but_cat: {len(num_but_cat)}')

    return cat_cols, num_cols, cat_but_car


# Analysis of the Target Variable with Categorical Variables

def target_summary_with_cat(dataframe, target, categorical_col):
    print(pd.DataFrame({"TARGET_MEAN": dataframe.groupby(categorical_col)[target].mean()}), end="\n\n\n")


# Analysis of Target Variable with Numerical Variables

def target_summary_with_num(dataframe, target, numerical_col):
    print(dataframe.groupby(target).agg({numerical_col: "mean"}), end="\n\n\n")


# Analysis of Correlation

def high_corr_cols(dataframe, plot=False, corr_th=0.90):
    """

    :param dataframe: is the dataframe whose variable names are to be retrieved.
    :param plot: bool
    :param corr_th: upper limit of correlation, default 0.9, int or float
    :return: high correlation columns names list
    """

    corr = dataframe.corr()

    cor_matrix = corr.abs()
    upper_triangle_matrix = cor_matrix.where(np.triu(np.ones(cor_matrix.shape), k=1).astype(bool))
    drop_list = [col for col in upper_triangle_matrix.columns if any(upper_triangle_matrix[col] > corr_th)]

    if plot:
        import seaborn as sns
        import matplotlib.pyplot as plt
        sns.set(rc={'figure.figsize': (15, 15)})
        sns.heatmap(corr, cmap="RdBu")
        plt.show()

    return drop_list
