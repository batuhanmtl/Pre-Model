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


# Accessing outliers

def grab_outliers(dataframe, col_name, index=False):
    """
    Displays observations containing outliers.
    If Index = True, returns indexes containing outliers

    :param dataframe: Pandas.DataFrame
    :param col_name: Variable name whose outliers will be check
    :param index: bool
    :return: list of outlier index
    """

    low, up = outlier_thresholds(dataframe, col_name)

    if dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].shape[0] > 10:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].head())

    else:
        print(dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))])

    if index:
        outlier_index = dataframe[((dataframe[col_name] < low) | (dataframe[col_name] > up))].index
        return outlier_index


# Delete outlier value


def remove_outlier(dataframe, col_name):
    """
    Delete outlier value
    :param dataframe: Pandas.DataFrame
    :param col_name: string
    variable name to delete outlier
    :return: outlier deleted dataframe
    """

    low_limit, up_limit = outlier_thresholds(dataframe, col_name)

    df_without_outliers = dataframe[~((dataframe[col_name] < low_limit) | (dataframe[col_name] > up_limit))]

    return df_without_outliers


# re-assignment with thresholds

def replace_with_thresholds(dataframe, variable):
    """
    Replace outliers with threshold values.
    :param dataframe: Pandas.DataFrame
    :param variable: string
    Variable name whose outlier will be suppressed
    :return: no return
    """
    low_limit, up_limit = outlier_thresholds(dataframe, variable)

    dataframe.loc[(dataframe[variable] < low_limit), variable] = low_limit

    dataframe.loc[(dataframe[variable] > up_limit), variable] = up_limit


# Capturing missing values


def missing_values_table(dataframe, na_name=False):
    """
    Displays the ratio of missing values in the variable.
    na_col If True, it returns a list of the names of the
    variables with missing values.
    :param dataframe:Pandas.DataFrame
    :param na_name: bool
    :return: list of the names of the
    variables with missing values
    """

    na_columns = [col for col in dataframe.columns if dataframe[col].isnull().sum() > 0]

    n_miss = dataframe[na_columns].isnull().sum().sort_values(ascending=False)

    ratio = (dataframe[na_columns].isnull().sum() / dataframe.shape[0] * 100).sort_values(ascending=False)

    missing_df = pd.concat([n_miss, np.round(ratio, 2)], axis=1, keys=['n_miss', 'ratio'])

    print(missing_df, end="\n")

    if na_name:
        return na_columns


# Examining the Relationship of Missing Values with the Dependent Variable


def missing_vs_target(dataframe, target, na_columns):
    """
    It shows the proportional relationship of the
    missing values with respect to the target variable.

    :param dataframe: Pandas.DataFrame

    :param target: string
    Target name

    :param na_columns: list

    :return: no return
    """

    temp_df = dataframe.copy()

    for col in na_columns:
        temp_df[col + '_NA_FLAG'] = np.where(temp_df[col].isnull(), 1, 0)

    na_flags = temp_df.loc[:, temp_df.columns.str.contains("_NA_")].columns

    for col in na_flags:
        print(pd.DataFrame({"TARGET_MEAN": temp_df.groupby(col)[target].mean(),
                            "Count": temp_df.groupby(col)[target].count()}), end="\n\n\n")


# Label Encoding

def label_encoder(dataframe, binary_col):
    """
    Encodes variables consisting of 2 classes

    :param dataframe: Pandas.DataFrame
    :param binary_col: binary variable name list
    :return: Encoded dataframe
    """

    from sklearn.preprocessing import LabelEncoder

    labelencoder = LabelEncoder()
    dataframe[binary_col] = labelencoder.fit_transform(dataframe[binary_col])

    return dataframe


# One-Hot Encoding

def one_hot_encoder(dataframe, categorical_cols, drop_first=True):
    """
    One-Hot Encoding
    :param dataframe: Pandas.DataFrame
    :param categorical_cols: Name list of categorical variables
    :param drop_first: bool
    :return: One-Hot encoded dataframe
    """

    dataframe = pd.get_dummies(dataframe, columns=categorical_cols,
                               drop_first=drop_first)

    return dataframe


# Rare Analyser

def rare_analyser(dataframe, target, cat_cols):
    """
    Shows analysis of rare data
    :param dataframe: Pandas.DataFrame
    :param target: target variable
    :param cat_cols: string
    categorical variables name
    :return: no return
    """

    for col in cat_cols:
        print(col, ":", len(dataframe[col].value_counts()))
        print(pd.DataFrame({"COUNT": dataframe[col].value_counts(),
                            "RATIO": dataframe[col].value_counts() / len(dataframe),
                            "TARGET_MEAN": dataframe.groupby(col)[target].mean()}), end="\n\n\n")


# Rare Encoder

def rare_encoder(dataframe, rare_perc):
    """
    :param dataframe: Pandas.DataFrame
    :param rare_perc: int or float
    Rarity percentage
    :return: Pandas.DataFrame
    Concatenated dataframe of rare values
    """

    temp_df = dataframe.copy()

    rare_columns = [col for col in temp_df.columns if temp_df[col].dtypes == 'O'
                    and (temp_df[col].value_counts() / len(temp_df) < rare_perc).any(axis=None)]

    for var in rare_columns:
        tmp = temp_df[var].value_counts() / len(temp_df)
        rare_labels = tmp[tmp < rare_perc].index
        temp_df[var] = np.where(temp_df[var].isin(rare_labels), 'Rare', temp_df[var])

    return temp_df
