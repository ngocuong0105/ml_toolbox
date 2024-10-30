import os
import sys
import subprocess
import dill as pickle  # import pickle
from urllib.request import urlopen
import pandas as pd
import numpy as np
from numba import jit
import polars as pl
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy.stats.mstats import winsorize
from sklearn.decomposition import PCA
from sklearn.metrics import confusion_matrix
from sklearn.feature_selection import VarianceThreshold
from typing import Any, Dict, Iterable, List, Optional, Union
from datetime import datetime
from dateutil.relativedelta import relativedelta
from functools import wraps
from time import time

"""
dev helpers
"""


def timing(f):
    """
    Timing decorator for functions
    """

    @wraps(f)
    def wrap(*args, **kw):
        print(f'Starting {f.__name__}')
        ts = time()
        result = f(*args, **kw)
        te = time()
        print(f"func: {f.__name__} took: {round(te-ts,0)} sec")
        return result

    return wrap


def ram_usage(f):
    """
    RAM usage decorator for functions
    """

    @wraps(f)
    def wrap(*args, **kw):
        import psutil
        import os

        process = psutil.Process(os.getpid())
        mem_before = process.memory_info().rss
        result = f(*args, **kw)
        mem_after = process.memory_info().rss
        print(
            f"func: {f.__name__} used: {round((mem_after-mem_before)/1024/1024,2)} MB"
        )
        return result

    return wrap


def _filter(df: pd.DataFrame, filters: Dict):
    cols = list(filters.keys())
    vals = [makelist(filters[col]) for col in cols]
    for col, val in zip(cols, vals):
        df = df[df[col].isin(val)]
    return df


def reduce_memory(data):
    for col in list(data):
        col_type = data[col].dtype

        if col_type != object:
            c_min = data[col].min()
            c_max = data[col].max()
            if str(col_type)[:3] == "int":
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    if c_min >= 0:
                        data[col] = data[col].astype(np.uint8)
                    else:
                        data[col] = data[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    if c_min >= 0:
                        data[col] = data[col].astype(np.uint16)
                    else:
                        data[col] = data[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    if c_min >= 0:
                        data[col] = data[col].astype(np.uint32)
                    else:
                        data[col] = data[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    if c_min >= 0:
                        data[col] = data[col].astype(np.uint64)
                    else:
                        data[col] = data[col].astype(np.int64)
            else:
                if (
                    c_min > np.finfo(np.float32).min
                    and c_max < np.finfo(np.float32).max
                ):
                    data[col] = data[col].astype(np.float32)
                else:
                    data[col] = data[col].astype(np.float64)

def git_commit_and_run():
    # Define the commit message
    commit_message = "Automatic commit before running the script"

    # Change to the directory where your Git repository is located
    repo_dir = os.path.dirname(os.path.abspath(__file__))
    os.chdir(repo_dir)

    try:
        # Add changes to the staging area
        subprocess.run(["git", "add", "."], check=True)

        # Commit the changes
        subprocess.run(["git", "commit", "-m", commit_message], check=True)

        print("Changes committed successfully.")

    except subprocess.CalledProcessError as e:
        print(f"An error occurred while committing changes: {e}")
        sys.exit(1)

"""
CALCULATIONS
"""
def winsor_df(df: pd.DataFrame, left_bound: int = 0.05, right_bound: int = 0.05):
    """
    For all numerical columns applies winsor.
    Assumes df does not have categorical columns!

    NB: the interval data remaining is [left_bound,1-right_bound]. I did not change right_bound to 1-right_bound, because
    of round errors: 1-0.8 = 0.19999996
    NB: number of rows remains the same (values not in the boundary are capped).
    """
    return df.apply(lambda x: winsor(x, left_bound, right_bound))


def winsor(s: pd.Series, left_bound: int = 0.05, right_bound: int = 0.05):
    """
    Replaces bottom l and top r values.
    Example:
    input: a = np.array([10, 4, 9, 8, 5, 3, 7, 2, 1, 6])
    output: winsor(a, l=0.1, r=0.2) = [8, 4, 8, 8, 5, 3, 7, 2, 2, 6]
    top 20% elements are 9 and 10 (replaced by 8).
    bottom 10% elements is 1 (replaced by 1).
    """
    return winsorize(s, limits=[left_bound, right_bound])


def variance_threshold_selector(data, threshold=0.5):
    """
    Remove columns with less than threshold variance, only for numeric columns.
    """
    selector = VarianceThreshold(threshold)
    num_cols = data.select_dtypes(include=np.number).columns
    selector.fit(data[num_cols])
    return data[data.columns[selector.get_support(indices=True)]]


def map_mult_dfs(
    df1: pd.DataFrame,
    df2: pd.DataFrame,
    key_col: str,
    val_col: str,
    factor_col: str,
    default_factor: int = 1,
):
    """
    Multiplies two dataframes, where the second one is used as a map.
          key_col, val_col, 'other_col'
    df1 =   'a'      1          blah
            'b'      2          blah
            'b'      3          blah

            key_col(index)|   factor_col
    df2 =   'a'                 2
            'b'                 3

    return:
            key_col, multiplied_val_col
             'a'        1*2
             'b'        2*2
             'b'        3*3
    """
    mapping = dict(df2[factor_col])
    return (
        df1[[val_col, key_col]]
        .set_index(key_col)
        .apply(lambda x: x * mapping.get(x.name, default_factor), 1)
        .reset_index()
    )


def get_fct_intervals(fct_start, last_fct_start, fct_hor, fct_freq):
    """
    Returns a list of tuples with the start and end date of the forecast intervals.
    fct_start: start date of the forecast
    fct_end: end date of the forecast
    fct_hor: forecast horizon
    fct_freq: forecast frequency, can be 'd' (daily), 'm' (monthly) or 'y' (yearly)

    Example:
    fct_start = '2020-01-01'
    last_fct_start = '2020-01-05'
    fct_hor = 7
    fct_freq = 'd'
    Output:
    [('2020-01-01', '2020-01-08'),
    ('2020-01-02', '2020-01-09'),
    ('2020-01-03', '2020-01-10'),
    ('2020-01-04', '2020-01-11'),
    ('2020-01-05', '2020-01-12')]
    """

    fct_start = datetime.strptime(fct_start, "%Y-%m-%d")
    last_fct_start = datetime.strptime(last_fct_start, "%Y-%m-%d")
    if fct_freq == "d":
        hor = relativedelta(days=fct_hor)
        incr = relativedelta(days=1)
    elif fct_freq == "m":
        hor = relativedelta(months=fct_hor)
        incr = relativedelta(months=1)
    elif fct_freq == "y":
        hor = relativedelta(years=fct_hor)
        incr = relativedelta(years=1)
    fct_intervals = []
    while fct_start <= last_fct_start:
        end = fct_start + hor
        fct_intervals.append((fct_start.strftime("%Y-%m-%d"), end.strftime("%Y-%m-%d")))
        fct_start += incr
    return fct_intervals


def normalize(df: pd.DataFrame, cols: Union[str, List[str]] = [], mode="meanstd"):
    """
    Normalize numerical columns. Default behaviour is to normalize all numerical columns.
    """
    if isinstance(cols, str):
        cols = [cols]
    if cols == []:
        num_df = df.select_dtypes("number")
        rest_df = df.select_dtypes(exclude="number")
    else:
        num_df = df[cols]
        rest_df = df[df.columns.difference(cols)]
    if mode == "meanstd":
        normal_df = (num_df - num_df.mean()) / num_df.std()
    elif mode == "minmax":
        normal_df = (num_df - num_df.min()) / (num_df.max() - num_df.min())
    return pd.concat([rest_df, normal_df], axis=1)


def relative_change(df: pd.Series, val_col: str, lag: int):
    """
    Adds a column with the relative change of the value column.
    df[val_col] = [1,2,3,4,3,0,2]
    df[relative_change] = [1,0.5,0.25,-0.25,0,1,0]
    Edge cases:
    0 -> 2, then relative change is 1 (instead of inf)
    2 -> 0, then relative change is 0
    """
    df.loc[:, "relative_change"] = df[val_col].pct_change(lag).shift(-1).fillna(0)
    df.loc[(df["relative_change"] == float("inf")), "relative_change"] = 1
    return df


def save_experiment(
    objects: Dict[str, Any],
    path: str,
    exp_name: str = "",
    description: str = "",
    new_dir: bool = True,
):
    """
    Saves experiment to csv. Tuned for forecasting experiments.
    """
    if len(os.listdir(path)) == 0:
        experiment_num = 1
    else:
        experiment_num = max([int(x) for x in os.listdir(path) if x.isnumeric()]) + int(
            new_dir
        )
    if new_dir:
        Path(path + f"/{experiment_num}").mkdir(parents=True, exist_ok=True)
    for name, obj in objects.items():
        if isinstance(obj, pd.DataFrame):
            obj.to_csv(path + f"/{experiment_num}/{name}.csv", index=False)
        else:
            save_pickle(obj, path + f"/{experiment_num}/{name}.pickle")
    meta_data = {
        "Run": exp_name,
        "Description": description,
        "Date": datetime.now().strftime("%d/%m/%Y %H:%M:%S"),
    }
    with open(path + f"/{experiment_num}/meta_data_{experiment_num}.json", "w") as fp:
        json.dump(meta_data, fp, indent=4)


def save_pickle(obj, path):
    with open(path, "wb") as fp:
        pickle.dump(obj, fp)


def load_pickle(path):
    with open(path, "rb") as fp:
        return pickle.load(fp)


def compare_experiments(path: str, rng: Iterable = None):
    """
    Compares experiments in folder.
    path: path to folder containing experiments
    """
    if rng is None:
        rng = range(1, len(os.listdir(path)) + 1)
    results = []
    for i in rng:
        with open(path + f"/{i}/meta_data_{i}.json", "r") as fp:
            results.append(json.load(fp))
    res = pd.DataFrame(results)
    res.index += 1
    return res


def makelist(itr):
    if itr is None:
        return None
    if isinstance(itr, str):
        return [itr]
    try:
        return list(itr)
    except TypeError:
        return [itr]


"""
FEATURE ENGINEERING
"""


def smooth_data(df: pd.DataFrame, val_col: str, pct_change: float) -> pd.DataFrame:
    """
    ADDs a column which is smoothed ts data. Checks for any sharp changes in val_col.
    Example:
    pct_change = 0.2
    time     val_col    val_col_smooth
        1    0.674981	1.732559
        2    0.739265	1.897564
        3    0.713551	1.831562
        4    1.054256	1.831562
        5    1.054256	1.831562
        6    1.060871	1.843054

    At time 4, the absolute change is 1.054256 - 0.674981 = 0.379275, which is greater than
    0.2 * the past mean = 0.2 * (0.674981 + 0.739265 + 0.713551) / 3 = 0.375
    Thus we lift the values at times 1,2,3 by the ratio of the current value to the previous value
    which is 1.054256 / 0.713551 = 1.48.

    Figure:
                                         ----- (sharp increase)


                                    -----
                               -----
                        ------ (sharp increase)


           ---     ---
        --    ----
      -

    The points in the begginng should be multiplied by 2 ratios as they are affected by 2 sharp increases.
    That is why we use the cumulative product of the lift array.
    """
    df = df.copy(deep=True)
    # df[val_col] = np.where(df[val_col] == 0, 1e-16, df[val_col])
    cum_mean = df[val_col].cumsum() / np.arange(1, len(df) + 1)
    abs_change = (df[val_col].shift(1) - df[val_col]).fillna(0).abs()
    ratios = (df[val_col] / df[val_col].shift(1)).fillna(1)
    lift = np.where(abs_change >= pct_change * cum_mean, ratios, 1)
    df[f"{val_col}_smooth"] = (
        np.cumprod(np.append(lift[1:], 1)[::-1])[::-1] * df[val_col]
    )
    # df[f"{val_col}_smooth"] = df[f"{val_col}_smooth"].fillna(np.median(df[f"{val_col}_smooth"].dropna()))
    return df


@timing
def add_lag_features(
    feature_names: List[str],
    data: pd.DataFrame,
    target: str,
    groupby: Union[List[str], str],
    lags: Union[List[int], int],
) -> pd.DataFrame:
    """
    Creates lag features with lags from 1 to lags+1 or a list lags.

    data: pd.DataFrame is the input data
    target: str is the target column name
    groupby: Union[List[str], str] is the column name or list of column names to group by
    lags: Union[List[int],int] is the number of lags or a list of lags

    return: tuple of (data, list of feature names)

    Input:
    data: is
        a	b
        x	1
        x	2
        x	3
        x	11
        x	12
        x	13
        y	4
        y	5
        z	6

    target = b, groupby = a, lags = 2 gives:

    Output:
        a   b  lag_target_1  lag_target_2
        0  x   1           NaN           NaN
        1  x   2           1.0           NaN
        2  x   3           2.0           1.0
        3  x  11           3.0           2.0
        4  x  12          11.0           3.0
        5  x  13          12.0          11.0
        6  y   4           NaN           NaN
        7  y   5           4.0           NaN
        8  z   6           NaN           NaN,

    feats = ['lag_target_1', 'lag_target_2']
    """
    if isinstance(groupby, str):
        groupby = [groupby]
    groupby_name = "_".join(groupby)

    if isinstance(lags, int):
        lags = [lags]

    for lag in lags:
        data[f"add_lag_features_{target}_{groupby_name}_{lag}"] = data.groupby(groupby)[
            target
        ].shift(lag)
        feature_names.append(f"add_lag_features_{target}_{groupby_name}_{lag}")
    return data


@timing
def add_differenced_features(
    feature_names: List,
    data: pd.DataFrame,
    target: str,
    groupby: Union[List[str], str],
    lags: Union[List[int], int],
) -> pd.DataFrame:
    """
    Creates lag features with lags from 1 to lags+1 or a list lags.
    data: pd.DataFrame is the input data
    target: str is the target column name
    groupby: Union[List[str], str] is the column name or list of column names to group by
    lags: Union[List[int],int] is the number of lags or a list of lags

    Input:
    data: is
        a	b
        x	1
        x	2
        x	3
        x	11
        x	12
        x	13
        y	4
        y	5
        z	6

    target = b, groupby = a, lags = 2 gives:

    Output:
    a   b  diff_target_1  diff_target_2
    x   1            NaN            NaN
    x   2            1.0            NaN
    x   3            1.0            2.0
    x  11            8.0            9.0
    x  12            1.0            9.0
    x  13            1.0            2.0
    y   4            NaN            NaN
    y   5            1.0            NaN
    z   6            NaN            NaN

    feats: List[str] is ['diff_target_1', 'diff_target_2']
    """
    if isinstance(groupby, str):
        groupby = [groupby]
    if isinstance(lags, int):
        lags = [lags]

    for lag in lags:
        data[f"add_differenced_features_{target}_{lag}"] = data.groupby(groupby)[
            target
        ].diff(lag)
        feature_names.append(f"add_differenced_features_{target}_{lag}")
    return data


def add_pct_change_features(
    feature_names: List,
    data: pd.DataFrame,
    target: str,
    groupby: Union[List[str], str],
    lags: Union[List[int], int],
) -> tuple:
    if isinstance(groupby, str):
        groupby = [groupby]
    if isinstance(lags, int):
        lags = [lags]

    for lag in lags:
        data[f"add_pct_change_features_{target}_{lag}"] = data.groupby(groupby)[
            target
        ].pct_change(lag)
        feature_names.append(f"add_pct_change_features_{target}_{lag}")
    return data


def add_rolling_features(
    feature_names: List,
    data: pd.DataFrame,
    target: str,
    func: str,
    groupby: Union[List[str], str],
    window_sizes: Union[List[int], int],
    shift_period: int = 0,
) -> pd.DataFrame:
    """
    Creates rolling mean features with window sizes from 2 to window_sizes+1 or a list window_sizes.
    window_size = 1 is just the prev element.
    data: pd.DataFrame is the input data
    target: str is the target column name
    groupby: Union[List[str], str] is the column name or list of column names to group by
    window_sizes: Union[List[int],int] is the number of window sizes or a list of window sizes
    shift_period: int is the number of periods to shift the target column by (use it if your feature dpeends on the column you want to predict)

    Input:
    data: is
    a	b
    x	1
    x	2
    x	3
    y	4
    y	5
    z	6

    window = [2], target = b, groupby = a gives

    Output:
        a	b	prev	rolling_mean_target_2
        x	1	NaN	    NaN
        x	2	1.0	    1.0
        x	3	2.0	    1.5
        y	4	NaN	    NaN
        y	5	4.0	    4.0
        z	6	NaN 	NaN

    feats: List[str] is ['rolling_mean_target_2']

    NB window = 1 is the same as lag features with lag = 1.
    """

    if isinstance(window_sizes, int):
        window_sizes = [window_sizes]

    for window in window_sizes:
        data["prev"] = data.groupby(groupby)[target].shift(shift_period)
        data[f"add_rolling_features_{target}_{func}_{window}"] = data.groupby(groupby)[
            "prev"
        ].transform(lambda s: s.rolling(window, min_periods=1).agg(func))
        feature_names.append(f"add_rolling_features_{target}_{func}_{window}")
    data = data.drop(columns="prev")
    return data


@timing
def add_rolling_features_polars(
    feature_names: List,
    data: pl.DataFrame,
    target: str,
    func: str,
    groupby: Union[List[str], str],
    window_sizes: Union[List[int], int],
    shift_period: int = 0,
) -> pl.DataFrame:
    """
    data_pl = pl.DataFrame(data)
    for col in ["add_fct"]:
        for agg in ['mean']:
            data_pl = add_rolling_features_polars(feature_names, data_pl, col, agg, groupby=[STOCK_ID, DATE_ID],window_sizes=[3,6,10])
    data = data_pl.to_pandas().set_index(data.index)
    """
    if isinstance(window_sizes, int):
        window_sizes = [window_sizes]
    if isinstance(groupby, str):
        groupby = [groupby]
    for window in window_sizes:
        data = data.with_columns(
            pl.col(target).shift(shift_period).over(groupby).alias("prev")
        )
        if func == "mean":
            data = data.with_columns(
                pl.col("prev")
                .rolling_mean(window_size=window)
                .over(groupby)
                .alias(f"add_rolling_features_{target}_{func}_{window}"),
            )
        elif func == "std":
            data = data.with_columns(
                pl.col("prev")
                .rolling_std(window_size=window)
                .over(groupby)
                .alias(f"add_rolling_features_{target}_{func}_{window}"),
            )
        elif func == "min":
            data = data.with_columns(
                pl.col("prev")
                .rolling_min(window_size=window)
                .over(groupby)
                .alias(f"add_rolling_features_{target}_{func}_{window}"),
            )
        elif func == "max":
            data = data.with_columns(
                pl.col("prev")
                .rolling_max(window_size=window)
                .over(groupby)
                .alias(f"add_rolling_features_{target}_{func}_{window}"),
            )
        elif func == "sum":
            data = data.with_columns(
                pl.col("prev")
                .rolling_mean(window_size=window)
                .over(groupby)
                .alias(f"add_rolling_features_{target}_{func}_{window}"),
            )
            data = data.with_columns(
                pl.col(f"add_rolling_features_{target}_{func}_{window}") * window
            )
        elif func == "median":
            data = data.with_columns(
                pl.col("prev")
                .rolling_median(window_size=window)
                .over(groupby)
                .alias(f"add_rolling_features_{target}_{func}_{window}"),
            )
        elif func == "skew":
            data = data.with_columns(
                pl.col("prev")
                .rolling_skew(window_size=window)
                .over(groupby)
                .alias(f"add_rolling_features_{target}_{func}_{window}"),
            )

        feature_names.append(f"add_rolling_features_{target}_{func}_{window}")
    data = data.drop(columns="prev")
    return data


def get_confusion_matrix(data, real, fct):
    """Get the confusion matrix"""
    tn, fp, fn, tp = confusion_matrix(data[real], data[fct]).ravel()
    return tn, fp, fn, tp


def apply_pca(
    df: pd.DataFrame, feature_cols: List[str], other_cols: List[str], n_components: int
) -> pd.DataFrame:
    """
    PCA transformation

    Assumes no Nans in df[feature_cols]
    """
    df_standardized = (df[feature_cols] - df[feature_cols].mean()) / df[
        feature_cols
    ].std()

    # Perform PCA
    pca = PCA()
    principal_components = pca.fit_transform(df_standardized)

    # Create new DataFrame with principal components
    columns = ["PC" + str(i + 1) for i in range(len(feature_cols))]
    df_pca = pd.DataFrame(data=principal_components, columns=columns)
    df_pca = pd.concat([df[other_cols], df_pca], axis=1)
    return df_pca.iloc[:, : len(other_cols) + n_components]


"""
PLOTS
"""


def sns_hist(
    df: pd.DataFrame,
    col: str,
    hue_col: str = None,
    binwidth: int = None,
    stat="probability",
    element="bars",
):
    """
    Seaborn histogram
    https://seaborn.pydata.org/generated/seaborn.histplot.html
    stat: Aggregate statistic to compute in each bin.
        count: show the number of observations in each bin
        frequency: show the number of observations divided by the bin width
        probability or proportion: normalize such that bar heights sum to 1
        percent: normalize such that bar heights sum to 100
        density: normalize such that the total area of the histogram equals 1

    element{“bars”, “step”, “poly”}:
        Visual representation of the histogram statistic. Only relevant with univariate data.
    """
    sns.histplot(df, x=col, hue=hue_col, binwidth=binwidth, stat=stat, element=element)


def hist(
    df: pd.DataFrame,
    col: str,
    name: str = "",
    histnorm: str = "probability",
    renderer="browser",
    yaxis=None,
    return_fig=False,
):
    """
    Histogram ~ probability density function
    col: column to be used as x axis - nb: you specify the x axis!
    Example: hist.png (microbisness_density is the target variable)
    """

    layout = dict(
        xaxis=dict(range=[df[col].quantile(0.01), df[col].quantile(0.99)]),
        yaxis=dict(range=yaxis),
    )
    fig = go.Figure(data=[go.Histogram(x=df[col], histnorm=histnorm)], layout=layout)
    fig.update_layout(
        title_text=f"{name}",  # title of plot
        xaxis_title_text=f"{col}",  # xaxis label
        yaxis_title_text="histnorm",  # yaxis label
        # bargap=0.2,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1,  # gap between bars of the same location coordinates
    )
    fig.update_traces(opacity=0.75)
    if return_fig:
        return fig
    fig.show(renderer=renderer)


def hist_xcol(
    df: pd.DataFrame,
    xcol: str,
    ycols: Union[List[str], str],
    histfunc="sum",
    renderer="browser",
    title="",
    orientation="v",
    order="ascending",
    bargap=0.2,
    bargroupgap=0.1,
    return_fig=False,
):
    """
    Histogram with Aggregation Function histfunc. Default is sum.
    xcol: column to be used as x axis - nb: you specify the x axis!
    ycols: column(s) to be used as y axis
    Check images folder for example.

    orientation: 'v','h'
    order: 'ascending', 'descending'

    If you choose horizontal orientation, you must change xcols and ycols
    """
    if isinstance(ycols, str):
        ycols = [ycols]
    fig = go.Figure()
    for ycol in ycols:
        fig.add_trace(
            go.Histogram(
                x=df[xcol],
                y=df[ycol],
                histfunc=histfunc,
                name=f"{ycol}",  # name used in legend and hover labels
                opacity=0.75,
                orientation=orientation,
            )
        )
    if title == "":
        title = ", ".join(ycols)
    fig.update_layout(
        title_text=title,  # title of plot
        xaxis_title_text=f"{xcol}",  # xaxis label
        yaxis_title_text="Value",  # yaxis label
        bargap=bargap,  # gap between bars of adjacent location coordinates
        bargroupgap=bargroupgap,  # gap between bars of the same location coordinates
    )
    fig.update_xaxes(categoryorder=f"total {order}")
    fig.update_yaxes(categoryorder=f"total {order}")
    if return_fig:
        return fig
    fig.show(renderer=renderer)


def count_plot(
    df: pd.DataFrame,
    col: str,
    title: str = "",
    figsize=(16, 5),
    pallette=None,
    order=None,
):
    """
    Count plot for values in a column col
    """
    plt.figure(figsize=figsize)
    sns.countplot(x=col, data=df, palette=pallette, order=order)
    plt.title(title)
    plt.show()


def count_nan_plot(df: pd.DataFrame, figsize=(18, 9)):
    df_missing = df.isna().sum() / len(df) * 100
    df_missing_index = df_missing.index
    df_missing_values = df_missing.values

    fig, ax = plt.subplots(figsize=figsize)
    barchart = sns.barplot(x=df_missing_index, y=df_missing_values, ax=ax)
    barchart.axes.set_title(
        "Share of missing values in train data", fontsize=24, loc="center"
    )
    barchart.bar_label(barchart.containers[0], fmt="%.1f%%", fontsize=10)
    barchart.yaxis.set_tick_params(labelsize=15)
    barchart.xaxis.set_tick_params(rotation=40, labelsize=10)
    plt.show()


def dist_plot(
    df: pd.DataFrame,
    ycols: Union[List, str],
    bin_size=0.2,
    width=300,
    height=200,
    title="",
    renderer="browser",
):
    """
    Plots pdf curve using KDE estimate.
    """
    if isinstance(ycols, str):
        ycols = [ycols]

    hist_data = [df[col] for col in ycols]
    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot(hist_data, ycols, bin_size=0.2, show_rug=False)

    # Add title
    if title == "":
        title = ", ".join(ycols)
    fig.update_layout(title_text=title, width=width, height=height)
    fig.show(renderer=renderer)


def cdf_share_plot(df: pd.DataFrame, ycol: str, renderer="browser", return_fig=False):
    """
    Plots cdf share curve. Sorts the values in descending order and
    plots the cumulative sum over the total sum.

    One usecase is to explore forecasting errors (ycol = error)
    What percentage of the highest errors account for most of the total error.

    See cdf_share_plot.ahtml for example. (rename to html file)
    """
    df = df.sort_values(ycol, ascending=False)
    data = []
    for value in np.linspace(0, 1, 500):
        data.append(
            {
                f"Share of highest {ycol}": value,
                "Share of total": df[df[ycol] > df[ycol].quantile(1 - value)][
                    ycol
                ].sum()
                / df[ycol].sum(),
            }
        )
    df_plot = pd.DataFrame(data)
    fig = px.line(
        df_plot,
        x=f"Share of highest {ycol}",
        y="Share of total",
        title=f"Skew of {ycol} values towards the highest",
    )

    fig.add_trace(
        go.Scatter(
            x=np.linspace(0, 1, 500), y=np.linspace(0, 1, 500), mode="lines", name="x=y"
        )
    )
    fig.update_xaxes(range=[0, 1])
    fig.update_yaxes(
        scaleanchor="x",
        scaleratio=1,
    )

    fig.update_layout(autosize=False, width=600, height=600, showlegend=True)
    if return_fig:
        return fig
    fig.show(renderer=renderer)


def bar_plot(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    hue: str = None,
    estimator: str = "mean",
    return_fig: bool = False,
    swap_axis: bool = False,
    figsize: tuple = (16, 10),
    ascending: bool = True,
    title="",
):
    """
    Bar blot with error bars/confidence intervals.
    If you want to have horizontal bars specify swap_axis = True.
    Requires seaborn >= 0.12.2. Note that in wandb seaborn figures does not render x-axis and y-axis

    sns.barplot(train,x=STOCK_ID,y=TARGET,errorbar=None,estimator='std')# with no error bar
    """
    plt.figure(figsize=figsize)
    order_x, order_y = xcol, ycol
    if swap_axis:
        xcol, ycol = ycol, xcol
    # order by mean of ycol
    if ascending is not None:
        fig = sns.barplot(
            data=df,
            x=xcol,
            y=ycol,
            errorbar="sd",
            estimator=estimator,
            order=df.groupby(order_x)
            .mean()
            .sort_values(order_y, ascending=ascending)
            .index.unique(),
            hue=hue,
        )
    else:
        fig = sns.barplot(
            data=df,
            x=xcol,
            y=ycol,
            errorbar="sd",
            estimator=estimator,
            order=df.groupby(order_x).mean().index.unique(),
            hue=hue,
        )
    fig.set_title(title)
    if return_fig:
        return fig.get_figure()


def line_plot(
    df: pd.DataFrame,
    xcol: str,
    ycols: Union[List[str], str],
    yaxis_range=None,
    renderer="browser",
    title="",
    **filters,
):
    """
    Plots multiple y lines over xcol
    Example: smooth.ahtml (rename to html file)
    """
    df = _filter(df, filters)
    df = df.sort_values(xcol)
    if isinstance(ycols, str):
        ycols = [ycols]
    fig = px.line(df, x=xcol, y=ycols, markers=True)
    fig.update_layout(title=title + f"{filters}", yaxis_range=yaxis_range)
    fig.show(renderer=renderer)


def sns_line_plot(
    df: pd.DataFrame,
    xcol: str,
    ycol: Union[str, List[str]],
    hue_col: str = None,
    title: str = "",
    new_plot: bool = True,
    **filters,
):
    """
    Seaborn line plot
    Allows to plot multiple y lines over xcol, where multiple columns are defined by hue!
    new_plot: is used when using that function in a for loop
    """
    df = _filter(df, filters)
    if new_plot:
        plt.figure()
    if isinstance(ycol, str):
        ax = sns.lineplot(data=df, x=xcol, y=ycol, hue=hue_col, legend=True)
    else:
        ax = sns.lineplot(
            data=pd.melt(df[[xcol] + ycol], id_vars=xcol),
            x=xcol,
            y="value",
            hue="variable",
        )
    ax.set_title(title)
    ax.xaxis.set_tick_params(rotation=45)


def scatter_plot(
    df: pd.DataFrame,
    xcol: str,
    ycols: Union[List[str], str],
    renderer="browser",
    size=12,
    trendline=None,
    **filters,
):
    """
    Scatter plot with trendline
    trendline: str can take "ols", "lowess", "glm", "rlm"
    """
    df = _filter(df, filters)
    df = df.sort_values(xcol)
    if isinstance(ycols, str):
        ycols = [ycols]
    fig = px.scatter(df, x=xcol, y=ycols, trendline=trendline)
    fig.update_traces(marker=dict(size=size))
    fig.show(renderer=renderer)


def sns_scatter_plot(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    hue_col: str = None,
    **filters,
):
    """
    Plots multiple y lines over xcol, where multiple columns are defined by hue!
    https://seaborn.pydata.org/generated/seaborn.scatterplot.html
    """
    df = _filter(df, filters)
    plt.figure()
    sns.scatterplot(data=df, x=xcol, y=ycol, hue=hue_col)


def point_plot(
    df: pd.DataFrame,
    xcol: str,
    ycols: Union[str, List[str]],
    figsize: tuple = (16, 10),
    legend=True,
):
    """
    Show point estimates and errors using dot marks.
    A point plot represents an estimate of central tendency for a numeric variable by the position of the dot
    and provides some indication of the uncertainty around that estimate using error bars.

    https://seaborn.pydata.org/generated/seaborn.pointplot.html
    """
    if isinstance(ycols, str):
        ycols = [ycols]
    plt.figure(figsize=figsize)
    for ycol, color in zip(ycols, sns.color_palette(n_colors=len(ycols))):
        sns.pointplot(x=xcol, y=ycol, data=df, color=color, label=ycol)
    if legend:
        plt.legend(loc="upper left")
    plt.show()


def reg_plot(df: pd.DataFrame, xcol: str, ycol: str, figsize: tuple = (16, 6)):
    """
    Regression plot with confidence intervals.
    """

    plt.figure(figsize=figsize)
    sns.regplot(x=df[xcol], y=df[ycol], ci=95, color="r")
    plt.show()


def reg_plots(
    df: pd.DataFrame, xcol: str, ycol: str, hue_col: str, figsize: tuple = (16, 6)
):
    """
    Regression plot with confidence intervals. See reg_plots.png.
    xcol = bmi
    ycol = charges
    hue_col = smoker
    """

    plt.figure(figsize=figsize)
    sns.lmplot(x=xcol, y=ycol, data=df, hue=hue_col, ci=95)
    plt.show()


def categorical_scatter_plot(
    df: pd.DataFrame, xcol: str, ycol: str, figsize: tuple = (16, 6)
):
    """
    NB: This could be SLOW for large datasets.

    For categorical plots look at https://seaborn.pydata.org/tutorial/categorical.html
    Scatter plot with categorical x axis. See categorical_scatter_plot.png.
    xcol = smoker
    ycol = charges

    """
    plt.figure(figsize=figsize)
    sns.swarmplot(x=df[xcol], y=df[ycol])
    plt.show()


def box_plot(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    points: str = None,
    title="",
    renderer="browser",
):
    """
    Box plot with scattered points.
    https://plotly.com/python/box-plots/ -> search for "Display the underlying data"

    points: str can take "all", None
    assumes xcol is categorical
    """

    fig = px.box(df, x=xcol, y=ycol, points=points)
    fig.update_layout(title=title)
    fig.show(renderer=renderer)


def corr_heatmap(corr: pd.DataFrame, figsize: tuple = (16, 6)):
    plt.figure(figsize=figsize)
    # define the mask to set the values in the upper triangle to True
    mask = np.triu(np.ones_like(corr, dtype=np.bool))
    heatmap = sns.heatmap(
        corr,
        mask=mask,
        vmin=-1,
        vmax=1,
        annot=True,
        annot_kws={"fontsize": 18},
        cmap="flare",
    )
    heatmap.set_title("Triangle Correlation Heatmap", fontdict={"fontsize": 20}, pad=16)


def vertical_lines(vertical_lines_x: List[Any]):
    """
    Usage: Run it with another plotting function that is using matplotlib or seaborn plot.

    sns_line_plot(train, SEC_IN_BUCKET, TARGET, hue_col=DATE_ID, stock_id=stock_id)
    vertical_lines([60,300,480])

    """
    for line_x in vertical_lines_x:
        plt.axvline(
            x=line_x, color="red", linestyle="--"
        )  # Customize the color and linestyle as needed


def fct_plot(
    df: pd.DataFrame,
    xcol: str,
    ycols: Union[List[str], str],
    forecast_start: Union[List[Union[str, int]], str] = None,
    title: str = "",
    renderer="browser",
    marker_size=8,
    return_fig=False,
    layout="default",
    **filters,
):
    """
    Most enhanced forecasting plot using plotly.

    forecast_start: can be string date, int index or list of string dates or int indexes.
    just make sure you choose the correct layout

    layout: default if you have dates.
    """
    df = _filter(df, filters)
    if len(filters):
        title = ", ".join(
            "{} = {}".format(key, value) for key, value in filters.items()
        )
    if layout == "default":
        layout = DEFAULT_LAYOUT
    fig = go.Figure(layout=layout, layout_title_text=title)
    ycols = makelist(ycols)
    for ycol in ycols:
        if marker_size:
            fig.add_scatter(
                x=df[xcol],
                y=df[ycol],
                name=ycol,
                mode="lines+markers",
                marker=dict(size=marker_size),
            )
        else:
            fig.add_scatter(x=df[xcol], y=df[ycol], name=ycol, mode="lines")

    if forecast_start is not None:
        if layout == "default":
            forecast_start = [pd.to_datetime(forecast_start)]
        if not isinstance(forecast_start, List):
            forecast_start = [forecast_start]
        for fct in forecast_start:
            fig.add_shape(
                go.layout.Shape(
                    type="line",
                    yref="paper",
                    x0=fct,
                    y0=0,
                    x1=fct,
                    y1=1,
                    line=dict(color="Red", width=1),
                )
            )
    fig.update_layout(title=title)
    if return_fig:
        return fig
    fig.show(renderer=renderer)


def fct_error_plot(
    df: pd.DataFrame,
    act: str,
    fct: str,
    renderer="browser",
    return_fig=False,
    **filters,
):
    """
    Histogram plot with the distribution of errors between actual and forecasted values.
    """
    df = _filter(df, filters)
    name = ", ".join("{} = {}".format(key, value) for key, value in filters.items())
    df["error"] = df[act] - df[fct]
    return hist(
        df,
        col="error",
        name="Error distribution " + name,
        renderer=renderer,
        return_fig=return_fig,
    )


def usa_plot(
    df,
    xcol,
    ycol,
    range_color=(0, 10),
    title="",
    locationmode="USA-states",
    renderer="browser",
    return_fig=False,
    ylabel="",
):
    df = df.groupby(xcol)[ycol].mean().reset_index()
    if ylabel == "":
        ylabel = ycol
    if locationmode == "fips":
        with urlopen(
            "https://raw.githubusercontent.com/plotly/datasets/master/geojson-counties-fips.json"
        ) as response:
            counties = json.load(response)
        fig = px.choropleth(
            df,
            geojson=counties,
            locations=xcol,
            color=ycol,
            color_continuous_scale="YlOrRd",
            range_color=range_color,
            scope="usa",
            labels={ycol: ylabel},
        )
    elif locationmode == "USA-states":
        fig = px.choropleth(
            df,
            locations=xcol,
            locationmode="USA-states",
            color=ycol,
            color_continuous_scale="YlOrRd",
            range_color=range_color,
            scope="usa",
            labels={ycol: ylabel},
        )
    if title == "":
        title = ycol
    fig.update_layout(title_text=title)
    if return_fig:
        return fig
    fig.show(renderer=renderer)


def plot_confusion_matrix(
    data: pd.DataFrame,
    real: str,
    fct: str,
    figsize: tuple = (8, 8),
    return_fig: bool = False,
):
    """Plot the confusion matrix"""
    tn, fp, fn, tp = get_confusion_matrix(data, real, fct)
    confusion = np.array([[tn, fp], [fn, tp]])
    plt.figure(figsize=figsize)
    fig = sns.heatmap(confusion, annot=True, fmt="d", cmap="YlGnBu")
    plt.title("Confusion matrix")
    plt.ylabel("Actual label")
    plt.xlabel("Predicted label")
    if return_fig:
        return fig.get_figure()
    plt.show()


def pie_chart(
    df: pd.DataFrame,
    xcol: str,
    ycol: str,
    renderer: str = "browser",
    return_fig: bool = False,
):
    fig = px.pie(
        df, values=ycol, names=xcol, title=f"{ycol} per {xcol}", hover_data=[ycol]
    )
    fig.update_traces(textposition="inside", textinfo="percent+label")
    if return_fig:
        return fig
    fig.show(renderer=renderer)


DEFAULT_LAYOUT = dict(
    xaxis=dict(
        type="date",
        rangeselector=dict(
            buttons=list(
                [
                    dict(count=7, label="1w", step="day", stepmode="backward"),
                    dict(count=1, label="1m", step="month", stepmode="backward"),
                    dict(count=3, label="3m", step="month", stepmode="backward"),
                    dict(count=6, label="6m", step="month", stepmode="backward"),
                    dict(count=1, label="1y", step="year", stepmode="backward"),
                    dict(step="all"),
                ]
            )
        ),
        rangeslider=dict(visible=True),
    ),
)
