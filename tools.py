import os
import pickle
from urllib.request import urlopen
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import plotly.figure_factory as ff
import seaborn as sns
import matplotlib.pyplot as plt
import json
from pathlib import Path
from scipy.stats.mstats import winsorize
from typing import Any, Dict, Iterable, List, Optional, Union
from datetime import datetime

"""
CALCULATIONS
"""


def winsor_df(df: pd.DataFrame, left_bound: int = 0.05, right_bound: int = 0.05):
    """
    For all numerical columns applies winsor.
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
    objects: Dict[str, Any], path: str, exp_name: str = "", description: str = "",
):
    """
    Saves experiment to csv. Tuned for forecasting experiments.
    """
    if len(os.listdir(path)) == 0:
        experiment_num = 1
    else:
        experiment_num = max([int(x) for x in os.listdir(path) if x.isnumeric()]) + 1

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


def lag_features(
    data: pd.DataFrame,
    target: str,
    groupby: Union[List[str], str],
    lags: Union[List[int], int],
) -> tuple:
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
    if isinstance(lags, int):
        lags = [i for i in range(1, lags + 1)]

    feats = []
    for lag in lags:
        data[f"lag_{target}_{lag}"] = data.groupby(groupby)[target].shift(lag)
        feats.append(f"lag_{target}_{lag}")
    return data, feats


def diff_features(
    data: pd.DataFrame,
    target: str,
    groupby: Union[List[str], str],
    lags: Union[List[int], int],
) -> tuple:
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
        lags = [i for i in range(1, lags + 1)]

    feats = []
    for lag in lags:
        data[f"diff_{target}_{lag}"] = data.groupby(groupby)[target].diff(lag)
        feats.append(f"diff_{target}_{lag}")
    return data, feats


def rolling_features(
    data: pd.DataFrame,
    target: str,
    func: str,
    groupby: Union[List[str], str],
    window_sizes: Union[List[int], int],
) -> tuple:
    """
    Creates rolling mean features with window sizes from 1 to window_sizes+1 or a list window_sizes.
    data: pd.DataFrame is the input data
    target: str is the target column name
    groupby: Union[List[str], str] is the column name or list of column names to group by
    window_sizes: Union[List[int],int] is the number of window sizes or a list of window sizes

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
        window_sizes = [i for i in range(1, window_sizes + 1)]
    feats = []
    for window in window_sizes:
        data["prev"] = data.groupby(groupby)[target].shift(1)
        data[f"{target}_roll_{func}_{window}"] = data.groupby(groupby)[
            "prev"
        ].transform(lambda s: s.rolling(window, min_periods=1).agg(func))
        feats.append(f"{target}_roll_{func}_{window}")
    data = data.drop(columns="prev")
    return data, feats


"""
PLOTS
"""


def hist(
    df: pd.DataFrame, col: str, name: str = "", renderer="browser", return_fig=False,
):
    """
    Histogram ~ probability density function
    col: column to be used as x axis - nb: you specify the x axis!
    Example: hist.png (microbisness_density is the target variable)
    """

    layout = dict(
        xaxis=dict(range=[df[col].quantile(0.01), df[col].quantile(0.99)]),
        yaxis=dict(range=[0, 1]),
    )
    fig = go.Figure(
        data=[go.Histogram(x=df[col], histnorm="probability")], layout=layout
    )
    fig.update_layout(
        title_text=f"{name}",  # title of plot
        xaxis_title_text=f"{col}",  # xaxis label
        yaxis_title_text="Fraction",  # yaxis label
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
    return_fig=False,
):
    """
    Histogram with Aggregation Function histfunc. Default is sum.
    xcol: column to be used as x axis - nb: you specify the x axis!
    ycols: column(s) to be used as y axis
    Check images folder for example.
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
            )
        )
    if title == "":
        title = ", ".join(ycols)
    fig.update_layout(
        title_text=title,  # title of plot
        xaxis_title_text=f"{xcol}",  # xaxis label
        yaxis_title_text="Value",  # yaxis label
        bargap=0.2,  # gap between bars of adjacent location coordinates
        bargroupgap=0.1,  # gap between bars of the same location coordinates
    )
    fig.update_xaxes(categoryorder="total ascending")
    if return_fig:
        return fig
    fig.show(renderer=renderer)


def dist_plot(df: pd.DataFrame, ycols: Union[List, str], title="", renderer="browser"):
    if isinstance(ycols, str):
        ycols = [ycols]

    hist_data = [df[col] for col in ycols]
    # Create distplot with curve_type set to 'normal'
    fig = ff.create_distplot(hist_data, ycols, bin_size=0.2, show_rug=False)

    # Add title
    if title == "":
        title = ", ".join(ycols)
    fig.update_layout(title_text=title)
    fig.show(renderer=renderer)


def bar_plot(
    df: pd.DataFrame,
    xcol,
    ycol: str,
    estimator: str = "mean",
    return_fig=False,
    title="",
):
    """
    Bar blot with error bars/confidence intervals.
    """
    # order by mean of ycol
    fig = sns.barplot(
        data=df,
        x=xcol,
        y=ycol,
        errorbar="sd",
        estimator=estimator,
        order=df.groupby(xcol).mean().sort_values(ycol).index.unique(),
    )
    fig.set_title(title)
    if return_fig:
        return fig.get_figure()


def line_plot(
    df: pd.DataFrame,
    xcol: str,
    ycols: Union[List[str], str],
    renderer="browser",
    **filters,
):
    """
    Plots multiple y lines over xcol
    Example: smooth.html
    """
    df = _filter(df, filters)
    df = df.sort_values(xcol)
    if isinstance(ycols, str):
        ycols = [ycols]
    fig = px.line(df, x=xcol, y=ycols, markers=True)
    fig.update_layout(title=f"{filters}")
    fig.show(renderer=renderer)


def scatter_plot(
    df: pd.DataFrame,
    xcol: str,
    ycols: Union[List[str], str],
    renderer="browser",
    size=12,
    trendline=None,
):
    df = df.sort_values(xcol)
    if isinstance(ycols, str):
        ycols = [ycols]
    fig = px.scatter(df, x=xcol, y=ycols, trendline=trendline)
    fig.update_traces(marker=dict(size=size))
    fig.show(renderer=renderer)


def box_plot(df: pd.DataFrame, xcol: str, ycol: str, renderer="browser"):
    fig = px.box(df, x=xcol, y=ycol, points="all")
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


def _filter(df: pd.DataFrame, filters: Dict):
    cols = list(filters.keys())
    vals = [makelist(filters[col]) for col in cols]
    for col, val in zip(cols, vals):
        df = df[df[col].isin(val)]
    return df


def fct_plot(
    df: pd.DataFrame,
    xcol: str,
    ycols: Union[List[str], str],
    forecast_start: str = None,
    name: str = "",
    renderer="browser",
    marker_size=8,
    return_fig=False,
    **filters,
):
    df = _filter(df, filters)
    forecast_start = pd.to_datetime(forecast_start)
    if len(filters):
        name = ", ".join("{} = {}".format(key, value) for key, value in filters.items())
    fig = go.Figure(layout=DEFAULT_LAYOUT, layout_title_text=name)
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
        fig.add_shape(
            go.layout.Shape(
                type="line",
                yref="paper",
                x0=forecast_start,
                y0=0,
                x1=forecast_start,
                y1=1,
                line=dict(color="Red", width=1),
            )
        )
    fig.update_layout(title=name)
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
    df = _filter(df, filters)
    name = ", ".join("{} = {}".format(key, value) for key, value in filters.items())
    df["error"] = df[fct] - df[act]
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
