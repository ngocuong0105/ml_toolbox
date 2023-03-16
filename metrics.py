"""
METRICS
"""
import pandas as pd
import numpy as np
import sklearn.metrics as skm


# root mean squared error
def rmse_p(data, real, fct):
    if len(data) == 0:
        return 0
    return np.sqrt(skm.mean_squared_error(data[real], data[fct])) / np.mean(data[real])


# mean absolute error
def mae_p(data, real, fct):
    if len(data) == 0:
        return 0
    return skm.mean_absolute_error(data[real], data[fct]) / np.mean(data[real])


# mean absolute percentage error
def mape(data, real, fct):
    """
    sum(abs(actual - forecast) / actual) / n
    """
    if len(data) == 0:
        return 0
    return skm.mean_absolute_percentage_error(data[real], data[fct])


# maximum residual error
def max_error(data, real, fct):
    if len(data) == 0:
        return 0
    return skm.max_error(data[real], data[fct])


# median absolute error
def med_abs_err(data, real, fct):
    if len(data) == 0:
        return 0
    return skm.median_absolute_error(data[real], data[fct])


# coefficient of determination
def r2_score(data, real, fct):
    if len(data) == 0:
        return 0
    return skm.r2_score(data[real], data[fct])


# mean of abs difference normalised
def mean_abs(data, real, fct):
    if len(data) == 0:
        return 0
    err = np.mean(
        abs(data[real] - data[fct])
        / (pd.concat([data[real], data[fct]], axis=1).max(axis=1) + 0.1 ** 10)
    )
    return err


# sum of absolute differences over sum of maximums
def abs_over_max(data, real, fct):
    if len(data) == 0:
        return 0
    err = np.sum(abs(data[fct] - data[real])) / sum(np.maximum(data[fct], data[real]))
    return err


# symmetric mean absolute percentage error
def smape(data, real, fct):
    """
    SMAPE - https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    define SMAPE = 0, when fct and real are 0.
    """
    real, fct = data[real], data[fct]
    if len(real) == 0:
        return float("inf")

    sm = np.sum(
        np.where(
            np.logical_and(real == fct, real == 0),
            0,
            2 * np.abs(fct - real) / (np.abs(real) + np.abs(fct)),
        )
    )
    return 100 / len(real) * sm


# vectorized symmetric mean absolute percentage error
def vsmape(data, real, fct):
    """
    SMAPE - https://en.wikipedia.org/wiki/Symmetric_mean_absolute_percentage_error
    define SMAPE = 0, when fct and real are 0.
    """
    real, fct = data[real], data[fct]
    return 100 * np.where(
        np.logical_and(real == fct, real == 0),
        0,
        2 * np.abs(fct - real) / (np.abs(real) + np.abs(fct)),
    )


def mape_range(data, real, fct):
    """
    MAPE with denominator is the range ov values in the real column. Good metric for linear models.

    Explanation:
    Consider n observations. Lets call them n dots. They have an x value and y value.
    The goal is to determine if the 39 dots are on a line. First we find the best fit line that goes through
    these n points using linear regression. Next for each of the n dots we compute their error which is the y
    axis distance they are from the line. If all n dots were on the line, then error would be zero.
    Once we have error, we need to decide if it is large or small. To do this,
    we find the range of y axis for dots. We takes dot with largest y value and subtract from dot with smallest y value.
    Then for each error we compute its ratio with range. So if range is 10 units.
    And dot is 1 unit away from line. Then error is 10%. Lastly we average all these percentage errors.
    If the average percentage is less than some threshold, we declare the points are linear.
    """
    if len(data) == 0:
        return 0
    return np.mean(np.abs(data[fct] - data[real])) / (
        np.max(data[real]) - np.min(data[real])
    )
