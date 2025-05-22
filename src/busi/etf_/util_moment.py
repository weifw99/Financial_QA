from scipy import stats
import numpy as np
import pandas as pd


def momentum_linear(closes, window=20):
    if len(closes) < window:
        return np.nan
    x = np.arange(window)
    y = np.log(closes[-window:])
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope * (r_value ** 2)


def momentum_simple(closes, window=20):
    if len(closes) < window:
        return np.nan
    return closes[-1] / closes[-window] - 1


def log_momentum_simple(closes, window=20):
    if len(closes) < window:
        return np.nan
    return np.log(closes[-1] / closes[-window])


def log_momentum_r2(closes, window=20):
    if len(closes) < window:
        return np.nan
    x = np.arange(window)
    y = np.log(closes[-window:])
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return np.log(closes[-1] / closes[-window]) * (r_value ** 2)


def line_log_momentum_r2(closes, window=20):
    if len(closes) < window:
        return np.nan
    x = np.arange(window)
    y = np.log(closes[-window:])
    slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
    return slope * np.log(closes[-1] / closes[-window]) * (r_value ** 2)


def momentum_dual(closes,
                  long_window=90, short_window=20,
                  smooth_long=20, smooth_short=5,
                  min_long_return=0.02, min_short_return=0.01,
                  long_weight=0.7, short_weight=0.3):
    def _score(prices, win, smooth, min_ret):
        if len(prices) < win:
            return np.nan
        smooth_prices = pd.Series(prices[-win:]).rolling(smooth).mean().dropna().values
        if len(smooth_prices) < win - smooth + 1:
            return np.nan
        total_return = np.log(smooth_prices[-1] / smooth_prices[0])
        if total_return < min_ret:
            return np.nan
        x = np.arange(len(smooth_prices))
        y = np.log(smooth_prices)
        slope, intercept, r_value, p_value, std_err = stats.linregress(x, y)
        return slope * (r_value ** 1.5)

    long_score = _score(closes, long_window, smooth_long, min_long_return)
    short_score = _score(closes, short_window, smooth_short, min_short_return)

    if np.isnan(long_score) and np.isnan(short_score):
        return np.nan
    elif np.isnan(long_score):
        return short_score * short_weight
    elif np.isnan(short_score):
        return long_score * long_weight
    else:
        return long_score * long_weight + short_score * short_weight


def momentum_dual_v2(
    closes,
    long_window=90,
    short_window=20,
    smooth_long=20,
    smooth_short=5,
    min_long_return=0.02,
    min_short_return=0.01,
    slope_positive_filter=True,
    weight_long=0.7,
    weight_short=0.3,
):
    def _compute(prices, win, smooth, min_ret):
        if len(prices) < win:
            return np.nan, np.nan
        smooth_prices = pd.Series(prices[-win:]).rolling(smooth).mean().dropna().values
        if len(smooth_prices) < win - smooth + 1:
            return np.nan, np.nan
        total_return = np.log(smooth_prices[-1] / smooth_prices[0])
        if total_return < min_ret:
            return np.nan, np.nan
        x = np.arange(len(smooth_prices))
        y = np.log(smooth_prices)
        slope, _, r_value, _, _ = stats.linregress(x, y)
        score = slope * (r_value ** 1.5)
        return score, slope

    long_score, long_slope = _compute(closes, long_window, smooth_long, min_long_return)
    short_score, short_slope = _compute(closes, short_window, smooth_short, min_short_return)

    # 过滤斜率方向
    if slope_positive_filter:
        if (long_slope is not None and long_slope < 0) or (short_slope is not None and short_slope < 0):
            return np.nan

    if np.isnan(long_score) and np.isnan(short_score):
        return np.nan
    elif np.isnan(long_score):
        return short_score * weight_short
    elif np.isnan(short_score):
        return long_score * weight_long
    else:
        return long_score * weight_long + short_score * weight_short