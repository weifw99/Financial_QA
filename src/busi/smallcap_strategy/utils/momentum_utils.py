import numpy as np
from scipy.stats import linregress


def calc_simple_return(prices, days=20):
    """简单收益率动量"""
    if len(prices) < days + 1:
        return np.nan
    return prices[-1] / prices[-(days + 1)] - 1


def calc_log_return(prices, days=20):
    """对数收益率动量"""
    if len(prices) < days + 1:
        return np.nan
    prev_price = prices[-(days + 1)]
    if prev_price == 0 or np.isnan(prev_price) or np.isnan(prices[-1]):
        return -999  # 或者你设置的极端异常值
    return np.log(prices[-1] / prev_price)


def calc_regression_slope(prices, days=20):
    """线性拟合斜率"""
    if len(prices) < days + 1:
        return np.nan
    y = np.log(prices[-(days + 1):])  # 使用 log(price)
    x = np.arange(len(y))
    slope, _, _, _, _ = linregress(x, y)
    return slope


def calc_slope_r2(prices, days=20):
    """复合动量 = slope × R²"""
    if len(prices) < days + 1:
        return np.nan
    y = np.log(prices[-(days + 1):])
    x = np.arange(len(y))
    slope, _, r_value, _, _ = linregress(x, y)
    return slope * (r_value ** 2)


def get_momentum(prices, method="slope_r2", days=20):
    """
    通用动量接口

    参数:
        prices (np.ndarray): 最近的价格序列
        method (str): 'return' / 'log' / 'slope' / 'slope_r2'
        days (int): 动量观察窗口
    返回:
        float: 动量值
    """
    if isinstance(prices, list):
        prices = np.array(prices)

    if method == "return":
        return calc_simple_return(prices, days)
    elif method == "log":
        return calc_log_return(prices, days)
    elif method == "slope":
        return calc_regression_slope(prices, days)
    elif method == "slope_r2":
        return calc_slope_r2(prices, days)
    else:
        raise ValueError(f"未知动量计算方式: {method}")