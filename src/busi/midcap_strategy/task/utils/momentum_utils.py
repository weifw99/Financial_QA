# utils/momentum_utils.py

import numpy as np
from scipy.stats import linregress


def get_momentum(prices: list | np.ndarray, method: str = 'log', days: int = 20) -> float:
    """
    计算动量指标：
    - log: 对数收益率（默认）
    - return: 简单收益率
    - slope: 线性拟合斜率（标准化）
    - r2: 回归拟合 R²（趋势强度）
    - slope_r2: 斜率 * R² 组合动量指标
    """
    prices = np.array(prices)
    if len(prices) < days + 1:
        return np.nan

    prices = prices[-(days + 1):]
    if np.any(prices <= 0):
        return np.nan

    if method == 'log':
        return np.log(prices[-1] / prices[0])
    elif method == 'return':
        return (prices[-1] - prices[0]) / prices[0]
    elif method in ['slope', 'r2', 'slope_r2']:
        x = np.arange(len(prices))
        log_prices = np.log(prices)
        slope, intercept, r_value, p_value, std_err = linregress(x, log_prices)
        if method == 'slope':
            return slope
        elif method == 'r2':
            return r_value ** 2
        elif method == 'slope_r2':
            return slope * (r_value ** 2)
    else:
        raise ValueError(f"不支持的动量方法: {method}")