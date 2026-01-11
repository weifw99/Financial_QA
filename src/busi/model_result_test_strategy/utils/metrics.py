
# utils/metrics.py
# 提供策略评估指标计算，包括收益率、年化波动、夏普比率、最大回撤等
import numpy as np
import pandas as pd

def calc_metrics(nav_series: pd.Series, rf=0.02):
    """
    计算常用回测评估指标：总收益、年化收益、年化波动、夏普比率、最大回撤
    :param nav_series: 净值序列（建议为 pd.Series，index 为日期）
    :param rf: 无风险利率，默认为 2%
    :return: dict 格式的指标结果
    """
    returns = nav_series.pct_change().dropna()
    total_return = nav_series.iloc[-1] / nav_series.iloc[0] - 1
    annual_return = (1 + total_return) ** (252 / len(nav_series)) - 1
    annual_vol = returns.std() * np.sqrt(252)
    sharpe = (annual_return - rf) / annual_vol if annual_vol != 0 else 0

    # 最大回撤
    cummax = nav_series.cummax()
    drawdown = nav_series / cummax - 1
    max_drawdown = drawdown.min()

    return {
        "Total Return": round(total_return, 4),
        "Annual Return": round(annual_return, 4),
        "Annual Volatility": round(annual_vol, 4),
        "Sharpe Ratio": round(sharpe, 4),
        "Max Drawdown": round(max_drawdown, 4)
    }

def plot_nav(nav_series: pd.Series, title='Net Value Curve', path=None):
    import matplotlib.pyplot as plt
    plt.figure(figsize=(10, 4))
    plt.plot(nav_series, label='Net Value')
    plt.title(title)
    plt.grid(True)
    plt.legend()
    if path:
        plt.savefig(path)
    plt.show()

def calculate_momentum(df, window=20):
    return df['close'].pct_change(periods=window)

