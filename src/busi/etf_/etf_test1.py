import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import traceback
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 🧱 1️⃣ 数据准备与时间过滤
# ============================================================
def prepare_data(df: pd.DataFrame, start_date=None, end_date=None):
    """
    准备数据：透视、清洗、按时间过滤
    """
    df = df.copy()
    df = df.sort_values(['symbol', 'date'])
    df['date'] = pd.to_datetime(df['date'])
    symbols = df['symbol'].unique()
    print(f"✅ 加载 {len(symbols)} 只ETF，样例：{symbols[:5]}")

    # 构建收盘价矩阵
    price_df = df.pivot(index='date', columns='symbol', values='close')
    price_df = price_df.fillna(method='ffill')
    price_df = price_df.fillna(0.0001)  # 用0填充

    # 时间过滤
    if start_date:
        price_df = price_df[price_df.index >= pd.to_datetime(start_date)]
    if end_date:
        price_df = price_df[price_df.index <= pd.to_datetime(end_date)]

    print(f"📅 数据区间：{price_df.index.min().date()} 至 {price_df.index.max().date()}，共 {len(price_df)} 个交易日")
    return price_df


# ============================================================
# 🧮 2️⃣ 动量计算
# ============================================================
def calc_momentum(prices: pd.DataFrame, window: int, method: str = 'total_return'):
    """
    计算动量指标
    """
    if method == 'total_return':
        mom = prices / prices.shift(window) - 1
    elif method == 'slope':
        mom = prices.rolling(window).apply(
            lambda x: np.polyfit(np.arange(len(x)), x, 1)[0] if np.all(np.isfinite(x)) else np.nan,
            raw=True
        )
    elif method == 'vol_adj_return':
        ret = prices.pct_change()
        cumret = prices / prices.shift(window) - 1
        vol = ret.rolling(window).std()
        mom = cumret / vol
    else:
        raise ValueError(f"未知动量计算方式: {method}")
    return mom


# ============================================================
# ⚙️ 3️⃣ 调仓日生成器
# ============================================================
def generate_rebalance_dates(price_df, freq='month', weekday=0, month_day=1):
    """
    根据调仓频率生成调仓日期序列：
    freq='week' → 每周调仓（weekday=0 表示周一）
    freq='month' → 每月调仓（month_day=1 表示每月1号，若不是交易日顺延）
    """
    dates = price_df.index
    rebalance_dates = []

    if freq == 'week':
        # 每周指定星期几（0=周一,...,4=周五）
        rebalance_dates = [d for d in dates if d.weekday() == weekday]
    elif freq == 'month':
        # 每月指定日期（若非交易日顺延）
        months = sorted(set((d.year, d.month) for d in dates))
        for y, m in months:
            target = pd.Timestamp(year=y, month=m, day=month_day)
            valid = dates[dates >= target]
            if len(valid) > 0:
                rebalance_dates.append(valid[0])
    else:
        raise ValueError("freq 参数必须是 'week' 或 'month'")

    print(f"📆 共生成 {len(rebalance_dates)} 个调仓日（频率={freq}）")
    return rebalance_dates


# ============================================================
# 🧭 4️⃣ 回测函数
# ============================================================
def backtest(prices: pd.DataFrame,
             momentum_window: int,
             method: str,
             n_select: int,
             fee_rate: float = 0.0005,
             slippage: float = 0.0003,
             freq: str = 'month',
             weekday: int = 0,
             month_day: int = 1):
    """
    ETF 动量轮动策略回测
    """
    print(f"\n🚀 回测开始：window={momentum_window}, method={method}, top={n_select}, freq={freq}")
    returns = prices.pct_change().dropna()
    momentum = calc_momentum(prices, window=momentum_window, method=method)
    rebalance_dates = generate_rebalance_dates(prices, freq=freq, weekday=weekday, month_day=month_day)

    portfolio_value = pd.Series(index=prices.index, dtype=float)
    portfolio_value.iloc[0] = 1.0
    holdings = None

    for j, date in enumerate(rebalance_dates):
        if date not in momentum.index:
            continue
        try:
            # 当前动量排名
            recent_mom = momentum.loc[date].dropna().sort_values(ascending=False)
            top_etfs = recent_mom.index[:n_select]
            print(f"📅 调仓日 {date.date()}：选择 {list(top_etfs)}")

            # ✅ 获取下一个调仓日期（安全方式）
            if j < len(rebalance_dates) - 1:
                next_date = rebalance_dates[j + 1]
            else:
                next_date = prices.index[-1]  # 最后一个日期

            # 提取期间收益
            mask = (returns.index > date) & (returns.index <= next_date)
            period_rets = returns.loc[mask, top_etfs]
            if period_rets.empty:
                continue

            portfolio_period = period_rets.mean(axis=1)

            # 手续费与滑点
            if holdings is not None:
                turnover_cost = fee_rate * 2 + slippage
                portfolio_period.iloc[0] -= turnover_cost

            holdings = top_etfs
            portfolio_value.loc[period_rets.index] = portfolio_value.loc[date] * (1 + portfolio_period).cumprod()

        except Exception as e:
            print(f"❌ 调仓 {date.date()} 出错: {e}")
            import traceback;
            traceback.print_exc()

    # 绩效统计
    nav = portfolio_value.ffill()
    daily_ret = nav.pct_change().dropna()
    annual_ret = (1 + daily_ret.mean()) ** 252 - 1
    annual_vol = daily_ret.std() * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else np.nan
    max_dd = ((nav / nav.cummax()) - 1).min()

    print(f"✅ 回测完成: Sharpe={sharpe:.2f}, 年化收益={annual_ret:.2%}, 最大回撤={max_dd:.2%}")
    return {
        'momentum_window': momentum_window,
        'method': method,
        'n_select': n_select,
        'freq': freq,
        'annual_ret': annual_ret,
        'annual_vol': annual_vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'nav': nav
    }


# ============================================================
# 🔍 5️⃣ 网格搜索
# ============================================================
def grid_search(prices, momentum_windows, methods, n_select_list, freq='month', weekday=0, month_day=1):
    results = []
    for mw, mtd, ns in product(momentum_windows, methods, n_select_list):
        try:
            res = backtest(prices, momentum_window=mw, method=mtd, n_select=ns,
                           freq=freq, weekday=weekday, month_day=month_day)
            results.append(res)
        except Exception as e:
            print(f"❌ 参数组合失败: {mw}, {mtd}, {ns}, 错误: {e}")
            traceback.print_exc()
    return pd.DataFrame(results)


# ============================================================
# 🧩 6️⃣ 主流程
# ============================================================
# 示例运行
# df = pd.read_csv('etf_data.csv')

from busi.etf_.bt_data import Getdata

pool_file = 'data/etf_strategy/etf_pool_120.csv'
pool_file = 'data/etf_strategy/etf_pool.csv'
pool_file = 'data/etf_strategy/etf_pool1.csv'
df = pd.read_csv(pool_file)
etf_codes = df['代码'].tolist()
# 获取数据源
datas = Getdata(symbols=etf_codes)
data_1 = datas.dailydata_no_index()

# 示例：df = pd.read_csv('etf_data.csv', parse_dates=['date'])
df = data_1.sort_values(['symbol', 'date']).copy()

price_df = prepare_data(df, start_date='2019-01-01', end_date='2025-01-01')

# 参数空间
momentum_windows = [60, 120]
methods = ['total_return', 'vol_adj_return']
n_select_list = [1, 2]

results_df = grid_search(price_df, momentum_windows, methods, n_select_list,
                         freq='month', month_day=10)

# 选最优
best = results_df.loc[results_df['sharpe'].idxmax()]
print("\n🌟 最优参数：")
print(best[['momentum_window', 'method', 'n_select', 'freq', 'sharpe', 'annual_ret', 'max_dd']])

# 绘图
plt.figure(figsize=(10,6))
plt.plot(best['nav'], label=f"{best['method']} {int(best['momentum_window'])}d Top{int(best['n_select'])}")
plt.title("ETF 动量轮动回测（最优参数）")
plt.xlabel("日期")
plt.ylabel("净值")
plt.legend()
plt.grid(True)
plt.show()
