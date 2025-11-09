import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
import traceback
import warnings
warnings.filterwarnings("ignore")


# ============================================================
# 🧱 1️⃣ 数据准备
# ============================================================
def prepare_data(df, start_date=None, end_date=None):
    df = df.copy()
    df['date'] = pd.to_datetime(df['date'])
    df = df.sort_values(['symbol', 'date'])

    # === 🧹 关键修复：去除重复 (symbol, date) ===
    if df.duplicated(subset=['symbol', 'date']).any():
        print(f"⚠️ 检测到 {df.duplicated(subset=['symbol', 'date']).sum()} 条重复记录，已自动去重（取最后一条）")
        df = df.drop_duplicates(subset=['symbol', 'date'], keep='last')

    price_df = df.pivot(index='date', columns='symbol', values='close')
    price_df = price_df.fillna(method='ffill')
    price_df = price_df.fillna(0.001)

    if start_date:
        price_df = price_df[price_df.index >= pd.to_datetime(start_date)]
    if end_date:
        price_df = price_df[price_df.index <= pd.to_datetime(end_date)]

    print(f"✅ 数据区间：{price_df.index.min().date()} ~ {price_df.index.max().date()}，共 {len(price_df)} 天")
    return price_df


# ============================================================
# 🧮 2️⃣ 动量计算
# ============================================================
def calc_momentum(prices, window, method='total_return'):
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
# 🗓️ 3️⃣ 调仓日期生成
# ============================================================
def generate_rebalance_dates(price_df, freq='month', weekday=0, month_day=10):
    dates = price_df.index
    rebalance_dates = []

    if freq == 'week':
        rebalance_dates = [d for d in dates if d.weekday() == weekday]
    elif freq == 'month':
        months = sorted(set((d.year, d.month) for d in dates))
        for y, m in months:
            target = pd.Timestamp(year=y, month=m, day=month_day)
            valid = dates[dates >= target]
            if len(valid) > 0:
                rebalance_dates.append(valid[0])
    else:
        raise ValueError("freq 必须为 'week' 或 'month'")
    print(f"📆 生成 {len(rebalance_dates)} 个调仓日（freq={freq}）")
    return rebalance_dates

# ============================================================
# ⚙️ 4️⃣ 回测逻辑（含资金分配、手续费、滑点）
# ============================================================
def backtest(prices,
             momentum_window=120,
             method='total_return',
             n_select=2,
             init_cash=1_000_000,
             fee_rate=0.0005,
             slippage=0.0003,
             freq='month',
             weekday=0,
             month_day=10):
    import traceback

    print(f"\n🚀 回测开始：window={momentum_window}, method={method}, top={n_select}, freq={freq}")

    returns = prices.pct_change().dropna()
    momentum = calc_momentum(prices, window=momentum_window, method=method)
    rebalance_dates = generate_rebalance_dates(prices, freq=freq, weekday=weekday, month_day=month_day)

    # 初始化资金
    total_value = pd.Series(index=prices.index, dtype=float)
    total_value.iloc[0] = init_cash
    holdings = {}  # symbol -> 持仓金额

    for j, date in enumerate(rebalance_dates):
        if date not in momentum.index:
            continue

        try:
            # === 选出动量Top N ===
            recent_mom = momentum.loc[date].dropna().sort_values(ascending=False)
            top_etfs = recent_mom.index[:n_select].tolist()
            print(f"📅 调仓日 {date.date()}：选择 {top_etfs}")

            if len(top_etfs) == 0:
                continue

            # === 安全获取当前账户资金 ===
            past_values = total_value.loc[:date].dropna()
            current_value = past_values.iloc[-1] if len(past_values) > 0 else init_cash
            print(f"💰 {date.date()} 账户余额：{current_value:.2f}")

            # === 资金分配：平均分配到N个标的 ===
            each_value = current_value / n_select
            holdings = {sym: each_value for sym in top_etfs}

            # === 获取下一个调仓日期（或最后一天） ===
            next_date = rebalance_dates[j + 1] if j < len(rebalance_dates) - 1 else prices.index[-1]

            # === 模拟每日组合净值变化 ===
            period_idx = (returns.index >= date) & (returns.index <= next_date)
            period_dates = returns.index[period_idx]
            if len(period_dates) == 0:
                print(f"⚠️ 无有效交易日: {date} ~ {next_date}, 跳过 period_idx: {period_idx}, returns.index: {returns.index}")
                continue

            port_daily = pd.Series(index=period_dates, dtype=float)

            # 每日净值更新
            for i, t in enumerate(period_dates):
                day_ret = returns.loc[t, top_etfs].fillna(0)
                daily_port_ret = day_ret.mean()
                current_value *= (1 + daily_port_ret)

                # 调仓当天扣手续费+滑点
                if i == 0:
                    current_value *= (1 - fee_rate - slippage)

                port_daily.loc[t] = current_value

            # 更新总资产
            total_value.loc[period_dates] = port_daily

            # 为下一次调仓更新资金
            current_value = port_daily.iloc[-1]

            print(f"💰 {date.date()} 调仓后净值：{current_value:.2f}")

        except Exception as e:
            print(f"❌ 调仓 {date.date()} 出错: {e}")
            traceback.print_exc()

    # === 绩效指标计算 ===
    nav = total_value.ffill()
    daily_ret = nav.pct_change().dropna()
    if len(daily_ret) == 0:
        print("⚠️ 无有效交易区间，跳过。")
        return None

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
# 🔍 5️⃣ 网格搜索 + 最优结果
# ============================================================
def grid_search(prices, momentum_windows, methods, n_select_list, **kwargs):
    results = []
    for mw, mtd, ns in product(momentum_windows, methods, n_select_list):
        try:
            res = backtest(prices, momentum_window=mw, method=mtd, n_select=ns, **kwargs)
            if res is not None and np.isfinite(res['sharpe']):
                results.append(res)
        except Exception as e:
            print(f"❌ 参数组合失败: {mw}, {mtd}, {ns}, 错误: {e}")
            traceback.print_exc()
    return pd.DataFrame(results)


# ============================================================
# 🧩 6️⃣ 示例主流程
# ============================================================
# df = pd.read_csv('etf_data.csv')
# price_df = prepare_data(df, start_date='2019-01-01', end_date='2025-01-01')

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
n_select_list = [2, 3]

results_df = grid_search(price_df, momentum_windows, methods, n_select_list,
                         init_cash=1_000_000, freq='month', month_day=10)

if results_df.empty:
    print("\n⚠️ 无有效结果，请检查数据或参数设置。")
else:
    best = results_df.loc[results_df['sharpe'].idxmax()]
    print("\n🌟 最优参数：")
    print(best[['momentum_window', 'method', 'n_select', 'freq', 'sharpe', 'annual_ret', 'max_dd']])

    # 绘制净值曲线
    plt.figure(figsize=(10,6))
    plt.plot(best['nav'], label=f"{best['method']} {int(best['momentum_window'])}d Top{int(best['n_select'])}")
    plt.title("ETF 动量轮动策略回测（资金分配版）")
    plt.xlabel("日期")
    plt.ylabel("净值")
    plt.legend()
    plt.grid(True)
    plt.show()
