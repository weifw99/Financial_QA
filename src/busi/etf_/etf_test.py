import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from itertools import product
from busi.etf_.bt_data import Getdata
import warnings
warnings.filterwarnings("ignore")

# ============================================================
# 🧱 1️⃣ 数据准备与格式化
# ============================================================
# 假设 df 是原始日线数据（包含多只 ETF）
# 列: ['symbol', 'date', 'open', 'high', 'low', 'close', 'volume']


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

# 提取所有 ETF 代码
symbols = df['symbol'].unique()
print(f"共加载 {len(symbols)} 只ETF，样例：{symbols[:5]}")

# 生成收盘价透视表 (index=date, columns=symbol)
price_df = df.pivot(index='date', columns='symbol', values='close').dropna(how='all')
price_df = price_df.fillna(method='ffill')  # 向前填充缺失值
price_df = price_df.fillna(0)
print(f"价格矩阵形状: {price_df.shape}")

# ============================================================
# 🧮 2️⃣ 动量计算函数
# ============================================================
def calc_momentum(prices: pd.DataFrame, window: int, method: str = 'total_return'):
    """
    根据不同方式计算动量
    参数:
        prices : DataFrame，每列为一只ETF的价格序列
        window : int，动量回看期天数
        method : str，动量计算方式
            'total_return' ：累计收益率
            'slope'        ：趋势斜率（回归拟合）
            'vol_adj_return'：波动调整后的累计收益
    返回:
        DataFrame，每列为ETF的动量值
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
# ⚙️ 3️⃣ 回测主逻辑
# ============================================================
def backtest(prices: pd.DataFrame,
             momentum_window: int,
             method: str,
             n_select: int,
             fee_rate: float = 0.0005,
             slippage: float = 0.0003,
             rebalance_period: int = 63):
    """
    ETF 动量轮动策略回测
    每季度调仓一次（默认63交易日），持有动量最高的N只ETF
    """
    returns = prices.pct_change().dropna()
    momentum = calc_momentum(prices, window=momentum_window, method=method)

    portfolio_value = pd.Series(index=prices.index, dtype=float)
    portfolio_value.iloc[0] = 1.0
    holdings = None

    print(f"🚀 回测开始：window={momentum_window}, method={method}, top={n_select}")

    for i in range(momentum_window, len(prices), rebalance_period):
        date = prices.index[i]
        # 当前时点所有ETF动量
        recent_mom = momentum.iloc[i].dropna().sort_values(ascending=False)
        top_etfs = recent_mom.index[:n_select]
        print(f"📅 调仓日 {date.date()}：选择 {list(top_etfs)}")

        next_idx = min(i + rebalance_period, len(prices) - 1)
        # 未来一个调仓周期内收益
        period_rets = returns.loc[prices.index[i+1:next_idx+1], top_etfs]
        portfolio_period = period_rets.mean(axis=1)

        # 扣除交易成本（首日）
        if holdings is not None:
            turnover_cost = fee_rate * 2 + slippage
            portfolio_period.iloc[0] -= turnover_cost

        holdings = top_etfs
        portfolio_value.loc[period_rets.index] = portfolio_value.loc[prices.index[i]] * (1 + portfolio_period).cumprod()

    # 绩效指标计算
    nav = portfolio_value.ffill()
    daily_ret = nav.pct_change().dropna()
    annual_ret = (1 + daily_ret.mean()) ** 252 - 1
    annual_vol = daily_ret.std() * np.sqrt(252)
    sharpe = annual_ret / annual_vol if annual_vol > 0 else np.nan
    max_dd = ((nav / nav.cummax()) - 1).min()

    print(f"✅ 完成 window={momentum_window}, method={method}, top={n_select}, Sharpe={sharpe:.2f}, 年化收益={annual_ret:.2%}")

    return {
        'momentum_window': momentum_window,
        'method': method,
        'n_select': n_select,
        'annual_ret': annual_ret,
        'annual_vol': annual_vol,
        'sharpe': sharpe,
        'max_dd': max_dd,
        'nav': nav
    }

# ============================================================
# 🔍 4️⃣ 网格搜索参数设定
# ============================================================
momentum_windows = [60, 120, 180]
methods = ['total_return', 'slope', 'vol_adj_return']
n_select_list = [1, 2, 3]

results = []

# 遍历所有参数组合
for mw, mtd, ns in product(momentum_windows, methods, n_select_list):
    try:
        res = backtest(price_df, momentum_window=mw, method=mtd, n_select=ns)
        results.append(res)
    except Exception as e:
        print(f"❌ 参数组合失败: {mw}, {mtd}, {ns}, 错误: {e}")

# 汇总结果
results_df = pd.DataFrame(results)
best = results_df.loc[results_df['sharpe'].idxmax()]
print("\n🌟 最优参数组合：")
print(best[['momentum_window', 'method', 'n_select', 'sharpe', 'annual_ret', 'max_dd']])

# ============================================================
# 📈 5️⃣ 可视化最优策略净值
# ============================================================
plt.figure(figsize=(10,6))
plt.plot(best['nav'], label=f"Best Strategy ({best['method']}, {int(best['momentum_window'])}天, Top {int(best['n_select'])})", color='blue')
plt.title("ETF 动量轮动策略回测（最优参数）")
plt.xlabel("日期")
plt.ylabel("组合净值")
plt.legend()
plt.grid(True)
plt.show()

# 打印前十名结果
print("\n🏁 前十策略表现：")
print(results_df.sort_values('sharpe', ascending=False).head(10)[
    ['momentum_window', 'method', 'n_select', 'sharpe', 'annual_ret', 'max_dd']
])
