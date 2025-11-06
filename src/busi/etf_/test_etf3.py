import pybroker as pyb
from pybroker import ExecContext, Strategy, StrategyConfig, FeeMode
import pandas as pd
import talib as ta
import numpy as np
import plotly.graph_objects as go

# ======================
# 1️⃣ 读取数据
# ======================
# csv_path = 'etf_prices.csv'  # 你的 ETF 数据
# df = pd.read_csv(csv_path)


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


df['date'] = pd.to_datetime(df['date'])
df.sort_values(['symbol', 'date'], inplace=True)

# 注册自定义指标
pyb.register_columns('roc_20', 'volatility_20', 'volume_mom_20', 'momentum_score')

# ======================
# 2️⃣ 计算指标
# ======================
def calc_indicators(df, momentum_window=20):
    indicators = []
    for sym, sub in df.groupby('symbol'):
        close = sub['close'].astype(float).values
        volume = sub['volume'].astype(float).values

        roc = ta.ROC(close, timeperiod=momentum_window)
        volatility = ta.STDDEV(close, timeperiod=momentum_window)
        vol_mom = ta.ROC(volume, timeperiod=momentum_window)

        # 简单动量融合：价量 + 波动调整
        momentum_score = roc * 0.6 + vol_mom * 0.3 - volatility * 0.1

        tmp = sub.copy()
        tmp['roc_20'] = roc
        tmp['volatility_20'] = volatility
        tmp['volume_mom_20'] = vol_mom
        tmp['momentum_score'] = momentum_score

        indicators.append(tmp)
    return pd.concat(indicators)

df = calc_indicators(df, momentum_window=20)

# ======================
# 3️⃣ 策略配置
# ======================
config = StrategyConfig(
    max_long_positions=3,
    fee_mode=FeeMode.ORDER_PERCENT,  # 按订单金额百分比收取
    fee_amount=0.0005,               # 万分之五
    subtract_fees=True                # 是否从现金中扣除手续费
)
pyb.param('target_size', 1 / config.max_long_positions)
pyb.param('rank_threshold', 3)  # Top N
pyb.param('max_drawdown', 0.2)  # 最大回撤止损

# ======================
# 4️⃣ 排名函数
# ======================
def rank(ctxs: dict[str, ExecContext]):
    scores = {sym: ctx.momentum_score[-1] for sym, ctx in ctxs.items()}
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    threshold = pyb.param('rank_threshold')
    top_scores = sorted_scores[:threshold]
    pyb.param('top_symbols', [s[0] for s in top_scores])

# ======================
# 5️⃣ 轮动函数
# ======================
def rotate(ctx: ExecContext):
    # 卖出不在 top N 的持仓
    if ctx.long_pos() and ctx.symbol not in pyb.param('top_symbols'):
        ctx.sell_all_shares()
    # 买入 top N ETF
    elif ctx.symbol in pyb.param('top_symbols') and not ctx.long_pos():
        target_size = pyb.param('target_size')
        ctx.buy_shares = ctx.calc_target_shares(target_size)
        ctx.score = ctx.momentum_score[-1]

# ======================
# 6️⃣ 回测
# ======================
symbols = df['symbol'].unique().tolist()
strategy = Strategy(df, start_date='2019-01-01', end_date='2025-01-01', config=config)
strategy.set_before_exec(rank)


roc_20 = pyb.indicator('roc_20', lambda data: ta.ROC(data.close.astype(np.float64), timeperiod=20))
volatility_20 = pyb.indicator('volatility_20', lambda data: ta.STDDEV(data.close.astype(np.float64), timeperiod=20))
volume_mom_20 = pyb.indicator('volume_mom_20', lambda data: ta.ROC(data.volume.astype(np.float64), timeperiod=20))

momentum_score = pyb.indicator(
    'momentum_score',
    lambda data: 0.6 * ta.ROC(data.close.astype(np.float64), timeperiod=20) +
                 0.3 * ta.ROC(data.volume.astype(np.float64), timeperiod=20) -
                 0.1 * ta.STDDEV(data.close.astype(np.float64), timeperiod=20)
)
strategy.add_execution(
    rotate,
    symbols,
    # indicators=[roc_20, volatility_20, volume_mom_20, momentum_score]
)

result = strategy.backtest(warmup=20)

# ======================
# 7️⃣ 输出交易记录 & 净值
# ======================
orders_df = result.orders
orders_df.to_csv('etf_momentum_orders.csv', index=False)
result.portfolio.to_csv('etf_momentum_portfolio.csv', index=False)

# 绘制交互式净值曲线
# fig = go.Figure()
# fig.add_trace(go.Scatter(y=result.nav, x=result.nav.index, mode='lines', name='Strategy NAV'))
# fig.show()

import plotly.graph_objects as go

print(result)
print(result.portfolio)
print(result.portfolio.columns)
fig = go.Figure()
fig.add_trace(
    go.Scatter(
        x=result.portfolio.index,
        y=result.portfolio['market_value'].values,
        mode='lines',
        name='Strategy NAV'
    )
)
fig.show()