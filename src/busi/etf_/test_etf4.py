import pybroker as pyb
from pybroker import ExecContext, Strategy, StrategyConfig, FeeMode
import pandas as pd
import numpy as np
import math
import itertools

# ======================
# 1️⃣ 数据读取 & 时间过滤
# ======================
from busi.etf_.bt_data import Getdata

datas = Getdata(symbols=None)
data_1 = datas.dailydata_no_index()
df = data_1.sort_values(['symbol', 'date']).copy()
df['date'] = pd.to_datetime(df['date'])


import os
data_dir = 'data/etf_strategy/'
files = [f for f in os.listdir(data_dir) if f.endswith('.csv')]
# pool_file = 'data/etf_strategy/etf_pool.csv'

etf_codes = []
etf_codes.append( df['symbol'].unique().tolist() )
for pool_file in files:
    df_pool = pd.read_csv(data_dir + pool_file)
    etf_codes.append( df_pool['代码'].tolist() )


# 注册自定义指标
pyb.register_columns('momentum_score')

# ======================
# 2️⃣ 数据过滤函数
# ======================
def filter_data(df, start_date, end_date):
    df = df[(df['date'] >= pd.to_datetime(start_date)) & (df['date'] <= pd.to_datetime(end_date))]
    return df


# ======================
# 2️⃣ 动量计算函数
# ======================
def calc_momentum_score(close, method, window):
    if len(close) <= window:
        return 0.0

    if method == 'simple_window':
        return close[-1] - close[-window]

    elif method == 'log_simple_window':
        return math.log(close[-1] / close[-window]) if close[-1] > 0 and close[-window] > 0 else 0.0

    elif method == 'linear_window':
        y = close[-window:]
        x = list(range(window))
        x_mean = sum(x) / window
        y_mean = sum(y) / window
        numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        denominator = sum((xi - x_mean) ** 2 for xi in x)
        return numerator / denominator if denominator != 0 else 0.0

    elif method == 'log_r2_window':
        y = [math.log(c) for c in close[-window:] if c > 0]
        if len(y) < window:
            return 0.0
        x = list(range(window))
        x_mean = sum(x) / window
        y_mean = sum(y) / window
        ss_total = sum((yi - y_mean) ** 2 for yi in y)
        ss_reg = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        slope = ss_reg / sum((xi - x_mean) ** 2 for xi in x) if sum((xi - x_mean) ** 2 for xi in x) != 0 else 0.0
        y_hat = [slope * (xi - x_mean) + y_mean for xi in x]
        ss_res = sum((yi - yhi) ** 2 for yi, yhi in zip(y, y_hat))
        r2 = 1 - ss_res / ss_total if ss_total != 0 else 0.0
        return r2

    elif method == 'line_log_r2_window':
        y = [math.log(c) for c in close[-window:] if c > 0]
        if len(y) < window:
            return 0.0
        x = list(range(window))
        x_mean = sum(x) / window
        y_mean = sum(y) / window
        ss_total = sum((yi - y_mean) ** 2 for yi in y)
        ss_reg = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
        slope = ss_reg / sum((xi - x_mean) ** 2 for xi in x) if sum((xi - x_mean) ** 2 for xi in x) != 0 else 0.0
        y_hat = [slope * (xi - x_mean) + y_mean for xi in x]
        ss_res = sum((yi - yhi) ** 2 for yi, yhi in zip(y, y_hat))
        r2 = 1 - ss_res / ss_total if ss_total != 0 else 0.0
        return slope * r2

    else:
        raise NotImplementedError(f"Momentum method {method} not implemented")


# ======================
# 3️⃣ 策略指标计算函数
# ======================
def calc_indicators(df, method='simple_window', window=20):
    indicators = []
    for sym, sub in df.groupby('symbol'):
        close = sub['close'].astype(float).values
        momentum_score = calc_momentum_score(close, method=method, window=window)
        tmp = sub.copy()
        tmp['momentum_score'] = momentum_score
        indicators.append(tmp)
    return pd.concat(indicators)


# ======================
# 3️⃣ 策略执行函数
# ======================
def rank(ctxs: dict[str, ExecContext]):
    scores = {sym: ctx.momentum_score[-1] for sym, ctx in ctxs.items()}
    sorted_scores = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    threshold = pyb.param('rank_threshold')
    top_scores = sorted_scores[:threshold]
    pyb.param('top_symbols', [s[0] for s in top_scores])
    pyb.param('top_scores', top_scores)


# ======================
# 0️⃣ 辅助函数：判断是否触发调仓
# ======================
def should_rebalance(date, freq):
    """
    date: pd.Timestamp 当前日期
    freq: dict, {'type': 'daily'/'weekly'/'monthly', 'value': None/int}
        - daily: 每天调仓
        - weekly: value=0-6 (周一=0，周日=6)
        - monthly: value=1-31
    """
    if freq['type'] == 'daily':
        return True
    elif freq['type'] == 'weekly':
        return pd.Timestamp(date).to_pydatetime().weekday() == freq.get('value', 0)
    elif freq['type'] == 'monthly':
        return pd.Timestamp(date).to_pydatetime().day == freq.get('value', 1)
    else:
        return False


def rotate(ctx: ExecContext):
    # 判断是否到调仓日
    today_ = ctx.date[-1]  # 举例取一个 symbol 的最新日期
    print(f'{ctx.symbol} rotate Checking date for {today_}...')
    # 比如：每月最后一个交易日
    if not should_rebalance(today_, pyb.param('rebalance_freq')):
        return  # 不触发调仓

    # 卖出不在 top N 的持仓
    if ctx.long_pos() and ctx.symbol not in pyb.param('top_symbols'):
        ctx.sell_all_shares()

    # 买入 top N
    elif ctx.symbol in pyb.param('top_symbols') and not ctx.long_pos():
        target_size = pyb.param('target_size')
        ctx.buy_shares = ctx.calc_target_shares(target_size)
        ctx.score = ctx.momentum_score[-1]


# ======================
# 4️⃣ 网格搜索参数
# ======================

param_grid = {
    'max_long_positions': [2, 3],
    'momentum_method': ['simple_window', 'log_simple_window', 'linear_window', 'log_r2_window', 'line_log_r2_window'],
    'momentum_window': [10, 20, 30],
    'etf_codes': etf_codes,
    'rebalance_freq': [
        {'type': 'daily'},
        {'type': 'weekly', 'value': 0},   # 每周一
        {'type': 'weekly', 'value': 2},
        {'type': 'weekly', 'value': 3},
        {'type': 'weekly', 'value': 3},   # 每周四
        {'type': 'weekly', 'value': 4},
        {'type': 'monthly', 'value': 1},  # 每月 1 号
        {'type': 'monthly', 'value': 5},
        {'type': 'monthly', 'value': 10},
        {'type': 'monthly', 'value': 15},
        {'type': 'monthly', 'value': 20},
        {'type': 'monthly', 'value': 25},
    ]
}

grid_combinations = list(itertools.product(
    param_grid['max_long_positions'],
    param_grid['momentum_method'],
    param_grid['momentum_window'],
    param_grid['rebalance_freq'],
    param_grid['etf_codes']
))

# ======================
# 5️⃣ 回测网格搜索
# ======================
results = []

for max_pos, method, window, freq, etf_code in grid_combinations:

    try:
        pyb.param('etf_code', etf_code)

        print(f"Backtest: max_pos={max_pos}, method={method}, window={window}, freq={freq}, etf_code={etf_code}")

        df_filtered = filter_data(df, '2019-01-01', '2025-01-01')
        df_mom = calc_indicators(df_filtered, method=method, window=window)

        config = StrategyConfig(
            max_long_positions=max_pos,
            fee_mode=FeeMode.ORDER_PERCENT,
            fee_amount=0.0005,
            subtract_fees=True
        )
        pyb.param('target_size', 1 / config.max_long_positions)
        pyb.param('rank_threshold', max_pos)
        pyb.param('rebalance_freq', freq)  # 设置调仓周期

        strategy = Strategy(df_mom, start_date='2019-01-01', end_date='2025-01-01', config=config)
        strategy.set_before_exec(rank)
        # print("Adding execution...", df_mom['symbol'].unique().tolist())
        # strategy.add_execution(rotate, df_mom['symbol'].unique().tolist())
        print("Adding execution...", etf_code)
        strategy.add_execution(rotate, etf_code)

        result = strategy.backtest(warmup=window)
        final_nav = result.portfolio['market_value'].iloc[-1]

        results.append({
            'max_long_positions': max_pos,
            'momentum_method': method,
            'momentum_window': window,
            'rebalance_freq': freq,
            'final_nav': final_nav,
            'portfolio': result.portfolio,
            'orders': result.orders,
            'metrics': result.metrics
        })
    except:
        continue


# ======================
# 6️⃣ 输出最优参数
# ======================
# pd.DataFrame(results).to_csv('etf_momentum_grid_results.csv', index=False)
best = max(results, key=lambda x: x['final_nav'])
print("===== 最优策略 =====")
print(f"持仓数: {best['max_long_positions']}, 动量方法: {best['momentum_method']}, 窗口: {best['momentum_window']}, 调仓周期: {best['rebalance_freq']}, 最终净值: {best['final_nav']:.2f}")

# 绘制净值曲线
# import plotly.graph_objects as go
# fig = go.Figure()
# fig.add_trace(
#     go.Scatter(
#         x=best['portfolio'].index,
#         y=best['portfolio']['market_value'].values,
#         mode='lines',
#         name='Strategy NAV'
#     )
# )
# fig.show()


import plotly.graph_objects as go

portfolio = best['portfolio'].copy()
orders = best['orders']
portfolio = portfolio.sort_index()

orders.to_csv('etf_momentum_portfolio.csv', index=False)
portfolio.to_csv('etf_momentum_portfolio.csv', index=False)

# 计算累计收益率
portfolio['cum_return'] = portfolio['market_value'] / portfolio['market_value'].iloc[0] - 1

# 计算回撤
portfolio['rolling_max'] = portfolio['market_value'].cummax()
portfolio['drawdown'] = (portfolio['market_value'] - portfolio['rolling_max']) / portfolio['rolling_max']

fig = go.Figure()

# 净值曲线
fig.add_trace(go.Scatter(
    x=portfolio.index,
    y=portfolio['market_value'],
    mode='lines',
    name='Strategy NAV'
))

# 累计收益率
fig.add_trace(go.Scatter(
    x=portfolio.index,
    y=portfolio['cum_return'],
    mode='lines',
    name='Cumulative Return',
    yaxis='y2'
))

# 回撤
fig.add_trace(go.Scatter(
    x=portfolio.index,
    y=portfolio['drawdown'],
    mode='lines',
    name='Drawdown',
    yaxis='y3'
))

# 设置多个 Y 轴
fig.update_layout(
    title='Strategy Performance',
    xaxis=dict(
        title='Date',
        rangeslider=dict(visible=True),  # 显示可拖动时间轴
        rangeselector=dict(               # 快捷选择按钮
            buttons=list([
                dict(count=1, label='1m', step='month', stepmode='backward'),
                dict(count=3, label='3m', step='month', stepmode='backward'),
                dict(count=6, label='6m', step='month', stepmode='backward'),
                dict(count=1, label='1y', step='year', stepmode='backward'),
                dict(step='all')
            ])
        )
    ),
    yaxis=dict(title='NAV'),
    yaxis2=dict(title='Cumulative Return', overlaying='y', side='left', position=0.5, showgrid=False, tickformat=".0%"),
    yaxis3=dict(title='Drawdown', overlaying='y', side='right', position=0.95, showgrid=False, tickformat=".0%")
)

fig.show()
