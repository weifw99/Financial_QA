import math

import backtrader as bt
import os
import pandas as pd
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np


class MomentumStrategy1(bt.Strategy):
    """
    动量策略V1
    """
    params = (
        ('top_n', 1),  # 选择前N个ETF
        ('min_momentum', -0.1),  # 最小动量阈值，调整为负值以允许负动量
        ('momentum_params', {
                             # 'simple_window': 5, # 负
                             # 'log_simple_window': 25, # 负
                             # 'log_r2_window': 25, # 0.8
                             'weighted_linear_mom': 25, # 0.8
                             # 'line_log_r2_window': 25, # 负
                             }),  # 动量计算参数
    )

    def __init__(self):
        super().__init__()
        self.etf_positions = {}  # 用于跟踪持仓的字典
        self.data_dict = {}  # 存储数据源的字典
        self.last_weekday = None  # 记录上一个交易日是周几

        self.last_trade_date = None  # 上次交易日期
        
        # 存储所有数据源
        for data in self.datas:
            self.data_dict[data._name] = data

    def log(self, txt, dt=None):
        """记录日志"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')

    def next(self):
        """每个bar调用一次"""
        current_date = self.datas[0].datetime.date(0)
        weekday = current_date.weekday()  # 0=周一, 2=周三, 4=周五

        # 每周三执行交易（避免重复执行）
        if weekday == 2 and self.last_trade_date != current_date:
            self.log("=== 每周三轮动交易触发 ===")
            self.rebalance_etfs()
            self.last_trade_date = current_date

        # 打印当前持仓
        self.print_positions()

    # ------------------------------------------------------
    # 核心轮动逻辑
    # ------------------------------------------------------
    def rebalance_etfs(self):
        """轮动逻辑：卖出非目标ETF，买入动量最强ETF"""
        self.log(f"开始计算动量分数，共 {len(self.data_dict)} 个ETF")
        momentum_scores = {}
        # momentum_scores_short = {}

        # 计算动量
        for name, data in self.data_dict.items():
            score = self.calculate_momentum(data)
            if score is not None and score > self.p.min_momentum:
                momentum_scores[name] = score
                self.log(f"ETF {name}: 动量 {score:.4f}")
            else:
                self.log(f"ETF {name}: 动量无效或低于阈值")

        # 选出动量最高的 top_n
        if not momentum_scores:
            self.log("⚠️ 无有效动量ETF，全部平仓避险")
            self.close_all_positions()
            return

        all_etfs = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)
        self.log(f"所有的ETF动量: { all_etfs }")
        top_etfs = all_etfs[:self.p.top_n]
        # 安全区间过滤：得分在(0, 5]范围内
        # 得分>0：确保正向动量，避免负向趋势
        # 得分<=5：避免动量过高，防止追高风险
        # 风险控制：如果所有ETF都不符合条件，则空仓避险
        # top_etfs = [(etf, score) for etf, score in top_etfs if score > 0 and score <= 5.1 ]
        # top_etfs = [(etf, score) for etf, score in top_etfs if score > -0.01 and score <= 5.1 ]
        top_etfs = [(etf, score) for etf, score in top_etfs if score > -0.01 and score <= 5.5 ]
        # 记录选中的ETF及其动量分数
        self.log(f"选中的ETF数量: {len(top_etfs)}")
        for name, score in top_etfs:
            self.log(f"ETF: {name}, 动量分数: {score:.4f}")

        target_etfs = [etf for etf, score in top_etfs]
        self.log(f"目标ETF: {target_etfs}")

        # -------------------
        # 1️⃣ 卖出非目标ETF
        # -------------------
        for name, pos in self.etf_positions.items():
            if pos > 0 and name not in target_etfs:
                self.log(f"卖出非目标ETF: {name}")
                self.close(self.data_dict[name])
                self.etf_positions[name] = 0

        # -------------------
        # 2️⃣ 买入目标ETF
        # -------------------
        current_value = self.broker.getvalue()
        cash = self.broker.getcash()
        self.log(f"当前账户价值: {current_value:.2f}, cash: {cash}")
        if not target_etfs:
            self.log("⚠️ 无目标ETF，保持空仓")
            return

        value_per_etf = cash / len(target_etfs)
        for name in target_etfs:
            data = self.data_dict[name]
            pos = self.getposition(data)

            if pos.size == 0:
                target_value = current_value * self.p.top_n / len(target_etfs)
                # self.log(f"买入ETF: {name}, 金额: {target_value:.2f}")
                self.log(f"买入ETF{name} 前：现金={self.broker.getcash():.2f}, 总资产={self.broker.getvalue():.2f}, 目标金额={target_value:.2f}, ETF价格={data.close[0]:.2f}")

                self.order_target_value(data, target_value*0.98)
                self.etf_positions[name] = target_value / data.close[0]
            else:
                self.log(f"继续持有: {name}")

    def close_all_positions(self):
        """平掉所有持仓"""
        if not self.etf_positions:
            self.log("当前无持仓")
            return
            
        self.log(f"开始平仓，当前持仓数量: {len(self.etf_positions)}")
        for name, pos in self.etf_positions.items():
            if pos > 0:
                self.log(f"平仓: {name}, 数量: {pos:.2f}")
                self.close(self.data_dict[name])
                self.etf_positions[name] = 0
        self.log("平仓完成")

    def notify_order(self, order):
        self.log(f"订单通知: {order.data._name}, 状态: {order.getstatusname()}")
        """订单状态通知"""
        if order.status in [order.Submitted, order.Accepted]:
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(f"买入完成: {order.data._name}, 价格: {order.executed.price:.2f}, 数量: {order.executed.size:.2f}, 成本: {order.executed.value:.2f}, 佣金: {order.executed.comm:.2f}")
            else:
                self.log(f"卖出完成: {order.data._name}, 价格: {order.executed.price:.2f}, 数量: {order.executed.size:.2f}, 收益: {order.executed.value:.2f}, 佣金: {order.executed.comm:.2f}")

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"订单取消/拒绝: {order.data._name}, 状态: {order.getstatusname()}")

    def notify_trade(self, trade):
        """交易通知"""
        if not trade.isclosed:
            return

        self.log(f"交易完成: {trade.data._name}, 毛利润: {trade.pnl:.2f}, 净利润: {trade.pnlcomm:.2f}")

    def calculate_momentum(self, data):
        """计算动量分数"""
        if not self.p.momentum_params:
            self.log(f"警告：未设置动量参数")
            return None

        # 获取动量计算参数
        params = self.p.momentum_params
        self.log(f"当前动量参数: {params}")

        close = data.close
        t = len(close) - 1  # 当前索引

        # -------------------------
        # 1️⃣ 简单动量
        if 'simple_window' in params:
            window = params['simple_window']
            if len(close) > window:
                momentum = close[0] - close[-window]
                self.log(f"{data._name}: [简单动量] 当前={close[0]:.2f}, {window}日前={close[-window]:.2f}, 动量={momentum:.4f}")
                return momentum

        # -------------------------
        # 2️⃣ 对数动量
        elif 'log_simple_window' in params:
            window = params['log_simple_window']
            if len(close) > window and close[0] > 0 and close[-window] > 0:
                momentum = math.log(close[0] / close[-window])
                self.log(f"{data._name}: [对数动量] 当前={close[0]:.2f}, {window}日前={close[-window]:.2f}, 动量={momentum:.4f}")
                return momentum
        # 线性回归 slope 动量
        elif 'linear_window' in params:
            window = params['linear_window']
            if len(close) > window:
                y = [close[-i] for i in reversed(range(window))]
                x = list(range(window))
                x_mean = sum(x) / window
                y_mean = sum(y) / window
                numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
                denominator = sum((xi - x_mean) ** 2 for xi in x)
                slope = numerator / denominator if denominator != 0 else 0.0
                self.log(f"[线性动量] slope={slope:.6f}, 窗口={window}")
                return slope
        # -------------------------
        # 4️⃣ 新增：加权线性回归动量（Weighted Linear Regression MOM）
        elif 'weighted_linear_mom' in params:
            window = params['weighted_linear_mom']
            if len(close) >= window:
                # 获取最近 window 个收盘价
                y_list = close.get(size=window)  # 返回 numpy.ndarray
                y = np.log(np.array(y_list))  # 对数价格
                n = len(y)
                x = np.arange(n)

                # 权重：最近数据权重更高
                weights = np.linspace(1, 2, n)

                # 加权线性回归
                slope, intercept = np.polyfit(x, y, 1, w=weights)

                # 年化收益率
                annualized_returns = np.exp(slope * 250) - 1

                # 加权 R²
                residuals = y - (slope * x + intercept)
                weighted_residuals = weights * residuals ** 2
                r_squared = 1 - (np.sum(weighted_residuals) / np.sum(weights * (y - np.mean(y)) ** 2))

                window_short = 5
                # score = annualized_returns * r_squared + (close[0] - close[-window_short])/(close[-window_short]+0.001) * r_squared
                score = annualized_returns * r_squared
                self.log(f"{data._name}: [年化收益率] annualized_returns={annualized_returns:.6f}, R²={r_squared:.6f}, window_short涨幅={(close[0] - close[-window_short])/(close[-window_short]+0.001):.6f}")
                self.log(f"{data._name}: [加权线性动量] slope={slope:.6f}, R²={r_squared:.6f}, score={score:.6f}")
                return score

        # -------------------------
        # 5️⃣ 对数回归 R² 动量
        elif 'log_r2_window' in params:
            window = params['log_r2_window']
            if len(close) > window and all(c > 0 for c in close.get(size=window) ):
                y = [math.log(c) for c in close.get(size=window) ]
                x = list(range(window))
                x_mean = sum(x) / window
                y_mean = sum(y) / window
                ss_total = sum((yi - y_mean) ** 2 for yi in y)
                ss_reg = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
                slope = ss_reg / sum((xi - x_mean) ** 2 for xi in x)
                y_hat = [slope * (xi - x_mean) + y_mean for xi in x]
                ss_res = sum((yi - yhi) ** 2 for yi, yhi in zip(y, y_hat))
                r2 = 1 - ss_res / ss_total if ss_total != 0 else 0.0
                self.log(f"{data._name}: [对数R²动量] R²={r2:.6f}, 窗口={window}")
                return r2

        # -------------------------
        # 6️⃣ 线性对数R² + slope 混合评分
        elif 'line_log_r2_window' in params:
            window = params['line_log_r2_window']
            if len(close) > window and all(c > 0 for c in close.get(size=window)):
                y = [math.log(c) for c in close.get(size=window)]
                x = list(range(window))
                x_mean = sum(x) / window
                y_mean = sum(y) / window
                ss_total = sum((yi - y_mean) ** 2 for yi in y)
                ss_reg = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
                slope = ss_reg / sum((xi - x_mean) ** 2 for xi in x)
                y_hat = [slope * (xi - x_mean) + y_mean for xi in x]
                ss_res = sum((yi - yhi) ** 2 for yi, yhi in zip(y, y_hat))
                r2 = 1 - ss_res / ss_total if ss_total != 0 else 0.0
                score = slope * r2
                self.log(f"{data._name}: [线性log R²动量] slope={slope:.6f}, R²={r2:.6f}, score={score:.6f}")
                return score

        else:
            self.log(f"⚠️ 未找到匹配的动量计算方式（当前 params: {params}）")
        return None

    def print_positions(self):
        current_date = self.datas[0].datetime.date(0)
        total_value = self.broker.getvalue()
        cash_value = self.broker.getcash()
        print(f"\n📊 {current_date} 当前账户总市值: {total_value:,.2f}, cash_value: {cash_value}")
        for d in self.datas:
            pos = self.getposition(d)
            if pos.size > 0:
                buy_price = pos.price
                current_price = d.close[0]
                market_value = pos.size * current_price
                cost = pos.size * buy_price
                profit = market_value - cost
                pnl_pct = 100 * profit / cost if cost else 0
                print(f"{d._name:<12} 持仓: {pos.size:>6} 购买价: {buy_price:.2f} 当前价: {current_price:.2f} 盈亏: {profit:.2f} ({pnl_pct:.2f}%)")

        print("\n")




def load_data_from_csv(code, fromdate, todate):
    df = pd.read_csv(f'data/{code}.csv', parse_dates=['date'], index_col='date')
    df = df[(df.index >= fromdate) & (df.index <= todate)]
    data = bt.feeds.PandasData(dataname=df, name=code)
    return data

def load_benchmark_data(code, fromdate, todate):
    df = pd.read_csv(f'data/{code}.csv', parse_dates=['date'], index_col='date')
    df = df[(df.index >= fromdate) & (df.index <= todate)]
    data = bt.feeds.PandasData(dataname=df, name=code)
    return data


def run_backtest(pool_file, topk, rebalance_day, stoploss):
    df = pd.read_csv(pool_file)
    etf_codes = df['基金代码'].tolist()
    start = datetime(2023, 1, 1)
    end = datetime(2025, 1, 1)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(1_000_000)
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

    for code in etf_codes:
        file_path = f'data/{code}.csv'
        if os.path.exists(file_path):
            data = load_data_from_csv(code, start, end)
            cerebro.adddata(data)
        else:
            print(f"[⚠️] 缺少数据文件：{file_path}")

    # 加载沪深300指数作为benchmark
    benchmark_code = '000300.SH'
    benchmark_data = load_benchmark_data(benchmark_code, start, end)
    cerebro.adddata(benchmark_data)

    cerebro.addstrategy(
        MomentumStrategy1,
        etf_list=etf_codes,
        rebalance_day=rebalance_day,
        topk=topk,
        stoploss=stoploss
    )

    print(f"\n🚀 启动回测（Top{topk}，调仓日：{rebalance_day}，止损：{stoploss*100:.1f}%）...\n")
    result = cerebro.run()
    strat = result[0]

    # 结果可视化
    cerebro.plot(style='candlestick', volume=False)

    # 输出分析指标
    portfolio_stats = strat.analyzers.getbyname('pyfolio').get_analysis()
    if portfolio_stats:
        returns = portfolio_stats['returns']
        print("\n✅ 回测完成，回报分析可通过 pyfolio 分析器查看（或后续导出）。")
