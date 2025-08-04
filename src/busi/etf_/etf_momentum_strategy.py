import math

import backtrader as bt
import os
import pandas as pd
import argparse
from datetime import datetime
import matplotlib.pyplot as plt
import numpy as np

from busi.etf_.util_moment import momentum_linear, momentum_simple, momentum_dual, log_momentum_r2, log_momentum_simple, \
    line_log_momentum_r2
from busi.smallcap_strategy.utils.momentum_utils import get_momentum


class MomentumStrategy1(bt.Strategy):
    """
    动量策略V1
    """
    params = (
        ('top_n', 5),  # 选择前N个ETF
        ('min_momentum', -0.1),  # 最小动量阈值，调整为负值以允许负动量
        ('max_position', 0.2),  # 最大持仓比例
        ('momentum_params', {

                             # 'simple_window': 20, # 负


                             # 'log_simple_window': 20, # 负
            # 'linear_window': 10, # 0.8

                             'log_r2_window': 15, # 0.8
                             # 'line_log_r2_window': 20, # 负
                             # 'long_window': 20,
                             # 'short_window': 10,
                             # 'slope_positive_filter': True,
                             }),  # 动量计算参数
    )

    def __init__(self):
        super().__init__()
        self.etf_positions = {}  # 用于跟踪持仓的字典
        self.data_dict = {}  # 存储数据源的字典
        self.last_weekday = None  # 记录上一个交易日是周几
        
        # 存储所有数据源
        for data in self.datas:
            self.data_dict[data._name] = data

    def log(self, txt, dt=None):
        """记录日志"""
        dt = dt or self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()}: {txt}')

    def next(self):
        # 获取当前日期
        current_date = self.datas[0].datetime.date(0)
        current_weekday = current_date.weekday()  # 0-6，0是周一，4是周五
        
        # 如果是周一，进行买入操作
        if current_weekday == 0:
            self.log("=== 周一买入信号 ===")
            self.buy_etfs()
        # 如果是周五，进行平仓操作
        elif current_weekday == 4:
            self.log("=== 周五平仓信号 ===")
            self.close_all_positions()
            
        self.last_weekday = current_weekday

    def buy_etfs(self):
        """买入动量最高的ETF"""
        # 计算所有ETF的动量分数
        momentum_scores = {}
        self.log(f"开始计算ETF动量分数，ETF数量: {len(self.data_dict)}")
        
        for name, data in self.data_dict.items():
            momentum = self.calculate_momentum(data)
            if momentum is not None:
                self.log(f"ETF: {name}, 原始动量分数: {momentum}")
                if momentum > self.p.min_momentum:
                    momentum_scores[name] = momentum
                    self.log(f"ETF: {name}, 有效动量分数: {momentum}")
                else:
                    self.log(f"ETF: {name}, 动量分数低于阈值: {momentum} < {self.p.min_momentum}")
            else:
                self.log(f"ETF: {name}, 动量分数无效")

        # 选择动量最高的N个ETF
        top_etfs = sorted(momentum_scores.items(), key=lambda x: x[1], reverse=True)[:self.p.top_n]
        
        # 记录选中的ETF及其动量分数
        self.log(f"选中的ETF数量: {len(top_etfs)}")
        for name, score in top_etfs:
            self.log(f"ETF: {name}, 动量分数: {score:.4f}")
        
        # 计算每个ETF的目标权重
        total_momentum = sum(score for _, score in top_etfs)
        if total_momentum > 0:
            target_weights = {name: min(score/total_momentum, self.p.max_position) 
                            for name, score in top_etfs}
            self.log(f"总动量分数: {total_momentum:.4f}")
        else:
            target_weights = {}
            self.log("警告：所有ETF的动量分数都为0或无效")

        # 调整持仓
        current_value = self.broker.getvalue()
        self.log(f"当前账户价值: {current_value:.2f}")
        
        # 调整现有持仓
        for name, weight in target_weights.items():
            data = self.data_dict[name]
            target_size = (current_value * weight) / data.close[0]
            
            if name in self.etf_positions:
                # 调整现有持仓
                current_size = self.etf_positions[name]
                if abs(target_size - current_size) > 1e-6:  # 避免微小调整
                    self.log(f"调整持仓: {name}, 目标数量: {target_size:.2f}, 当前数量: {current_size:.2f}, 权重: {weight:.2%}")
                    self.order_target_size(data, target_size)
                    self.etf_positions[name] = target_size
            else:
                # 开新仓
                self.log(f"开新仓: {name}, 数量: {target_size:.2f}, 权重: {weight:.2%}, 价格: {data.close[0]:.2f}")
                self.order_target_size(data, target_size)
                self.etf_positions[name] = target_size

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
        t = len(close) - 1  # 当前索引，不一定用得上

        # 简单动量: 当前收盘价 - N 日前收盘价
        if 'simple_window' in params:
            window = params['simple_window']
            if len(close) > window:
                momentum = close[0] - close[-window]
                self.log(f"[简单动量] 当前={close[0]:.2f}, {window}日前={close[-window]:.2f}, 动量={momentum:.4f}")
                return momentum

        # 对数动量: log(当前/过去N日)
        elif 'log_simple_window' in params:
            window = params['log_simple_window']
            if len(close) > window and close[0] > 0 and close[-window] > 0:
                momentum = math.log(close[0] / close[-window])
                self.log(f"[对数动量] 当前={close[0]:.2f}, {window}日前={close[-window]:.2f}, 动量={momentum:.4f}")
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

        # 对数回归 R² 动量
        elif 'log_r2_window' in params:
            window = params['log_r2_window']
            if len(close) > window and all(c > 0 for c in close.get(size=window)):
                y = [math.log(close[-i]) for i in reversed(range(window))]
                x = list(range(window))
                x_mean = sum(x) / window
                y_mean = sum(y) / window
                ss_total = sum((yi - y_mean) ** 2 for yi in y)
                ss_reg = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
                slope = ss_reg / sum((xi - x_mean) ** 2 for xi in x)
                y_hat = [slope * (xi - x_mean) + y_mean for xi in x]
                ss_res = sum((yi - yhi) ** 2 for yi, yhi in zip(y, y_hat))
                r2 = 1 - ss_res / ss_total if ss_total != 0 else 0.0
                self.log(f"[对数R²动量] R²={r2:.6f}, 窗口={window}")
                return r2

        # 线性对数R² + slope 混合评分
        elif 'line_log_r2_window' in params:
            window = params['line_log_r2_window']
            if len(close) > window and all(c > 0 for c in close.get(size=window)):
                y = [math.log(close[-i]) for i in reversed(range(window))]
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
                self.log(f"[线性log R²动量] slope={slope:.6f}, R²={r2:.6f}, score={score:.6f}")
                return score

        # 双动量：短期动量 - 长期动量
        elif 'short_window' in params and 'long_window' in params:
            sw = params['short_window']
            lw = params['long_window']
            if len(close) > lw:
                short_mom = close[0] - close[-sw]
                long_mom = close[0] - close[-lw]
                dual_mom = short_mom - long_mom

                if 'slope_positive_filter' in params:
                    # 可选：添加线性slope过滤
                    slope_window = sw
                    y = [close[-i] for i in reversed(range(slope_window))]
                    x = list(range(slope_window))
                    x_mean = sum(x) / slope_window
                    y_mean = sum(y) / slope_window
                    numerator = sum((xi - x_mean) * (yi - y_mean) for xi, yi in zip(x, y))
                    denominator = sum((xi - x_mean) ** 2 for xi in x)
                    slope = numerator / denominator if denominator != 0 else 0.0
                    if slope <= 0:
                        self.log(f"[双动量V2] slope<=0，过滤，值为0")
                        return 0.0
                    self.log(f"[双动量V2] short={short_mom:.4f}, long={long_mom:.4f}, 结果={dual_mom:.4f}")
                else:
                    self.log(f"[双动量] short={short_mom:.4f}, long={long_mom:.4f}, 结果={dual_mom:.4f}")

                return dual_mom

        self.log(f"⚠️ 未找到匹配的动量计算方式（当前 params: {params}）")
        return None


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
