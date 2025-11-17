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
        ('take_profit', 0.10),  # 止盈阈值（10%）
        ('stop_loss', 0.03),  # 止损阈值（3%）
    )

    def __init__(self):
        super().__init__()
        self.etf_positions = {}  # 用于跟踪持仓的字典
        self.etf_stops = set()  # 用于跟踪止损的标的
        self.etf_takes = set()  # 用于跟踪止盈的标的
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

        # 每日检查止盈止损
        self.check_stop_take_profit()

        # 每周三执行交易（避免重复执行）
        if weekday == 2 and self.last_trade_date != current_date:
            self.log("=== 每周三轮动交易触发 ===")
            self.rebalance_etfs()
            self.last_trade_date = current_date

        # 打印当前持仓
        self.print_positions()

    # ------------------------------------------------------
    # 新增止盈止损逻辑
    # ------------------------------------------------------
    def check_stop_take_profit(self):
        """每天检查止盈止损"""
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size <= 0:
                continue

            current_price = data.close[0]
            buy_price = pos.price
            change_pct = (current_price - buy_price) / buy_price

            # 止盈
            if change_pct >= self.p.take_profit:
                self.log(f"📈 达到止盈条件 {data._name}: 当前涨幅 {change_pct*100:.2f}%，执行止盈卖出")
                self.close(data)
                self.etf_positions[data._name] = 0
                self.etf_takes.add(data._name)


            # 止损
            elif change_pct <= -self.p.stop_loss:
                self.log(f"📉 达到止损条件 {data._name}: 当前跌幅 {change_pct*100:.2f}%，执行止损卖出")
                self.close(data)
                self.etf_positions[data._name] = 0
                self.etf_stops.add(data._name)
        # 触发了止盈，并且空仓，需要平衡
        if len(self.etf_takes) >0 :
            self.log(f"⚠️ 触发止盈，开始平衡")
            self.rebalance_etfs()
            self.etf_takes.clear()
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
        top_etfs = [(etf, score) for etf, score in all_etfs if score > -0.01 and score <= 5.5 and etf not in self.etf_stops]
        top_etfs = top_etfs[:self.p.top_n]
        # 安全区间过滤：得分在(0, 5]范围内
        # 得分>0：确保正向动量，避免负向趋势
        # 得分<=5：避免动量过高，防止追高风险
        # 风险控制：如果所有ETF都不符合条件，则空仓避险
        # top_etfs = [(etf, score) for etf, score in top_etfs if score > 0 and score <= 5.1 ]
        # top_etfs = [(etf, score) for etf, score in top_etfs if score > -0.01 and score <= 5.1 ]

        # top_etfs = [(etf, score) for etf, score in top_etfs if score > -0.01 and score <= 5.5 ]
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

        self.etf_stops.clear()

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
        self.etf_stops.clear()
        self.etf_takes.clear()

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
                # annualized_returns = np.exp(slope * 21) - 1

                # 加权 R²
                residuals = y - (slope * x + intercept)
                weighted_residuals = weights * residuals ** 2
                r_squared = 1 - (np.sum(weighted_residuals) / np.sum(weights * (y - np.mean(y)) ** 2))

                window_short = 10
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



# -*- coding: utf-8 -*-
"""
MomentumStrategy V2 — 完整策略文件
加入：
1. 动量短期斜率 = 对“动量序列”做回归
2. 斜率自适应归一化 = slope_z = slope / volatility
3. 使用 ATR 或 return 波动率
4. 自适应阈值 slope_z_threshold

你可以直接在 Backtrader 主程序中 import 使用。
"""

import backtrader as bt
import numpy as np
import pandas as pd
import datetime


class MomentumStrategyV2(bt.Strategy):
    params = dict(
        # ==== 动量相关 ====
        mom_window=25,                   # 长期动量窗口
        min_momentum=0.0,                # 最低动量过滤
        momentum_method='weighted_linear_mom',

        # ==== 短期动量斜率 ====
        slope_filter_window=5,           # 短期动量序列长度
        vol_window=10,                   # 波动率窗口（ATR 或 std）
        slope_z_threshold=-1.0,          # 自适应阈值（越负越严）

        # ==== 轮动 ====
        num_positions=1,
        rebalance_weekday=2,             # 默认周三

        # ==== 止盈止损 ====
        take_profit=0.10,
        stop_loss=0.03,

        debug=True,
    )

    # ------------------------------------------------------
    # 交易与轮动
    # ------------------------------------------------------

    def __init__(self):
        self.data_dict = {d._name: d for d in self.datas}
        self.etf_stops = {name: -1 for name in self.data_dict}

    # ------------------------------------------------------
    # 工具函数
    # ------------------------------------------------------

    def log(self, txt):
        if self.p.debug:
            dt = self.datas[0].datetime.date(0)
            print(f"{dt}: {txt}")

    # ========== 动量计算（用于长期） ==========

    def calculate_momentum_from_array(self, close_array):
        """对一个 close array 计算动量（使用你指定的方法）"""
        arr = np.array(close_array)
        if len(arr) < self.p.mom_window:
            return None

        # 回归 slope + R^2（你原逻辑）
        try:
            # y = arr
            # x = np.arange(len(arr))
            # slope, intercept = np.polyfit(x, y, 1)
            # y_pred = slope * x + intercept
            # ss_res = np.sum((y - y_pred) ** 2)
            # ss_tot = np.sum((y - np.mean(y)) ** 2)
            # r2 = 1 - ss_res / ss_tot
            # return (slope / arr[0]) * r2 * 252

            # 获取最近 window 个收盘价
            y_list = arr  # 返回 numpy.ndarray
            y = np.log(np.array(y_list))  # 对数价格
            n = len(y)
            x = np.arange(n)

            # 权重：最近数据权重更高
            weights = np.linspace(1, 2, n)

            # 加权线性回归
            slope, intercept = np.polyfit(x, y, 1, w=weights)

            # 年化收益率
            annualized_returns = np.exp(slope * 250) - 1
            # annualized_returns = np.exp(slope * 21) - 1

            # 加权 R²
            residuals = y - (slope * x + intercept)
            weighted_residuals = weights * residuals ** 2
            r_squared = 1 - (np.sum(weighted_residuals) / np.sum(weights * (y - np.mean(y)) ** 2))

            window_short = 10
            # score = annualized_returns * r_squared + (close[0] - close[-window_short])/(close[-window_short]+0.001) * r_squared
            score = annualized_returns * r_squared
            self.log(
                f" [年化收益率] annualized_returns={annualized_returns:.6f}, R²={r_squared:.6f}, window_short涨幅={(y_list[-1] - y_list[-window_short]) / (y_list[-window_short] + 0.001):.6f}")
            self.log(f"[加权线性动量] slope={slope:.6f}, R²={r_squared:.6f}, score={score:.6f}")
            return score
        except:
            return None

    def calculate_momentum(self, data):
        closes = data.close.get(size=self.p.mom_window)
        closes1 = np.array(closes, dtype=float)
        print(f'{data._name}: 获取最近 {self.p.mom_window} 个 close 数据： {closes}')
        # print(f'{data._name}: 获取最近 {self.p.mom_window} 个 close 数据： {closes1}')
        if len(closes) < self.p.mom_window:
            return None
        return self.calculate_momentum_from_array(closes)

    # ========== 计算短期动量序列 ==========

    def compute_recent_momentum_series(self, data, window):
        """
        计算最近 window 个时间点对应的动量值序列（用于短期斜率计算）。
        实现要点：
          - 每个动量值基于固定长度 self.p.mom_window 的子序列计算
          - 返回数组按时间从旧 -> 新 排序（便于 polyfit）
        """
        # 需要的最小总长度：mom_window + window - 1
        need = self.p.mom_window + window
        closes = data.close.get(size=need)

        if closes is None or len(closes) < need:
            self.log(
                f"{data._name}: compute_recent_momentum_series 数据不足 need={need}, got={len(closes) if closes is not None else 0}")
            return None

        # 将数组转为 numpy，并按时间从旧到新排序
        arr = np.array(closes, dtype=float)  # get() 旧到新
        print(f'{data._name}: 获取最近 {need} 个 close 数据： {np.array(closes, dtype=float)}， len: {len(closes)}')
        # print(f'{data._name}: 获取最近 {need} 个 close 数据： {arr}， len: {len(arr)}')

        mom_list = []
        for i in range(window ):
            # i=i+1
            sub = arr[i: i + self.p.mom_window]  # 每个子序列长度固定为 mom_window
            print(f"{data._name}: 子序列 {i} = {sub}, len: {len( sub)}")
            if len(sub) < self.p.mom_window:
                self.log(f"{data._name}: 子序列长度不足 i={i}, len(sub)={len(sub)}")
                return None
            m = self.calculate_momentum_from_array(sub)
            if m is None:
                self.log(f"{data._name}: 计算子序列动量失败 i={i}")
                return None
            mom_list.append(m)

        # mom_list 是从最早窗口到最近窗口（旧->新），这正是 compute_momentum_slope 所需要的顺序
        # self.log(f"{data._name}: 短期动量序列 (旧->新) = {mom_list}")
        return np.array(mom_list, dtype=float)


    def compute_momentum_slope(self, momentum_series):
        x = np.arange(len(momentum_series))
        slope, _ = np.polyfit(x, momentum_series, 1)
        return slope

    # ========== 波动率计算（ATR 或 Std） ==========

    def compute_volatility(self, data, window):
        # ATR
        atr_list = []
        for i in range(1, window + 1):
            try:
                h = data.high[-i]
                l = data.low[-i]
                c_prev = data.close[-i - 1]
                tr = max(h - l, abs(h - c_prev), abs(l - c_prev))
                atr_list.append(tr)
            except:
                return None
        atr = np.mean(atr_list)
        return atr

    # ========== 自适应 slope_z（核心） ==========

    def adaptive_momentum_slope_filter(self, data):
        mom_series = self.compute_recent_momentum_series(
            data, self.p.slope_filter_window
        )
        if mom_series is None:
            return None

        print(f"{data._name}: mom_series: {mom_series}")
        slope = self.compute_momentum_slope(mom_series)
        vol = self.compute_volatility(data, self.p.vol_window)

        if vol is None or vol == 0:
            return slope

        slope_z = slope / vol
        return slope_z
        # return slope


    # 止盈止损监控
    def check_take_profit_stop_loss(self):
        tp = self.p.take_profit
        sl = -self.p.stop_loss
        trades_hit_tp = False

        for data in self.datas:
            name = data._name
            if self.getposition(data).size == 0:
                continue

            entry_price = self.getposition(data).price
            current_price = data.close[0]
            change = (current_price - entry_price) / entry_price

            if change >= tp:
                self.log(f"止盈: SELL {name}")
                self.close(data)
                trades_hit_tp = True

            elif change <= sl:
                self.log(f"止损: SELL {name}")
                self.etf_stops[name] = self.datas[0].datetime.date(0)
                self.close(data)

        return trades_hit_tp

    # 轮动执行
    def rebalance_etfs(self):
        self.log("开始轮动评估…")
        scores = {}
        scores1 = {}

        for name, data in self.data_dict.items():
            # ---- 长期动量 ----
            mom_long = self.calculate_momentum(data)

            # ---- 自适应短期动量斜率 slope_z ----
            slope_z = self.adaptive_momentum_slope_filter(data)

            self.log(f"{name}: 长期动量={mom_long}, slope_z={slope_z}")

            if mom_long is None or mom_long <= self.p.min_momentum:
                continue
            # if slope_z is None:
            #     continue
            #
            # if slope_z < self.p.slope_z_threshold:
            #     self.log(
            #         f"🚫 {name}: slope_z={slope_z:.3f} < 阈值 {self.p.slope_z_threshold}，短期动量转弱 → 过滤"
            #     )
            #     continue

            # self.log(f"{name}: 长期动量={mom_long:.4f}, slope_z={slope_z:.3f}")
            scores[name] = mom_long
            # scores1[name] = (mom_long, slope_z, mom_long + slope_z) # 44.8 1.36
            # scores1[name] = (mom_long, slope_z, 2*mom_long + slope_z) # 30.7 2.37
            # scores1[name] = (mom_long, slope_z, 2*mom_long + 1.5*slope_z) # 35.5 #1.8
            # scores1[name] = (mom_long, slope_z, slope_z) #  # 36.27 0.93
            # scores1[name] = (mom_long, slope_z, mom_long + 2*slope_z) #  # 31.61 1.16
            scores1[name] = (mom_long, slope_z, mom_long ) #  # 31.61 0.89

        if not scores:
            self.log("无符合条件标的 → 清仓")
            for data in self.datas:
                self.close(data)
            return

        # 选 top N
        selected_all1 = sorted(scores, key=scores.get, reverse=True)
        self.log(f"标的动量-all: {selected_all1}")
        selected_all = sorted(scores.items(), key=lambda x: x[1], reverse=True)
        selected_all1 = sorted(scores1.items(), key=lambda x: x[1][2], reverse=True)
        self.log(f"标的动量-all: {str(selected_all)}")
        self.log(f"标的动量-all1: {str(selected_all1)}")
        # selected = selected_all[:self.p.num_positions]
        selected = selected_all1[:self.p.num_positions]
        self.log(f"选中标的: {selected}")


        # 添加过滤逻辑


        selected = [x[0] for x in selected]
        # ---- 卖出非目标 ----
        for data in self.datas:
            if data._name not in selected:
                if self.getposition(data).size > 0:
                    self.close(data)

        # ---- 买入目标 ----
        total_value = self.broker.getvalue()*0.98
        target_value = total_value / len(selected)

        for name in selected:
            data = self.data_dict[name]
            cur_pos = self.getposition(data).size

            if cur_pos == 0:
                price = data.close[0]
                size = int(target_value / price / 100)  * 100
                if size > 0:
                    self.log(f"BUY {name}: price={ price},size={size}")
                    self.buy(data, size=size)

    # ------------------------------------------------------
    # next() 主循环
    # ------------------------------------------------------
    def next(self):
        dt = self.datas[0].datetime.date(0)

        # ---- 每日止盈止损 ----
        hit_tp = self.check_take_profit_stop_loss()

        # ---- 触发止盈就立即轮动 ----
        if hit_tp:
            self.rebalance_etfs()
            return

        # ---- 每周 rebalance ----
        if dt.weekday() == self.p.rebalance_weekday:
            self.rebalance_etfs()

        self.print_positions()

    # ------- 订单/成交日志：建议加入 notify_order / notify_trade 来记录成交明细 -------
    def notify_order(self, order):
        # 记录订单状态（下单/成交/取消）
        if order.status in [order.Submitted, order.Accepted]:
            # 下单被接收
            self.log(f"订单 {order.data._name}: 状态 {order.getstatusname()} (提交/接受)")
            return

        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    f"买入成交: {order.data._name}, 价格={order.executed.price:.4f}, 数量={order.executed.size}, 成本={order.executed.value:.2f}, 佣金={order.executed.comm:.2f}")
            else:
                self.log(
                    f"卖出成交: {order.data._name}, 价格={order.executed.price:.4f}, 数量={order.executed.size}, 收益={order.executed.value:.2f}, 佣金={order.executed.comm:.2f}")
        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log(f"订单 {order.data._name}: 被取消/拒绝/保证金不足 状态 {order.getstatusname()}")

    def notify_trade(self, trade):
        if not trade.isclosed:
            return
        self.log(f"交易关闭: {trade.data._name}, 毛利={trade.pnl:.2f}, 净利={trade.pnlcomm:.2f}")

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