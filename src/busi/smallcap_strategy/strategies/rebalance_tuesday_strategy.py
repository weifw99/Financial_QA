import backtrader as bt
import datetime
import numpy as np
from busi.smallcap_strategy.utils.momentum_utils import get_momentum
import csv
import pandas as pd


class RebalanceTuesdayStrategy(bt.Strategy):


    params = dict(
        min_mv=10e8,  # 最小市值 10亿，0.2376； 13/14亿 0.2464
        min_profit=0,  # 最小净利润
        min_revenue=1e8,  # 最小营业收入
        rebalance_weekday=2,  # 每周调仓日（0 = 周一数据）周二早上开盘买入
        # 1 0.21
        # 2 0.12
        # 3 0.06
        # 4 0.14
        # 5 0.08
        hold_count_high=5,  # 行情好时持股数（集中）
        hold_count_low=5,  # 行情差时持股数（分散）
        hight_price=100,  # 个股最高限价
        momentum_days=15,  # 动量观察窗口
        momentum_days_short=10,  # 动量观察窗口
        trend_threshold=-0.02,  # 快速熔断阈值（小市值单日下跌5%）
        stop_loss_pct=0.09,  # 个股止损线（跌幅超过6%）
        take_profit_pct=0.5,  # 个股止盈线（涨幅超过50%）
        null_index='etf_SZ511880',  # 空仓期备选 etf
        # smallcap_index=['csi932000', 'sz399101', 'sh000852'],  # 小市值指数列表（中证2000 + 中小综指 + 中证 1000）
        # smallcap_index=[ 'sz399101', 'sh000852'],  # 小市值指数列表（中证2000 + 中小综指 + 中证 1000）
        # smallcap_index=[ 'csi932000', 'sz399101', 'sh000852', 'sh000046', 'sz399005', 'sz399401'],  # 小市值指数列表（中证2000 + 中小综指 + 中证 1000）
        # smallcap_index=[ 'csi932000', 'sh000046', 'sz399005', 'sz399401'],  # 小市值指数列表（中证2000 + 中小综指 + 中证 1000）
        # smallcap_index=[ 'csi932000', 'sz399101', 'sz399005' ],  # 小市值指数列表（中证2000 + 中小综指 + 中证 1000）
        # smallcap_index=[ 'csi932000', 'sz399101', ],  # 小市值指数列表（中证2000 + 中小综指 + 中证 1000）
        # smallcap_index=[ 'csi932000', 'sz399101', ],  # 0.138
        # smallcap_index=['sz399101','sz399649','sz399663','sz399377','sh000046','sz399408','sz399401' ],  # -0.1
        # smallcap_index=['sz399101','sz399649','sz399663','sz399377','sh000046','sz399408', ],  # -0.1
        # smallcap_index=['sz399101','sz399649','sz399663','sz399377','sh000046', ],  # 0.4
        # smallcap_index=['sz399101','sz399649','sz399663','sz399377', ],  # 0.06
        # smallcap_index=['sz399101','sz399649','sz399663', ],  # 0.08
        # smallcap_index=['sz399101','sz399649', ],  # 0.04  'sz399663'有用
        # smallcap_index=['sz399101', ],  # 0.05
        # smallcap_index=['csi932000', ],  # 0.13
        # smallcap_index=['sz399663', ],  # 0.07
        # smallcap_index=['sh000852', ],  # 0.1139
        # smallcap_index=['sh000852','csi932000', 'sz399663' ],  # 0.08
        # smallcap_index=['sh000852','csi932000', 'sz399663','sz399101', ],  #0.1287
        # smallcap_index=['csi932000', 'sz399663', ],  # 0.1381
        # smallcap_index=['csi932000', 'sz399101', 'sz399005'], # 0.1381
        # smallcap_index=['BK1158'], # 到 7 月 4 号， 0.2376
        # smallcap_index=['csi932000', 'sz399101', 'BK1158'], # 到 7 月 4 号， 0.2376  （全部股票）
        # smallcap_index=['csi932000', 'sz399101', ], # 到 7 月 4 号， 0.2032 （全部股票）
        # smallcap_index=['csi932000', 'sz399101', 'sz399005'], # 到 7 月 4 号， 0.2032 （全部股票）
        # smallcap_index=['csi932000', 'sz399101', 'BK1158'], # 到 7 月 4 号， 0.2376 (zz1000/zz2000/微盘股)
        # smallcap_index=['csi932000', 'sz399101'], # 到 7 月 4 号， 0.2028 中小综指-399101成分股 20亿限制

        # smallcap_index=['csi932000', 'sz399101', 'BK1158', 'sz399005', 'sz399008'], # 0.3847
        # smallcap_index=['csi932000', 'sz399101', 'BK1158', 'sz399005','sz399401'], # 0.3989
        # smallcap_index=['csi932000', 'sz399101', 'BK1158', 'sz399401'], # 0.4031
        # smallcap_index=['csi932000', 'sz399101', 'BK1158', 'sz399008'], # 0.3654
        # smallcap_index=['csi932000', 'sz399101', 'BK1158'], # 0.40
        # smallcap_index=['BK1158'], # 0.46
        # smallcap_index=['sz399101','BK1158'], # 0.50
        smallcap_index=['csi932000', 'BK1158'], # 0.53
        # smallcap_index=['csi932000', 'sz399101', 'BK1158'], # 0.53
        # smallcap_weight=[1, 1.1, 1.2], #
        # smallcap_weight=[1, 1], # 1.6687
        smallcap_weight=[0.9, 1], # 1.6716
        # smallcap_weight=[0.8, 1.2], # 1.6716
        # smallcap_weight=[0.7, 1.3], # 1.6395
        # smallcap_weight=[0.5, 1.5],  # 1.4806
        # smallcap_index=['sz399101', 'BK1158'], #
        # smallcap_index=['csi932000', 'sz399101', 'BK1158', 'sz399005','sz399401', 'sz399008'], # 0.3728
        # smallcap_index=[ 'sz399101', 'BK1158', 'sz399005','sz399401', 'sz399008'], # 0.3339
        # smallcap_index=['csi932000', 'sz399101', 'BK1158', 'sz399005','sz399401','sh000046'],
        # smallcap_index=['csi932000', 'sz399101', 'BK1158'],  # 到 7 月 4 号， 0.2028 中小综指-399101成分股 20亿限制

        # smallcap_index=['csi932000', 'sz399101','sz399005'],  # 到 7 月 4 号， 0.2028 中小综指-399101成分股 20亿限制
        # smallcap_index=['sz399005', 'BK1158'], # 到 7 月 4 号，0.2376 全部
        # smallcap_index=['sz399005', 'BK1158'], # 到 7 月 4 号，0.1727 sz399005
        # smallcap_index=['sz399005', 'sz399101'], # 到 7 月 4 号，0.129 sz399005
        # smallcap_index=['sz399005', 'csi932000'], # 到 7 月 4 号，0.1616 sz399005
        # smallcap_index=['csi932000', 'sz399101', 'BK1158'], # 到 7 月 4 号， 0.1727 sz399005
        # smallcap_index=['sz399101', 'sh000852', 'sh000046', 'sz399005', 'sz399401'], # 到 7 月 4 号， 0.1657 sz399005
        # smallcap_index=[   'sh000852','sz399004','sh000905', 'sh000991'], # 到 7 月 4 号， 0.1727 sz399005
        # smallcap_index=[  'sz399004', 'sz399005', 'sz399006',], # 到 7 月 4 号， 0.1727 sz399005

        # 399101,中小综指
        # 399008,中小300
        # 399401,中小盘
        # 399602,中小成长
        # 399005,中小100
        # 000046,上证中小
        # [ 'sz399649','sz399663','sz399377','sh000046','sz399408','sz399401' ]
        # sz399649, 中小红利  sz399663,中小低波 sz399377,小盘价值 sh000046,上证中小 sz399408,小盘低波 sz399401,中小盘

        # 'csi932000',
        # 'sz399101',
        # 'sz399005',
        # 'sh000046',
        # 'sz399401'

        # smallcap_index=[ 'csi932000', 'sz399005', 'sz399401'],  # 小市值指数列表（中证2000 + 中小综指 + 中证 1000）
        # large_indices=['sh.000300', 'etf_SH159919', 'sh.000016', 'etf_SZ510050', 'etf_SZ510880', 'sh000905']
        # large_indices=['sh.000300', 'etf_SH159919', 'sh.000016', 'etf_SZ510050', 'sh000905']
        large_indices=['sh.000300', 'sh.000016', 'sh.000905']
        # large_indices=['sh.000300', 'etf_SH159919', 'sh.000016', 'etf_SZ510050', 'etf_SZ510880','sh000132' ]
        # '000132','000133','000010','000009'
    )
    def __init__(self):
        self.clear_until = None
        self.do_rebalance_today = False

        self.rebalance_flag = False
        self.to_buy_list = []
        self.rebalance_date = datetime.date(1900, 1, 1)  # ✅ 初始化为一个不可能的历史时间
        # 日志缓存
        self.buy_info = {}  # 每个标的的买入信息 {symbol: {...}}
        self.log_raw_log = []  #
        self.trade_logs = []  # 聚合后的交易
        self.signal_logs = []  # 调仓生成的信号
        self.stop_loss_logs = []  # 止损数据
        self.slope_logs = []  # 斜率数据
        self.close_days = 0 # 空仓的天数
        self.not_mom_3 = 0 # 动量迭出 top3的天数记录
        self.not_mom_1 = 0 # 动量迭出 top1的天数记录

        # 写入 RAW 日志表头
        with open("log_raw.csv", "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow([
                "datetime", "symbol", "type",
                "price", "size", "value", "commission",
                "open_price", "close_price"  # ← 新增两列
            ])

        self.log("初始化策略完成")

    # 日志工具
    def log_raw(self, row):
        with open("log_raw.csv", "a", newline="", encoding="utf-8") as f:
            self.log_raw_log.append(row)
            csv.writer(f).writerow(row)

    def _symbol(self, data):
        return getattr(data, "_name", getattr(data, "_dataname", "unknown"))


    # -----------------------------
    # ✔️  BUY / SELL 日志系统
    # -----------------------------
    def notify_order(self, order):
        dt = self.datas[0].datetime.datetime(0)
        data = order.data
        symbol = self._symbol(data)

        # 当前日期的开盘、收盘价（买卖发生的当天）
        cur_open = data.open[0]
        cur_close = data.close[0]

        if order.status in [order.Submitted, order.Accepted]:
            return

        # =============================
        #        订单成交 Completed
        # =============================
        if order.status == order.Completed:

            # ------------------ BUY ------------------
            if order.isbuy():

                # 写入原始日志
                self.log_raw([
                    dt.strftime('%Y-%m-%d'), symbol, "BUY",
                    order.executed.price,
                    order.executed.size,
                    order.executed.value,
                    order.executed.comm,
                    cur_open,  # ← 新增：买入日开盘
                    cur_close  # ← 新增：买入日收盘
                ])

                # 缓存买入信息
                self.buy_info[symbol] = {
                    "buy_date": dt,
                    "buy_price": order.executed.price,
                    "buy_size": order.executed.size,
                    "buy_comm": order.executed.comm,
                    "buy_open": cur_open,
                    "buy_close": cur_close,
                }

            # ------------------ SELL ------------------
            else:
                self.log_raw([
                    dt.strftime('%Y-%m-%d'), symbol, "SELL",
                    order.executed.price,
                    order.executed.size,
                    order.executed.value,
                    order.executed.comm,
                    cur_open,  # ← 新增：卖出日开盘
                    cur_close  # ← 新增：卖出日收盘
                ])

                # 匹配买单 → 聚合为一行
                if symbol in self.buy_info:
                    info = self.buy_info.pop(symbol)

                    holding_days = (dt.date() - info["buy_date"].date()).days
                    pnl = (order.executed.price - info["buy_price"]) * order.executed.size
                    ret = order.executed.price / info["buy_price"] - 1

                    self.trade_logs.append({
                        "symbol": symbol,
                        "buy_date": info["buy_date"].strftime('%Y-%m-%d'),
                        "buy_price": info["buy_price"],
                        "buy_open_price": info["buy_open"],
                        "buy_close_price": info["buy_close"],
                        "buy_size": info["buy_size"],
                        "sell_date": dt.strftime('%Y-%m-%d'),
                        "sell_price": order.executed.price,
                        "sell_open_price": cur_open,
                        "sell_close_price": cur_close,
                        "sell_size": order.executed.size,
                        "holding_days": holding_days,
                        "pnl": pnl,
                        "return": ret,
                        "buy_comm": info["buy_comm"],
                        "sell_comm": order.executed.comm,
                    })
        elif order.status in [order.Margin, order.Rejected, order.Canceled]:
            reason = "资金不足" if order.status == order.Margin else \
                "被拒绝" if order.status == order.Rejected else "被取消"
            # 写入失败订单
            self.log_raw([dt.strftime('%Y-%m-%d'), symbol, f"REJECT-{reason}",
                          order.price, order.size, None, None, cur_open, cur_close])


    # -----------------------------
    # ✔️  回测结束保存 trade_summary.csv
    # -----------------------------
    def stop(self):
        self.log("策略结束")

        if self.trade_logs:
            df = pd.DataFrame(self.trade_logs).sort_values("buy_date")
            df.to_csv("trade_summary.csv", index=False, encoding="utf-8")
            print("\ntrade_summary.csv saved:")
            print(df.head())

        if self.signal_logs:
            df = pd.DataFrame(self.signal_logs).sort_values("signal_date")
            df.to_csv("signal_summary.csv", index=False, encoding="utf-8")
            print("\nsignal_summary.csv saved:")
            print(df.head())

        if self.stop_loss_logs:
            df = pd.DataFrame(self.stop_loss_logs).sort_values("date")
            df.to_csv("stop_loss_summary.csv", index=False, encoding="utf-8")
            print("\nstop_loss_summary.csv saved:")
            print(df.head())

        if self.slope_logs:
            df = pd.DataFrame(self.slope_logs).sort_values("date")
            df.to_csv("slope_summary.csv", index=False, encoding="utf-8")
            print("\nslope_summary.csv saved:")
            print(df.head())


    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0)
        print(f"{dt.strftime('%Y-%m-%d')} - {txt}")

    def get_days_since_last_sell(self):
        """
        返回距最近一次卖出操作的天数。
        如果没有卖出记录，返回 None。
        """
        last_sell_date = None

        for row in self.log_raw_log:
            dt, symbol, side = row[0], row[1], row[2]
            if side == "SELL":
                # 覆盖为最新的 SELL 日期
                last_sell_date = dt

        if last_sell_date is None:
            return None

        # 转换日期
        sell_dt = datetime.datetime.strptime(last_sell_date, "%Y-%m-%d")
        return (datetime.datetime.now() - sell_dt).days

    def next_open(self):
        print('\n\n')

        self.log("next_open")
        dt = self.datas[0].datetime.datetime(0)
        weekday = dt.weekday()
        # dt.weekday() 的返回值含义：
        # 0 → 星期一（Monday）
        # 1 → 星期二（Tuesday）
        # 2 → 星期三（Wednesday）
        # 3 → 星期四（Thursday）
        # 4 → 星期五（Friday）
        # 5 → 星期六（Saturday）
        # 6 → 星期日（Sunday）
        '''
        在 next_open 方法中调用:
        self.close(data) 会以当天的开盘价执行卖出操作
        这是因为启用了 cheat_on_open=True 模式，允许基于当日开盘价进行交易决策
        在 next 方法中调用:
        self.close(data) 会以下一个可用价格（通常是下一周期的开盘价）执行
        '''

        hold_num = len({d for d, pos in self.positions.items() if pos.size > 0})
        if hold_num == 0:
            # self.close_days = self.close_days+1
            self.close_days = self.get_days_since_last_sell()
        else:
            self.close_days = 0

        self.log(f'next_open 账户净值: {self.broker.getvalue()}, 可用资金: {self.broker.getcash()}, 持仓个数:  {hold_num}, 空仓天数: {self.close_days}')

        # 全局熔断，卖出所有
        is_momentum_ok = self.check_momentum_rank(top_k=1, momentum_days=self.p.momentum_days)
        is_momentum_ok_3 = self.check_momentum_rank(top_k=2, momentum_days=self.p.momentum_days)
        is_momentum_ok_ = self.check_momentum_rank_short(top_k=2, momentum_days=self.p.momentum_days_short)
        self.log(f'next_open 检测结果, is_momentum_ok_3：{is_momentum_ok_3}, is_momentum_ok_： {is_momentum_ok_}, is_momentum_ok： {is_momentum_ok}')
        # is_check_trend = self.check_trend_crash()
        is_check_trend = self.check_combo_trend_crash()
        self.log(f'next_open SmallCapStrategy.next stop loss result, is_check_trend：{is_check_trend}, is_momentum_ok： {is_momentum_ok}')

        holding_num = self.get_pos_holding_num()
        max_days = self.get_max_holding_days()
        min_days = self.get_min_holding_days()
        self.log(f'next_open 持仓数：{holding_num},最大持仓天数：{max_days}, 最小持仓天数：{min_days}')
        if hasattr(self, "entry_dates"):
            self.log(self.entry_dates)


        pct_1 = self.smallcap_price_change(days=1)
        pct_2 = self.smallcap_price_change(days=2)
        pct_3 = self.smallcap_price_change(days=3)

        self.log(f"next_open 小市值指数涨跌幅: 1日：{pct_1}, 2日：{pct_2}, 3日：{pct_3}")

        score = self.get_small_mem_return(window_size=6, momentum_days=3)
        slope4 = get_momentum(score[:-1], method='slope', days=5)
        slope = get_momentum(score[1:], method='slope', days=5)
        self.log(f"get_small_mem_return score: {score}, slope: {slope}")
        self.slope_logs.append({
            "date": dt.strftime('%Y-%m-%d'),
            "slope": slope,
            "score": score[-1] if len(score)>0 else 0,
        })

        # score = self.get_small_mem_return(window_size=6, momentum_days=self.p.momentum_days)
        # if not is_momentum_ok_3:
        #     self.not_mom_3 = self.not_mom_3 + 1
        # else:
        #     self.not_mom_3 = 0
        #
        # if not is_momentum_ok:
        #     self.not_mom_1 = self.not_mom_1 + 1
        # else:
        #     self.not_mom_1 = 0

        # if (is_check_trend or not is_momentum_ok) and (not is_momentum_ok_3 or min_days >  1):
        # if (not is_momentum_ok) and (not is_momentum_ok_3 or min_days > 2 ): # 两个条件的回测结果一样
        if (not is_momentum_ok_) and ( ((not is_momentum_ok) and min_days > 2 ) or ( (not is_momentum_ok_3)  )): # 两个条件的回测结果一样
            self.log(f"next_open 触发止损，卖出所有, 最小持仓 {min_days} 天, 检查持仓天数，至少要持仓两天，进一步检查动量的强度")
            # 继续检查动量的强度， 如果跌出 top3，直接清仓

            # if pct_1 <= -0.045 or pct_2 <= -0.06 :
            #     self.log(f"next_open 触发止损，卖出所有, 小市值指数涨跌幅: 1日：{pct_1}, 2日：{pct_2}, 3日：{pct_3}")
            #     self.sell_all()
            #     return
            # if  slope < -0.0012:
            if  slope < -0.0012:
                self.log(f"next_open 触发止损，卖出所有, slope={slope}")
                self.sell_all()
                return

        # if pct_1 <= -0.045 or pct_2 <= -0.06 or pct_3 <= -0.075 :
        if pct_1 <= -0.045 or pct_2 <= -0.06 :
            self.log(f"next_open 触发止损，卖出所有, 小市值指数涨跌幅: 1日：{pct_1}, 2日：{pct_2}, 3日：{pct_3}")
            self.sell_all()
            return

        # if slope4 > slope and (slope4 - slope > 0.01 ):
        # 0.0101, -0.0097
        # if  slope < -0.0097 and (slope4 - slope > 0.015 ):
        #     self.log(f"next_open 触发调仓日，准备先卖后买, slope={slope}")
        #     self.log("next_open 当前持仓如下：")
        #     self.sell_all()
        #     return

        # 个股止盈止损
        self.check_individual_stop()
        # self.check_individual()

        hold_num = len({d for d, pos in self.positions.items() if pos.size > 0})
        if hold_num == 0:
            # self.close_days = self.close_days+1
            close_days = self.get_days_since_last_sell()
            if close_days:
                self.close_days = close_days
            else:
                self.close_days = 0
        else:
            self.close_days = 0
        if (is_momentum_ok) and ( ( weekday == self.p.rebalance_weekday and self.rebalance_date != dt.date() ) or hold_num == 0 ):
        # if is_momentum_ok and ( ( weekday == self.p.rebalance_weekday and self.rebalance_date != dt.date() ) or (hold_num == 0 and self.close_days>3) ):
        # if is_momentum_ok and ( ( weekday == self.p.rebalance_weekday and self.rebalance_date != dt.date() )  ):
            self.rebalance_date = dt.date()
            self.log(f"next_open 触发调仓日，准备先卖后买, weekday={weekday}, hold_num={hold_num}, close_days={self.close_days}")
            self.log("next_open 当前持仓如下：")
            self.print_positions()

            if not self.validate_index_data():
                self.log("next_open ⚠️ 指数数据不足，跳过调仓")
                return

            # if self.check_stop_conditions(dt):
            #     return

            # hold_num = self.adjust_stock_num_bt()
            # print(f"✅ 本轮建议持股数量为: {hold_num}")

            candidates = self.filter_stocks()

            # is_momentum_ok = self.check_momentum_rank(top_k=1)
            # hold_num = self.p.hold_count_high if is_momentum_ok else self.p.hold_count_low

            to_hold = set(candidates[:self.p.hold_count_high])
            self.log(f"next_open 待持仓：{[d._name for d in to_hold]}")
            current_hold = {d for d, pos in self.positions.items() if pos.size > 0}

            to_sell = current_hold - to_hold
            to_buy = to_hold - current_hold
            self.log(f"next_open to_sell：{[d._name for d in to_sell]}")
            self.log(f"next_open to_buy：{[d._name for d in to_buy]}")

            self.signal_logs.append({
                "signal_date": dt.date().strftime('%Y-%m-%d'),
                "to_sell": [d._name for d in to_sell],
                "to_buy": [d._name for d in to_buy],
            })

            self.to_buy_list=sorted(list(to_buy))

            for d in to_sell:
                self.log(f"next_open 💸 清仓：{d._name}")
                self.close(d)  # 以开盘价卖出
                # self.sell(d, price=d.close[0]) # 以收盘价卖出
                if hasattr(self, "entry_dates"):
                    if d._name in self.entry_dates:
                        self.entry_dates.pop(d._name)

            self.log(f"next_open ✅ 待买入：{self.to_buy_list}")

            self.rebalance_flag = True
        # 原来 next 方法中的逻辑，一到 next_open中， 执行购买逻辑可以使用当天 open价格，在 next buy 中，使用下一周期的开盘价
        if self.rebalance_flag and self.to_buy_list:
            self.rebalance_flag = False

            total_value = self.broker.getvalue()
            total_cash = self.broker.getcash()
            total_per_stock = total_value*0.99 / max(len(to_hold), 1)
            cash_per_stock = total_cash*0.99 / max(len(self.to_buy_list), 1)

            self.log(f"next 📥 开始买入，账户现金: {total_cash:.2f}")

            # 获取持仓大小

            for d in to_hold:
                price = d.open[0]
                if price is None or np.isnan(price) or price <= 0:
                    continue
                size = int(total_per_stock // price)
                size = (size // 100) * 100

                pos = self.getposition(d)
                if pos.size > 0:
                    self.log(f"next 📈 持仓：{d._name} size={pos.size}")
                    add_size = size - pos.size
                else:
                    add_size = size
                self.log(f"next 📥 准备买入：{d._name} size={add_size} total_per_stock: {total_per_stock}, price: {price}, mv: {d.mv[0]}")
                if add_size >= 100:
                    self.log(f"next 📥 买入：{d._name} size={add_size}")
                    self.buy(d, size=add_size)
                    if hasattr(self, "entry_dates"):
                        self.entry_dates[d._name] = self.datas[0].datetime.date(0)
                else:
                    self.log(f"next ⚠️ 资金不足，跳过买入：{d._name} size={add_size}")

            # for d in self.to_buy_list:
            #     price = d.open[0]
            #     if price is None or np.isnan(price) or price <= 0:
            #         continue
            #     size = int(cash_per_stock // price)
            #     size = (size // 100) * 100
            #     self.log(f"next 📥 准备买入：{d._name} size={size} cash_per_stock: {cash_per_stock}, price: {price}, mv: {d.mv[0]}")
            #     if size >= 100:
            #         self.log(f"next 📥 买入：{d._name} size={size}")
            #         self.buy(d, size=size)
            #         if hasattr(self, "entry_dates"):
            #             self.entry_dates[d._name] = self.datas[0].datetime.date(0)
            #     else:
            #         self.log(f"next ⚠️ 资金不足，跳过买入：{d._name} size={size}")

            self.to_buy_list = []

    def next(self):
        print('\n\n')

        self.log("next")

        # 个股止盈止损
        # self.check_individual_stop()

        # is_momentum_ok = self.check_momentum_rank(top_k=1)
        # if self.rebalance_flag and self.to_buy_list:
        #     self.rebalance_flag = False
        #
        #     total_cash = self.broker.getcash()
        #     cash_per_stock = total_cash / max(len(self.to_buy_list), 1)
        #
        #     self.log(f"next 📥 开始买入，账户现金: {total_cash:.2f}")
        #
        #     for d in self.to_buy_list:
        #         price = d.close[0]
        #         if price is None or np.isnan(price) or price <= 0:
        #             continue
        #         size = int(cash_per_stock // price)
        #         size = (size // 100) * 100
        #         self.log(f"next 📥 准备买入：{d._name} size={size} cash_per_stock: {cash_per_stock}, price: {price}, mv: {d.mv[0]}")
        #         if size >= 100:
        #             self.log(f"next 📥 买入：{d._name} size={size}")
        #             self.buy(d, size=size)
        #             if hasattr(self, "entry_dates"):
        #                 self.entry_dates[d._name] = self.datas[0].datetime.date(0)
        #         else:
        #             self.log(f"next ⚠️ 资金不足，跳过买入：{d._name} size={size}")
        #
        #     self.to_buy_list = []
        self.log("next，持仓如下：")
        self.print_positions()


    def check_stop_conditions(self, dt):
        # if self.check_trend_crash():
        if self.check_combo_trend_crash():
            print(f"🚨 {dt.date()} 触发趋势止损")
            self.sell_all()
            self.clear_until = dt.date() + datetime.timedelta(days=7)
            self.is_cleared = True
            return True

        if not self.check_momentum_rank(top_k=1, momentum_days=self.p.momentum_days):
            print(f"⚠️ {dt.date()} 动量止损触发")
            self.sell_all()
            self.clear_until = dt.date() + datetime.timedelta(days=7)
            self.is_cleared = True
            return True

        self.is_cleared = False
        return False

    def check_individual_stop(self):
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size <= 0:
                continue

            hold_num = self.get_holding_days( data)
            # 当天不可以卖出，当天买入的股票算持有一天，第二天才能卖
            if hold_num < 2:
                continue

            buy_price = pos.price
            current_price = data.open[0]

            if np.isnan(current_price) or current_price == 0:
                continue

            change_pct = (current_price - buy_price) / buy_price

            if change_pct >= self.p.take_profit_pct:
                print(f"✅ 止盈触发：{data._name} 涨幅 {change_pct:.2%}")
                self.close(data)
                self.stop_loss_logs.append({
                    "symbol": data._name,
                    "date": data.datetime.date(0).strftime('%Y-%m-%d'),
                    "pos_size": pos.size,
                    "hold_num": hold_num,
                    "action_type": 'stop_profit',
                })
                if hasattr(self, "entry_dates"):
                    if data._name in self.entry_dates:
                        self.entry_dates.pop(data._name)
                continue

            if change_pct <= -self.p.stop_loss_pct:
                print(f"⛔ 止损触发：{data._name} 跌幅 {change_pct:.2%}")
                self.close(data)
                self.stop_loss_logs.append({
                    "symbol": data._name,
                    "date": data.datetime.date(0).strftime('%Y-%m-%d'),
                    "pos_size": pos.size,
                    "hold_num": hold_num,
                    "action_type": 'stop_loss',
                })
                if hasattr(self, "entry_dates"):
                    if data._name in self.entry_dates:
                        self.entry_dates.pop(data._name)

    def check_individual(self):
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size <= 0:
                continue

            buy_price = pos.price
            current_price = data.open[0]

            if np.isnan(current_price) or current_price == 0:
                continue
            hold_num = self.get_holding_days(data)
            # 当天不可以卖出，当天买入的股票算持有一天，第二天才能卖
            if hold_num < 2:
                continue
            if hold_num > 20:

                change_pct = (current_price - buy_price) / buy_price

                if change_pct >= 0.08:
                    print(f"✅ 止盈触发：{data._name} 涨幅 {change_pct:.2%}")
                    self.close(data)
                    self.stop_loss_logs.append({
                        "symbol": data._name,
                        "date": data.datetime.date(0).strftime('%Y-%m-%d'),
                        "pos_size": pos.size,
                        "hold_num": hold_num,
                        "action_type": 'stop_profit',
                    })
                    if hasattr(self, "entry_dates"):
                        if data._name in self.entry_dates:
                            self.entry_dates.pop(data._name)
                    continue

                # if change_pct <= -self.p.stop_loss_pct:
                #     print(f"⛔ 止损触发：{data._name} 跌幅 {change_pct:.2%}")
                #     self.close(data)
                #     self.stop_loss_logs.append({
                #         "symbol": data._name,
                #         "date": data.datetime.date(0).strftime('%Y-%m-%d'),
                #         "pos_size": pos.size,
                #         "hold_num": hold_num,
                #         "action_type": 'stop_loss',
                #     })
                #     if hasattr(self, "entry_dates"):
                #         if data._name in self.entry_dates:
                #             self.entry_dates.pop(data._name)


    def validate_index_data(self):
        names = self.p.smallcap_index + self.p.large_indices
        for name in names:
            d = self.getdatabyname(name)
            if len(d) < self.p.momentum_days + 1 or np.isnan(d.close[0]):
                return False
        return True

    def get_index_return(self, name, days):
        try:
            d = self.getdatabyname(name)
        except Exception as e:
            print(f"⚠️ 指数 {name} 获取失败: {e}")
            return -999

        if len(d) < days:
            return -999

        prices = d.close.get(size=days + 1)
        if prices is None or len(prices) < days:
            return -999

        if np.any(np.isnan(prices)) or prices[-1] == 0:
            return -999
        prices = prices[:-1]  # 去掉最后一天 当天的 close 价格应该不可见
        print('get_index_return:' , name, prices)
        momentum_log = get_momentum(prices, method='log', days=days)
        momentum_slope = get_momentum(prices, method='return', days=days)
        # 组合方式（例如加权平均）
        combo_score = 0.5 * momentum_log + 0.5 * momentum_slope
        return combo_score

    def get_small_mem_return(self, window_size=5, momentum_days=15):

        scores = []
        for name in self.p.smallcap_index:
            d = self.getdatabyname(name)
            if len(d) < momentum_days:
                continue
            prices = d.close.get(size=momentum_days + window_size)
            if prices is None or len(prices) < momentum_days + window_size:
                continue
            if np.any(np.isnan(prices)) or prices[-1] == 0:
                continue

            mems = []
            prices = prices[:-1]  # 去掉最后一天 当天的 close 价格应该不可见
            print('get_small_mem_return:' , name, prices)
            for i in range(window_size):
                prices1 = prices[i:momentum_days+i]
                # print('get_index_return:', i, name, prices1)
                momentum_log = get_momentum(prices1, method='log', days=momentum_days)
                momentum_slope = get_momentum(prices1, method='return', days=momentum_days)
                # 组合方式（例如加权平均）
                combo_score = 0.5 * momentum_log + 0.5 * momentum_slope
                mems.append(combo_score)
            if len(mems) > 0:
                scores.append(mems)
        # print(f'📊 小市值动量get_small_mem_return: {scores} ')

        if len(scores) > 0:
            # return np.mean(scores, axis=0)

            # 转成 numpy 并匹配长度
            arrays = [np.array(a, dtype=float) for a in scores]

            length_set = {len(a) for a in arrays}
            if len(length_set) != 1:
                raise ValueError("所有数组长度必须一致")

            # 加权相加
            weighted_sum = np.zeros_like(arrays[0])
            for arr, w in zip(arrays, self.p.smallcap_weight):
                weighted_sum += arr * w

            # 求均值（对加权后的 N 组求平均）
            result = weighted_sum / len(scores)
            return result
        return []




    def get_combined_smallcap_momentum(self, momentum_days=15):
        scores = [self.get_index_return(name, momentum_days) for name in self.p.smallcap_index]
        valid_scores = [s*w for s, w in zip(scores, self.p.smallcap_weight) if s > -999]
        print(f'📊 小市值动量scores: {scores}, valid_scores:{valid_scores}, ✅ 合并动量: {np.mean(valid_scores)}')
        # 倒序排序并取前2个元素
        # top2_scores = sorted(valid_scores, reverse=True)[:3]
        # return np.max(top2_scores) if top2_scores else -999
        # smallcap_weight
        return np.mean(valid_scores)
        # return np.sum(top2_scores) if top2_scores else -999

    def check_recent_recovery(self):
        # momentum_days = int(self.p.momentum_days_short/3)
        # momentum_days = self.p.momentum_days
        momentum_days = 10
        recovery_scores = []
        recovery_slopes = []
        for i in range(4):
            day_scores = []
            day_slopes = []
            for name in self.p.smallcap_index:
                d = self.getdatabyname(name)
                if len(d) < momentum_days + i + 1:
                    return False
                prices = d.close.get(size=momentum_days + 1 + i)
                prices = prices[:-1]
                print('check_recent_recovery:', i , name, prices)

                if np.any(np.isnan(prices)):
                    return False
                # 修改切片操作，确保获取的数据长度为 momentum_days
                if i == 0:
                    # 当 i=0 时，获取最后 momentum_days 个数据点
                    selected_prices = prices[-(momentum_days):]
                else:
                    # 当 i>0 时，获取倒数第 i+1 天之前 momentum_days 个数据点
                    selected_prices = prices[-(momentum_days + i):-i]
                print('check_recent_recovery selected_prices:', i, name, selected_prices)
                score = get_momentum(selected_prices, method="log", days=momentum_days)
                day_scores.append(score)
                slope = get_momentum(recovery_scores, method='slope', days=4)
                day_slopes.append(slope)
            day_scores = [s * w for s, w in zip(day_scores, self.p.smallcap_weight)]
            recovery_scores.append(np.mean(day_scores))
            recovery_slopes.append(np.max(day_slopes))
            # recovery_scores.append(np.mean(day_scores))
        print(f'📊 最近几个动量: {recovery_scores}')
        recovery_scores.sort(reverse=True)
        slope = get_momentum(recovery_scores[1:], method='slope', days=4)
        print(f'🚨 趋势动量 slope: {slope}')
        return slope >= 0
        # return recovery_slopes[0] >= 0
        # return (recovery_scores[0] > recovery_scores[1] > recovery_scores[2] > recovery_scores[3]
        #         or (recovery_scores[0] > recovery_scores[1] > recovery_scores[2]
        #             and recovery_scores[0] > recovery_scores[1] > recovery_scores[3]
        #             )
        #         or (recovery_scores[0] > recovery_scores[1] > recovery_scores[3]
        #             and recovery_scores[0] > recovery_scores[2] > recovery_scores[3]
        #             )
        #         )
        # return (recovery_scores[0] > recovery_scores[1] > recovery_scores[2]
        #             and recovery_scores[0] > recovery_scores[1] > recovery_scores[3]
        #             ) or (recovery_scores[0] > recovery_scores[1] > recovery_scores[3]
        #             and recovery_scores[0] > recovery_scores[2] > recovery_scores[3]
        #             )



    # 计算小市值组合指数的最近几天跌幅，求最大值，days=1 ，计算昨日的涨跌幅
    def smallcap_price_change(self, days=3):
        pcts = []
        for name in self.p.smallcap_index:
            try:
                d = self.getdatabyname(name)
            except Exception as e:
                print(f"⚠️ 指数 {name} 获取失败: {e}")
                continue
            if len(d) < days:
                continue
            pct = (d.close[-1] - d.open[-days]) / (d.open[-days] + 0.0001)
            # print(f'📊 {name}  pct: {pct}  open : {d.open.get(size=days + 1)}  close : {d.close.get(size=days + 1)}')
            pcts.append(pct)
            # if days == 1:
            #     pct = (d.close[-1] - d.open[-days]) / d.open[-days]
            #     pcts.append(pct)
            # else:
            #     prices = d.close.get(size=days + 1)
            #     if prices is None or len(prices) < days:
            #         continue
            #     prices = prices[:-1]  # 去掉最后一天 当天的 close 价格应该不可见
            #     pct = (prices[-1] - prices[0]) / prices[0]
            #     pcts.append(pct)
        if len(pcts) > 0:
            return np.min(pcts)
        return 0



    def check_momentum_rank(self, top_k=1, momentum_days=15):
        combo_score = self.get_combined_smallcap_momentum(momentum_days=momentum_days)
        returns = {name: self.get_index_return(name, momentum_days) for name in self.p.large_indices}
        returns['__smallcap_combo__'] = combo_score

        sorted_returns = sorted(returns.items(), key=lambda x: x[1], reverse=True)
        print(f'📊 动量排名: {sorted_returns}')

        in_top_k = '__smallcap_combo__' in [x[0] for x in sorted_returns[:top_k]]
        is_recovering = self.check_recent_recovery()

        # if not in_top_k and not is_recovering :
        if not in_top_k :
            print(f"⚠️ 小市值组合动量跌出第一，未回升，且分数不高 -> 止损, in_top_k:{in_top_k}, is_recover:{is_recovering},  combo_score: {combo_score}")
            return False
        return True

    def check_momentum_rank_short(self, top_k=1, momentum_days=15):
        combo_score = self.get_combined_smallcap_momentum(momentum_days=momentum_days)
        returns = {name: self.get_index_return(name, momentum_days) for name in self.p.large_indices}
        returns['__smallcap_combo__'] = combo_score

        sorted_returns = sorted(returns.items(), key=lambda x: x[1], reverse=True)
        print(f'📊 动量排名: {sorted_returns}')

        in_top_k = '__smallcap_combo__' in [x[0] for x in sorted_returns[:top_k]]
        is_recovering = self.check_recent_recovery()

        # if not in_top_k and not is_recovering :
        if not in_top_k :
            print(f"⚠️ 小市值组合动量跌出第一，未回升，且分数不高 -> 止损, in_top_k:{in_top_k}, is_recover:{is_recovering},  combo_score: {combo_score}")
            return False
        return True

    def get_volatility(self, name, days=10):
        try:
            d = self.getdatabyname(name)
            if len(d) < days + 1:
                return 0
            close = np.array(d.close.get(size=days + 1))
            ret = np.diff(np.log(close))
            return np.std(ret) * np.sqrt(252)
        except:
            return 0

    def check_trend_crash(self):
        try:
            d = self.getdatabyname(self.p.smallcap_index[0])
        except Exception as e:
            print(f"⚠️ 获取指数数据失败: {e}")
            return False

        if len(d) < 4:
            print("⚠️ 指数数据不足4天")
            return False

        close = np.array(d.close.get(size=4))
        open_ = np.array(d.open.get(size=4))
        if np.any(np.isnan(close)) or np.any(np.isnan(open_)):
            print("⚠️ 有缺失的价格数据")
            return False

        daily_return = close / open_ - 1
        crash_days = np.sum(daily_return < -0.03)
        avg_return = daily_return.mean()
        vol = self.get_volatility(self.p.smallcap_index[0], days=10)

        print(f'📉 全局熔断判断：3日跌幅={daily_return}, avg={avg_return:.2%}, vol={vol:.2%}')

        if (crash_days >= 2 or avg_return < -0.04) and vol < 0.2:
            print("🚨 触发更稳健的趋势熔断机制")
            return True

        return False


    def check_combo_trend_crash(self):
        """
        多个小市值指数组合的趋势判断：
        若过去3天内，平均跌幅超阈值，或波动率极低+连续下跌，触发止损。
        """
        indices = self.p.smallcap_index  # 多个小市值指数列表，如 ['csi932000', 'sz399101', 'custom_microcap']

        close_mat = []
        open_mat = []

        for name in indices:
            try:
                d = self.getdatabyname(name)
                if len(d) < 4:
                    print(f"⚠️ 指数 {name} 数据不足4天")
                    return False
                close = np.array(d.close.get(size=4))
                open_ = np.array(d.open.get(size=4))
                if np.any(np.isnan(close)) or np.any(np.isnan(open_)):
                    print(f"⚠️ 指数 {name} 存在缺失值")
                    return False
                close_mat.append(close)
                open_mat.append(open_)
            except Exception as e:
                print(f"⚠️ 获取指数 {name} 数据失败: {e}")
                return False

        close_avg = np.mean(close_mat, axis=0)
        open_avg = np.mean(open_mat, axis=0)
        daily_return = close_avg / open_avg - 1

        crash_days = np.sum(daily_return < -0.025)
        avg_return = daily_return.mean()
        vol = np.std(np.diff(np.log(close_avg))) * np.sqrt(252)

        print(f'📉 组合趋势止损判断：3日组合涨跌={daily_return}, 平均={avg_return:.2%}, 波动率={vol:.2%}')

        if (crash_days >= 2 or avg_return < -0.03) and vol < 0.2:
            # 最近 3 天至少 2 天跌超 2.5%，或者平均跌超 3%。且波动率较低。
            print("🚨 触发组合小市值指数的趋势熔断机制")
            return True

        return False


    def compute_correlation_beta1(self, stock_data, index_data, window=20):
        """
        计算相关系数与回归斜率
        参数：
            stock_data: backtrader 的 lines 对象
            index_data: backtrader 的 lines 对象
            window: 回看窗口期
        返回：
            corr: 相关系数
            beta: 回归斜率
        """
        import numpy as np
        from sklearn.linear_model import LinearRegression
        try:
            if len(stock_data) < window + 1 or len(index_data) < window + 1:
                return np.nan, np.nan

            stock_close = np.array(stock_data.close.get(size=window + 1))
            index_close = np.array(index_data.close.get(size=window + 1))

            if np.any(np.isnan(stock_close)) or np.any(np.isnan(index_close)):
                return np.nan, np.nan

            stock_ret = np.diff(np.log(stock_close))
            index_ret = np.diff(np.log(index_close))

            # 相关系数
            corr = np.corrcoef(stock_ret, index_ret)[0, 1]

            # β 回归斜率
            model = LinearRegression()
            model.fit(index_ret.reshape(-1, 1), stock_ret)
            beta = model.coef_[0]

            return corr, beta
        except Exception as e:
            print(f"⚠️ 计算相关性失败: {e}")
            return np.nan, np.nan

    def compute_correlation_beta(self, stock_data, index_data, window=20):
        """
        计算相关系数与回归斜率（β）更稳健版本
        """
        import numpy as np
        from sklearn.linear_model import LinearRegression
        try:
            if len(stock_data) < window + 1 or len(index_data) < window + 1:
                return np.nan, np.nan

            stock_close = np.array(stock_data.close.get(size=window + 1))
            index_close = np.array(index_data.close.get(size=window + 1))

            if np.any(stock_close <= 0):
                print(f"⚠️ 股票收盘价含非正数: {stock_data._name}, {stock_close}")
            # 去除 <= 0 的收盘价
            if np.any(stock_close <= 0) or np.any(index_close <= 0):
                return np.nan, np.nan

            # 计算对数收益率
            stock_ret = np.diff(np.log(stock_close))
            index_ret = np.diff(np.log(index_close))

            # 筛除任何 NaN / inf
            mask = (~np.isnan(stock_ret) & ~np.isnan(index_ret) &
                    ~np.isinf(stock_ret) & ~np.isinf(index_ret))
            stock_ret = stock_ret[mask]
            index_ret = index_ret[mask]

            if len(stock_ret) < 5:
                return np.nan, np.nan

            # 相关系数
            corr = np.corrcoef(stock_ret, index_ret)[0, 1]

            # 回归斜率 β
            model = LinearRegression()
            model.fit(index_ret.reshape(-1, 1), stock_ret)
            beta = model.coef_[0]

            return corr, beta
        except Exception as e:
            print(f"⚠️ 计算相关性失败: {e}")
            return np.nan, np.nan

    def filter_stocks(self):
        candidates = []

        # 加在原有财务条件通过后：
        # index_data = self.getdatabyname(self.p.smallcap_index[1])  # 默认第一个指数为基准

        for d in self.datas:
            if d._name in self.p.smallcap_index + self.p.large_indices:
                continue
            try:

                # pubDate	公司发布财报的日期
                # roeAvg	净资产收益率(平均)(%)	归属母公司股东净利润/[(期初归属母公司股东的权益+期末归属母公司股东的权益)/2]*100%
                # statDate	财报统计的季度的最后一天, 比如2017-03-31, 2017-06-30
                # netProfit	净利润(元)
                # MBRevenue	主营营业收入(元)  # 季度可能为 null
                # mv 市值
                # 使用 pd.merge_asof 实现按时间向前填充匹配
                # profit_ttm 归属母公司股东的净利润TTM

                # 获取前一天的数据
                is_st = d.is_st[-1]
                turn = d.turn[-1]
                close = d.close[-1]
                amount = d.amount[-1]

                mv = d.mv[-1]
                lt_mv = d.lt_mv[-1]
                lt_share_rate = d.lt_share_rate[-1]

                # 年度数据
                profit_y = d.profit_y[-1]
                revenue_y = d.revenue_y[-1]
                roeAvg_y = d.roeAvg_y[-1]
                profit_ttm_y = d.profit_ttm_y[-1]

                # 季度数据
                profit_q = d.profit_q[-1]
                revenue_single_q = d.revenue_single_q[-1]  # 季度可能为 null
                roeAvg_q = d.roeAvg_q[-1]
                profit_ttm_q = d.profit_ttm_q[-1]

                if (lt_mv > self.p.min_mv
                        and lt_share_rate >= 0.85
                        and mv > self.p.min_mv
                        and is_st == 0
                        and turn > 1.5
                        and amount > 4000000
                        # and 8 < close < self.p.hight_price# 0.6569
                        # and 6 < close < self.p.hight_price# 0.6223
                        and 5 < close < self.p.hight_price # 6223
                        # and 2 < close < self.p.hight_price
                        # and 10 < close < self.p.hight_price # 6503
                        # 年度数据
                        and profit_y > 0
                        and roeAvg_y > 0
                        and profit_ttm_y > 0
                        and revenue_y > self.p.min_revenue

                        # 季度数据
                        # and profit_q > 0
                        # and roeAvg_q > 0
                        # and profit_ttm_q > 0
                        # and revenue_single_q > self.p.min_revenue
                ):
                    # corr, beta = self.compute_correlation_beta(d, index_data, window=5)
                    # if np.isnan(corr) or np.isnan(beta):
                    #     continue
                    #
                    # print(f"{d._name} corr={corr:.2f}, beta={beta:.2f}")

                    # 设置门槛条件
                    # if corr < 0.3 and beta < 0.5:  #  选取 corr > 0.3 and beta > 0.35:
                    #     continue
                    # if corr < 0.3:
                    #     continue
                    # if (beta < 0.35 ):
                    #     continue
                    # 选取 window=5 csi932000 corr < 0.3: 0.151 # 截止日期 2025-06-24
                    # 选取 window=5 csi932000 corr < 0.3 or (beta < 0.35 or beta > 2) 0.137
                    # 选取 window=5 csi932000 corr < 0.3 and (beta < 0.35 or beta > 2) 0.14
                    # 选取 window=5 csi932000 beta < 0.35 or beta > 2: 0.133
                    # 选取 window=5 csi932000  beta < 0.35 0.122

                    # 选取 window=5 sz399005 corr < 0.3: 0.1616
                    # 选取 window=5 sz399005 corr < 0.3 or (beta < 0.35 or beta > 2) 0.1722
                    # 选取 window=5 sz399005 corr < 0.3 and (beta < 0.35 or beta > 2) 0.1616
                    # 选取 window=5 sz399005 beta < 0.35 or beta > 2: 0.1722
                    # 选取 window=5 sz399005  beta < 0.35  0.1616

                    # short_momentum_days = 7
                    # min_short_momentum = 0.01  # 最小涨幅1%
                    #
                    # prices = d.close.get(size=short_momentum_days + 1)
                    # if prices is not None and len(prices) == short_momentum_days + 1:
                    #     momentum = (prices[-1] - prices[0]) / prices[0]
                    #     if momentum < min_short_momentum:
                    #         print(f"⚠️ 短期动量过滤（选股时过滤“静止股”），股票跳过: {d._name}, 最近5日涨幅: {momentum:.2%}，最近5日价格: {prices}")
                    #         continue  # 静止股票跳过

                    # candidates.append((d, mv))
                    candidates.append((d, lt_mv, mv))
            except:
                print(f"⚠️ 获取股票数据失败: {d._name}")
                continue
        # candidates = sorted(candidates, key=lambda x: x[1])
        # candidates = sorted(candidates, key=lambda x: (x[1], id(x[0])) )
        # candidates = sorted(candidates, key=lambda x: x[2], reverse=False)
        candidates = sorted(candidates, key=lambda x: (x[2], x[1], id(x[0]) ))
        if len(candidates) > 0:
            print("filter_stocks len：", len(candidates), f'{candidates[0][0]._name} mv min: ', candidates[0][1],
                  f'{candidates[-1][0]._name} mv max: ', candidates[-1][1])
        else:
            print("filter_stocks len：", len(candidates))
        return [x[0] for x in candidates]

    def sell_all(self):
        self.log('💰 清仓 - sell_all')
        for data, pos in self.positions.items():
            if pos.size > 0:
                self.log(f'💰 清仓 - sell_all - code: {data._name}, size: {pos.size}')
                self.close(data)

                self.stop_loss_logs.append({
                    "symbol": data._name,
                    "date": data.datetime.date(0).strftime('%Y-%m-%d'),
                    "pos_size": pos.size,
                    "action_type": 'sell_all',
                })

        self.entry_dates = {}


    def adjust_stock_num_bt(self):
        """
        基于中小综指的 MA 差值，动态调整持股数。
        原始逻辑保持一致：
            - diff >= 500 → 3
            - 200 <= diff < 500 → 3
            - -200 <= diff < 200 → 4
            - -500 <= diff < -200 → 5
            - diff < -500 → 6
        """
        index_name = 'sz399101'  # 或者根据 self.p.smallcap_index[0]
        ma_para = 10

        try:
            d = self.getdatabyname(index_name)
        except Exception as e:
            print(f"⚠️ 无法获取指数数据 {index_name}: {e}")
            return 4

        if len(d) < ma_para + 1:
            print(f"⚠️ 指数数据不足，返回默认值")
            return 4

        # 计算 MA 均值
        try:
            closes = d.close.get(size=ma_para)
            if len(closes) < ma_para or np.any(np.isnan(closes)):
                return 4
            ma = np.mean(closes)
            close_today = d.close[0]
            diff = close_today - ma
        except Exception as e:
            print(f"⚠️ 计算 MA 差值失败: {e}")
            return 4

        print(f"📊 指数当前价: {close_today:.2f}, MA({ma_para}): {ma:.2f}, 差值: {diff:.2f}")

        # 按原始逻辑返回结果
        if diff >= 500:
            return 5
        elif 200 <= diff < 500:
            return 5
        elif -200 <= diff < 200:
            return 6
        elif -500 <= diff < -200:
            return 8
        else:
            return 10

    def print_positions(self):
        total_value = self.broker.getvalue()
        cash_value = self.broker.getcash()
        self.log(f"\n📊 当前账户总市值: {total_value:,.2f}, cash_value: {cash_value:,.2f}")
        for d in self.datas:
            pos = self.getposition(d)
            if pos.size > 0:
                buy_price = pos.price
                current_price = d.close[0]
                open_price = d.open[0]
                if (current_price/(open_price+0.0001)-1) >= 0.095:
                    self.log(f"{d._name:<12}️ 涨停: {d._name}, 幅度:{current_price/open_price-1}")
                # self.log(f"{d._name:<12} 持仓: {pos.size:>6} 购买价: {buy_price:.2f} 开仓价: {open_price:.2f}, 幅度:{current_price/open_price-1}")
                market_value = pos.size * current_price
                cost = pos.size * buy_price
                profit = market_value - cost
                pnl_pct = 100 * profit / cost if cost else 0
                self.log(f"{d._name:<12} 市值:  {pos.size*current_price} 持仓: {pos.size:>6} 购买价: {buy_price:.2f} 当前价: {current_price:.2f} 盈亏: {profit:.2f} ({pnl_pct:.2f}%), 持仓天数: {self.get_holding_days(d)}")

    def get_holding_days(self, data):
        pos = self.getposition(data)
        if pos.size == 0:
            return 0

        # 用 pos.price 记录的开仓价格，找对应的 bar index
        # 这里简单做：每次开仓，记录 entry_date（必须维护）
        if not hasattr(self, "entry_dates"):
            self.entry_dates = {}
        name = data._name
        if name not in self.entry_dates:
            # 第一次开仓
            self.entry_dates[name] = self.datas[0].datetime.date(0)

        today = self.datas[0].datetime.date(0)
        return (today - self.entry_dates[name]).days

    def get_pos_holding_num(self):
        days = [self.get_holding_days(d) for d in self.datas]
        days = [d for d in days if d > 0]
        return len(days) if days else 0

    def get_max_holding_days(self):
        days = [self.get_holding_days(d) for d in self.datas]
        days = [d for d in days if d > 0]
        return max(days) if days else 0

    def get_min_holding_days(self):
        days = [self.get_holding_days(d) for d in self.datas]
        days = [d for d in days if d > 0]
        return min(days) if days else 0