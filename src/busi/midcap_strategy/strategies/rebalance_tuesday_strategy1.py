import backtrader as bt
import datetime
import numpy as np
from busi.midcap_strategy.utils.momentum_utils import get_momentum


class RebalanceTuesdayStrategy1(bt.Strategy):

    params = dict(
        min_mv=10e8,  # 最小市值 10亿，0.2376； 13/14亿 0.2464
        min_profit=0,  # 最小净利润
        min_revenue=1e8,  # 最小营业收入
        rebalance_weekday=0,  # 每周调仓日（0 = 周一数据）周二早上开盘买入
        # 1 0.21
        # 2 0.12
        # 3 0.06
        # 4 0.14
        # 5 0.08
        hold_count_high=10,  # 行情好时持股数（集中）
        hold_count_low=5,  # 行情差时持股数（分散）
        hight_price=50,  # 个股最高限价
        momentum_days=15,  # 动量观察窗口
        trend_threshold=-0.05,  # 快速熔断阈值（小市值单日下跌5%）
        stop_loss_pct=0.06,  # 个股止损线（跌幅超过6%）
        take_profit_pct=0.5,  # 个股止盈线（涨幅超过50%）
        null_index='etf_SZ511880',  # 空仓期备选 etf
        smallcap_index=['csi932000', 'sz399101', 'BK1158'],  # 到 7 月 4 号， 0.2028 中小综指-399101成分股 20亿限制
        large_indices=['sh.000300', 'etf_SH159919', 'sh.000016', 'etf_SZ510050', 'etf_SZ510880', 'sh000905']
    )

    def __init__(self):
        self.clear_until = None
        self.do_rebalance_today = False

        self.rebalance_flag = False
        self.to_buy_list = []
        self.to_sell_list = []
        self.rebalance_date = datetime.date(1900, 1, 1)  # ✅ 初始化为一个不可能的历史时间
        self.log("初始化策略完成")

    def log(self, txt):
        dt = self.datas[0].datetime.datetime(0)
        print(f"{dt.strftime('%Y-%m-%d')} - {txt}")

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

        self.log(f'next_open 账户净值: {self.broker.getvalue()}, 可用资金: {self.broker.getcash()}, 持仓个数:  {len( {d for d, pos in self.positions.items() if pos.size > 0} )}')
        # 个股止盈止损
        self.check_individual_stop()



        if weekday == self.p.rebalance_weekday and self.rebalance_date != dt.date():
            self.rebalance_date = dt.date()
            self.log("next_open 触发调仓日，准备先卖后买")
            self.log("next_open 当前持仓如下：")
            self.print_positions()

            if not self.validate_index_data():
                self.log("next_open ⚠️ 指数数据不足，跳过调仓")
                return

            # print(f"✅ 本轮建议持股数量为: {hold_num}")

            candidates = self.filter_stocks()

            hold_num = self.p.hold_count_high

            to_hold = set(candidates[:hold_num])
            self.log(f"next_open 待持仓：{to_hold}")
            current_hold = {d for d, pos in self.positions.items() if pos.size > 0}

            to_sell = current_hold - to_hold
            to_buy = to_hold - current_hold
            self.log(f"next_open to_sell：{to_sell}")
            self.log(f"next_open to_buy：{to_buy}")

            self.to_buy_list=list(to_buy)
            self.to_sell_list=list(to_sell)

            self.log(f"next_open ✅ 待卖出：{self.to_sell_list}")
            self.log(f"next_open ✅ 待买入：{self.to_buy_list}")

            for d in self.to_sell_list:
                self.log(f"next_open 💸 清仓：{d._name}")
                # self.sell(d, price=d.close[0]) # 以收盘价卖出
                self.close(d) #
                self.to_sell_list = []

            self.rebalance_flag = True

        # 原来 next 方法中的逻辑，一到 next_open中， 执行购买逻辑可以使用当天 open价格，在 next buy 中，使用下一周期的开盘价
        if self.rebalance_flag and self.to_buy_list:
            self.rebalance_flag = False

            total_value = self.broker.getvalue()
            total_cash = self.broker.getcash()
            total_per_stock = total_value * 0.99 / max(len(to_hold), 1)
            cash_per_stock = total_cash * 0.99 / max(len(self.to_buy_list), 1)

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
                self.log(
                    f"next 📥 准备买入：{d._name} size={add_size} total_per_stock: {total_per_stock}, price: {price}, mv: {d.mv[0]}")
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

        # if self.to_sell_list and len(self.to_sell_list) >0:
        #     for d in self.to_sell_list:
        #         self.log(f"next 💸 清仓：{d._name}")
        #         self.close(d)
        #     self.to_sell_list = []

        self.log("next")
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
        #         else:
        #             self.log(f"next ⚠️ 资金不足，跳过买入：{d._name} size={size}")
        #
        #     self.to_buy_list = []
        self.log("next，持仓如下：")
        self.print_positions()

    def stop(self):
        print('\n\n')

        self.log("策略结束")


    def check_individual_stop(self):
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size <= 0:
                continue

            buy_price = pos.price
            current_price = data.close[0]

            if np.isnan(current_price) or current_price == 0:
                continue

            change_pct = (current_price - buy_price) / buy_price

            # if change_pct >= self.p.take_profit_pct:
            #     print(f"✅ 止盈触发：{data._name} 涨幅 {change_pct:.2%}")
            #     self.close(data)
            #     continue

            if change_pct <= -self.p.stop_loss_pct:
                print(f"⛔ 止损触发：{data._name} 跌幅 {change_pct:.2%}")
                self.close(data)


    def validate_index_data(self):
        names = self.p.smallcap_index + self.p.large_indices
        for name in names:
            d = self.getdatabyname(name)
            if len(d) < self.p.momentum_days + 1 or np.isnan(d.close[0]):
                return False
        return True

    def filter_stocks(self):
        candidates = []

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
                score = d.score[-1]

                if (lt_mv > self.p.min_mv
                        and lt_share_rate >= 0.85
                        and mv > self.p.min_mv
                        and is_st == 0
                        and turn > 1.5
                        and amount > 4000000
                        and 2 < close < self.p.hight_price
                        # 年度数据
                        and profit_y > 0
                        and roeAvg_y > 0
                        and profit_ttm_y > 0
                        and revenue_y > self.p.min_revenue

                        # 季度数据
                        # and profit_q > 0
                        # and roeAvg_q > 0
                        and profit_ttm_q > 0
                        and score > 0
                        # and revenue_single_q > self.p.min_revenue
                ):

                    candidates.append((d, mv, score))
            except:
                print(f"⚠️ 获取股票数据失败: {d._name}")
                continue
        candidates = sorted(candidates, key=lambda x: x[2], reverse=True)
        print('candidates:', candidates)
        if len(candidates) > 0:
            print("filter_stocks len：", len(candidates), f'{candidates[0][0]._name} mv min: ', candidates[0][1],
                  f'{candidates[-1][0]._name} mv max: ', candidates[-1][1])
        else:
            print("filter_stocks len：", len(candidates))
        candidates1 = candidates[:100]
        candidates1 = sorted(candidates1, key=lambda x: x[1], reverse=False)

        return [x[0] for x in candidates1]

    def sell_all(self):
        print('💰 清仓 - sell_all')
        for data, pos in self.positions.items():
            if pos.size > 0:
                self.close(data)

    def print_positions(self):
        total_value = self.broker.getvalue()
        cash_value = self.broker.getcash()
        print(f"\n📊 当前账户总市值: {total_value:,.2f}, cash_value: {cash_value}")
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


