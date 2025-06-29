# strategies/smallcap_strategy.py
# 小市值策略主类，使用 notify_timer 控制调仓标志，在 next 中统一调仓逻辑

import backtrader as bt
from datetime import datetime, timedelta
import numpy as np
from busi.smallcap_strategy.utils.momentum_utils import get_momentum


class SmallCapStrategy(bt.Strategy):
    params = dict(
        min_mv=10e8,
        min_profit=0,
        min_revenue=1e8,
        rebalance_weekday=1,  # 周二调仓
        hold_count_high=3,
        hold_count_low=3,
        hight_price=50,
        momentum_days=20,
        trend_threshold=-0.05,
        stop_loss_pct=0.08,  # 个股止损线（跌幅超过8%）
        take_profit_pct=0.5,  # 个股止盈线（涨幅超过50%）
        null_index='etf_SZ511880',
        smallcap_index='csi932000',
        # smallcap_index='csi932000',
        large_indices=['sh.000300', 'etf_SH159919', 'sh.000016', 'etf_SZ510050', 'etf_SZ510880']
    )

    def __init__(self):
        self.clear_until = None
        self.is_cleared = False
        self.do_rebalance_today = False

        self.add_timer(
            when=bt.Timer.SESSION_START,
            weekdays=[self.p.rebalance_weekday],
            weekcarry=True,
            timername='rebalance_timer',
        )

    def notify_timer(self, timer, when, *args, **kwargs):
        if kwargs.get('timername') == 'rebalance_timer':
            dt = self.data0.datetime.date(0)
            print(f"📅 {dt} notify_timer 触发，设置调仓标志")
            self.do_rebalance_today = True

    def next(self):
        """
        主逻辑, 每次都会调用
        """
        dt = self.data0.datetime.datetime(0)
        print('📈 next 执行时间:', self.datetime.datetime(0), '账户净值:', self.broker.getvalue())
        # ✅ 先执行个股止盈止损
        self.check_individual_stop()

        # ✅ 再看是否需要调仓
        if self.do_rebalance_today:
            self.do_rebalance_today = False
            self.handle_rebalance(dt)
            return

        # todo 止损模块应该每天都计算
        # 动量止损
        is_momentum_ok = self.check_momentum_rank()
        is_check_trend = self.check_trend_crash()
        # 快速趋势止损（小市值单日下跌5%） + 动量判断
        print(
            f'SmallCapStrategy.next stop loss result, is_check_trend：{is_check_trend}, is_momentum_ok： {is_momentum_ok}')
        if is_check_trend or not is_momentum_ok:
            self.sell_all()
            # self.sell_all_not_etf()
            self.is_cleared = True
            return

    def check_individual_stop(self):
        """
        检查每只股票是否触发止盈或止损
        """
        for data in self.datas:
            pos = self.getposition(data)
            if pos.size <= 0:
                continue

            buy_price = pos.price
            current_price = data.close[0]

            if np.isnan(current_price) or current_price == 0:
                continue

            change_pct = (current_price - buy_price) / buy_price

            # 止盈
            if change_pct >= self.p.take_profit_pct:
                print(f"✅ 止盈触发：{data._name} 涨幅 {change_pct:.2%}")
                self.close(data)
                continue

            # 止损
            if change_pct <= -self.p.stop_loss_pct:
                print(f"⛔ 止损触发：{data._name} 跌幅 {change_pct:.2%}")
                self.close(data)

    def handle_rebalance(self, dt):
        print(f"🔁 {dt.date()} 开始调仓逻辑")

        if not self.validate_index_data():
            print("⚠️ 指数数据不足，跳过调仓")
            return

        if self.check_stop_conditions(dt):
            return

        candidates = self.filter_stocks()
        is_momentum_ok = self.check_momentum_rank()
        hold_num = self.p.hold_count_high if is_momentum_ok else self.p.hold_count_low
        to_hold = set(candidates[:hold_num])
        current_hold = {d for d, pos in self.positions.items() if pos.size > 0}

        to_sell = current_hold - to_hold
        to_buy = to_hold - current_hold

        for d in to_sell:
            print(f"💸 清仓：{d._name}")
            self.close(d)

        # available_cash = self.broker.getcash() * 0.95
        available_cash = self.broker.getcash()
        cash_per_stock = available_cash / max(len(to_buy), 1)

        for d in to_buy:
            price = d.close[0]
            if price is None or np.isnan(price) or price <= 0:
                continue
            size = int(cash_per_stock // price)
            size = (size // 100) * 100
            if size >= 100:
                print(f"📥 买入：{d._name} size={size}")
                self.buy(d, size=size)

        self.print_positions()

    def check_stop_conditions(self, dt):
        if self.check_trend_crash():
            print(f"🚨 {dt.date()} 触发趋势止损")
            self.sell_all()
            self.clear_until = dt.date() + timedelta(days=7)
            self.is_cleared = True
            return True

        if not self.check_momentum_rank():
            print(f"⚠️ {dt.date()} 动量止损触发")
            self.sell_all()
            self.clear_until = dt.date() + timedelta(days=7)
            self.is_cleared = True
            return True

        self.is_cleared = False
        return False

    def validate_index_data(self):
        names = [self.p.smallcap_index] + self.p.large_indices
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

        if len(d) < days + 1:
            return -999

        prices = d.close.get(size=days + 1)
        if prices is None or len(prices) < days + 1:
            return -999

        if np.any(np.isnan(prices)) or prices[-1] == 0:
            return -999

        return get_momentum(prices, method="log", days=days)

    def get_volatility(self, name, days=10):
        """
        计算近 N 日对数收益率的年化波动率
        """
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
        """
        更稳健的趋势熔断判断：
        - 连续3日中至少2天下跌超过-3%
        - 或3日均跌幅超过-4%
        - 且波动率不高（年化<2%）
        """
        try:
            d = self.getdatabyname(self.p.smallcap_index)
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

        # 日收益率
        daily_return = close / open_ - 1

        # 条件1：3日中2天跌幅 > -3%
        crash_days = np.sum(daily_return < -0.03)

        # 条件2：3日日均跌幅
        avg_return = daily_return.mean()

        # 条件3：指数波动率（近10日）
        vol = self.get_volatility(self.p.smallcap_index, days=10)

        print(f'📉 全局熔断判断：3日跌幅={daily_return}, avg={avg_return:.2%}, vol={vol:.2%}')

        if (crash_days >= 2 or avg_return < -0.04) and vol < 0.2:
            print("🚨 触发更稳健的趋势熔断机制")
            return True

        return False

    def check_momentum_rank(self):
        indices = [self.p.smallcap_index] + self.p.large_indices
        returns = {name: self.get_index_return(name, self.p.momentum_days) for name in indices}
        sorted_returns = sorted(returns.items(), key=lambda x: x[1], reverse=True)
        print(f'📊 动量排名: {sorted_returns}')
        return sorted_returns[0][0] == self.p.smallcap_index

    def filter_stocks(self):
        candidates = []
        for d in self.datas:
            if d._name in [self.p.smallcap_index] + self.p.large_indices:
                continue
            try:
                mv = d.mv[0]
                profit = d.profit[0]
                revenue = d.revenue[0]
                is_st = d.is_st[0]
                close = d.close[0]
                roeAvg = d.roeAvg[0]
                profit_ttm = d.profit_ttm[0]

                if (mv > self.p.min_mv
                        and profit > 0
                        and 2 < close < self.p.hight_price
                        and roeAvg > 0
                        and profit_ttm > 0
                        and revenue > self.p.min_revenue
                        and is_st == 0):
                    candidates.append((d, mv))
            except:
                continue

        candidates = sorted(candidates, key=lambda x: x[1])
        return [x[0] for x in candidates]

    def sell_all(self):
        print('💰 清仓 - sell_all')
        for data, pos in self.positions.items():
            if pos.size > 0:
                self.close(data)

    def print_positions(self):
        total_value = self.broker.getvalue()
        print(f"\n📊 当前账户总市值: {total_value:,.2f}")
        for d in self.datas:
            pos = self.getposition(d)
            if pos.size > 0:
                buy_price = pos.price
                current_price = d.close[0]
                market_value = pos.size * current_price
                cost = pos.size * buy_price
                profit = market_value - cost
                pnl_pct = 100 * profit / cost if cost else 0
                print(f"{d._name:<12} 持仓: {pos.size:>6} 当前价: {current_price:.2f} 盈亏: {profit:.2f} ({pnl_pct:.2f}%)")