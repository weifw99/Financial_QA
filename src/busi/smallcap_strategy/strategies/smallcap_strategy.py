# strategies/smallcap_strategy.py
# 小市值策略主类，使用 notify_timer 控制调仓标志，在 next 中统一调仓逻辑

import backtrader as bt
from datetime import datetime, timedelta
import numpy as np
from busi.smallcap_strategy.utils.momentum_utils import get_momentum


class SmallCapStrategy(bt.Strategy):
    params = dict(
        min_mv=10e8,  # 最小市值 10亿
        min_profit=0,  # 最小净利润
        min_revenue=1e8,  # 最小营业收入
        rebalance_weekday=1,  # 每周调仓日（1 = 周一数据）周二早上开盘买入
        hold_count_high=5,  # 行情好时持股数（集中）
        hold_count_low=5,  # 行情差时持股数（分散）
        hight_price=50,  # 个股最高限价
        momentum_days=15,  # 动量观察窗口
        trend_threshold=-0.05,  # 快速熔断阈值（小市值单日下跌5%）
        stop_loss_pct=0.06,  # 个股止损线（跌幅超过6%）
        take_profit_pct=0.5,  # 个股止盈线（涨幅超过50%）
        null_index='etf_SZ511880',  # 空仓期备选 etf
        # smallcap_index=['csi932000', 'sz399101', 'sh000852'],  # 小市值指数列表（中证2000 + 中小综指 + 中证 1000）
        # smallcap_index=[ 'sz399101', 'sh000852'],  # 小市值指数列表（中证2000 + 中小综指 + 中证 1000）
        # smallcap_index=[ 'csi932000', 'sz399101', 'sh000852', 'sh000046', 'sz399005', 'sz399401'],  # 小市值指数列表（中证2000 + 中小综指 + 中证 1000）
        # smallcap_index=[ 'csi932000', 'sh000046', 'sz399005', 'sz399401'],  # 小市值指数列表（中证2000 + 中小综指 + 中证 1000）
        smallcap_index=[ 'csi932000','sz399101'],  # 小市值指数列表（中证2000 + 中小综指 + 中证 1000）
        # smallcap_index=[ 'csi932000', 'sz399005', 'sz399401'],  # 小市值指数列表（中证2000 + 中小综指 + 中证 1000）
        large_indices=['sh.000300', 'etf_SH159919', 'sh.000016', 'etf_SZ510050', 'etf_SZ510880', 'sh000905']
    )

    def __init__(self):
        self.clear_until = None
        self.is_cleared = False
        self.do_rebalance_today = False
        self.empty_days = 0
        self.max_empty_days = 5

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
        dt = self.data0.datetime.datetime(0)
        print('📈 next 执行时间:', self.datetime.datetime(0), '账户净值:', self.broker.getvalue())
        self.check_individual_stop()

        current_pos = sum([pos.size for _, pos in self.positions.items()])
        if current_pos == 0:
            self.empty_days += 1
        else:
            self.empty_days = 0

        if self.do_rebalance_today:
            self.do_rebalance_today = False
            self.handle_rebalance(dt)
            return

        is_momentum_ok = self.check_momentum_rank(top_k=2)
        # is_check_trend = self.check_trend_crash()
        is_check_trend = self.check_combo_trend_crash()
        print(f'SmallCapStrategy.next stop loss result, is_check_trend：{is_check_trend}, is_momentum_ok： {is_momentum_ok}')

        if is_check_trend or not is_momentum_ok:
            self.sell_all()
            self.is_cleared = True
            return

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

            if change_pct >= self.p.take_profit_pct:
                print(f"✅ 止盈触发：{data._name} 涨幅 {change_pct:.2%}")
                self.close(data)
                continue

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
        is_momentum_ok = self.check_momentum_rank(top_k=2)
        hold_num = self.p.hold_count_high if is_momentum_ok else self.p.hold_count_low

        if self.empty_days >= self.max_empty_days:
            print(f"📆 已空仓 {self.empty_days} 天，强制放宽调仓限制")
            is_momentum_ok = True
            hold_num = self.p.hold_count_high

        to_hold = set(candidates[:hold_num])
        print(f"{dt.date()} 待持仓：{to_hold}")
        current_hold = {d for d, pos in self.positions.items() if pos.size > 0}

        to_sell = current_hold - to_hold
        to_buy = to_hold - current_hold
        print(f"{dt.date()} to_sell：{to_sell}")
        print(f"{dt.date()} to_buy：{to_buy}")

        for d in to_sell:
            print(f"💸 清仓：{d._name}")
            self.close(d)

        available_cash = self.broker.getcash()
        cash_per_stock = available_cash / max(len(to_buy), 1)

        for d in to_buy:
            price = d.close[0]
            if price is None or np.isnan(price) or price <= 0:
                continue
            size = int(cash_per_stock // price)
            size = (size // 100) * 100
            print(f"📥 准备买入：{d._name} size={size} cash_per_stock: {cash_per_stock}, price: {price}")
            if size >= 100:
                print(f"📥 买入：{d._name} size={size}")
                self.buy(d, size=size)
            else:
                print(f"⚠️ 跳过买入：{d._name} size={size}")

        self.print_positions()

    def check_stop_conditions(self, dt):
        # if self.check_trend_crash():
        if self.check_combo_trend_crash():
            print(f"🚨 {dt.date()} 触发趋势止损")
            self.sell_all()
            self.clear_until = dt.date() + timedelta(days=7)
            self.is_cleared = True
            return True

        if not self.check_momentum_rank(top_k=2):
            print(f"⚠️ {dt.date()} 动量止损触发")
            self.sell_all()
            self.clear_until = dt.date() + timedelta(days=7)
            self.is_cleared = True
            return True

        self.is_cleared = False
        return False

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

        if len(d) < days + 1:
            return -999

        prices = d.close.get(size=days + 1)
        if prices is None or len(prices) < days + 1:
            return -999

        if np.any(np.isnan(prices)) or prices[-1] == 0:
            return -999
        momentum_log = get_momentum(prices, method='log', days=days)
        momentum_slope = get_momentum(prices, method='slope_r2', days=days)

        # 组合方式（例如加权平均）
        combo_score = 0.5 * momentum_log + 0.5 * momentum_slope
        return combo_score
        # return get_momentum(prices, method="log", days=days)
        # return get_momentum(prices, method="slope_r2", days=days)

    def get_combined_smallcap_momentum(self):
        scores = [self.get_index_return(name, self.p.momentum_days) for name in self.p.smallcap_index]
        valid_scores = [s for s in scores if s > -999]
        print(f'📊 小市值动量: {scores}')
        return np.mean(valid_scores) if valid_scores else -999

    def check_recent_recovery(self):
        recovery_scores = []
        for i in range(3):
            day_scores = []
            for name in self.p.smallcap_index:
                d = self.getdatabyname(name)
                if len(d) < self.p.momentum_days + i + 1:
                    return False
                prices = d.close.get(size=self.p.momentum_days + i + 1)
                if np.any(np.isnan(prices)):
                    return False
                score = get_momentum(prices[-(self.p.momentum_days + 1 + i):-i or None], method="log",
                                     days=self.p.momentum_days)
                day_scores.append(score)
            recovery_scores.append(np.mean(day_scores))
        print(f'📊 最近三个动量: {recovery_scores}')
        return recovery_scores[2] > recovery_scores[1] > recovery_scores[0]

    def check_momentum_rank(self, top_k=2):
        combo_score = self.get_combined_smallcap_momentum()
        returns = {name: self.get_index_return(name, self.p.momentum_days) for name in self.p.large_indices}
        returns['__smallcap_combo__'] = combo_score

        sorted_returns = sorted(returns.items(), key=lambda x: x[1], reverse=True)
        print(f'📊 动量排名: {sorted_returns}')

        in_top_k = '__smallcap_combo__' in [x[0] for x in sorted_returns[:top_k]]
        is_recovering = self.check_recent_recovery()

        if not in_top_k and not is_recovering :
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

        crash_days = np.sum(daily_return < -0.03)
        avg_return = daily_return.mean()
        vol = np.std(np.diff(np.log(close_avg))) * np.sqrt(252)

        print(f'📉 组合趋势止损判断：3日组合涨跌={daily_return}, 平均={avg_return:.2%}, 波动率={vol:.2%}')

        if (crash_days >= 2 or avg_return < -0.04) and vol < 0.2:
            print("🚨 触发组合小市值指数的趋势熔断机制")
            return True

        return False

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

                is_st = d.is_st[0]
                turn = d.turn[0]
                close = d.close[0]
                amount = d.amount[0]

                mv = d.mv[0]

                # 年度数据
                profit_y = d.revenue_y[0]
                revenue_y = d.revenue_y[0]
                roeAvg_y = d.roeAvg_y[0]
                profit_ttm_y = d.profit_ttm_y[0]

                # 季度数据
                profit_q = d.revenue_q[0]
                revenue_q = d.revenue_q[0] # 季度可能为 null
                roeAvg_q = d.roeAvg_q[0]
                profit_ttm_q = d.profit_ttm_q[0]

                if (mv > self.p.min_mv
                        and is_st == 0
                        and 2 < close < self.p.hight_price
                        # 年度数据
                        and profit_y > 0
                        and roeAvg_y > 0
                        and profit_ttm_y > 0
                        and revenue_y > self.p.min_revenue

                        # 季度数据
                        and profit_q > 0
                        and roeAvg_q > 0
                        and profit_ttm_q > 0
                ):

                    candidates.append((d, mv))
            except:
                print(f"⚠️ 获取股票数据失败: {d._name}")
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