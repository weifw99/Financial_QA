# strategies/smallcap_strategy.py
# 小市值策略主类，包含调仓逻辑、止损机制与行情判断
import backtrader as bt
from datetime import datetime, timedelta

class SmallCapStrategy(bt.Strategy):
    params = dict(
        min_mv=10e8,                   # 最小市值
        min_profit=0,                  # 最小净利润
        min_revenue=1e8,              # 最小营业收入
        rebalance_weekday=1,         # 每周调仓日（1 = 周二）
        rebalance_time=1000,         # 调仓时间（上午10点）
        hold_count_high=5,           # 行情好时持股数（集中）
        hold_count_low=10,           # 行情差时持股数（分散）
        momentum_days=20,            # 动量观察窗口
        trend_threshold=-0.05,       # 快速熔断阈值（小市值单日下跌5%）
        smallcap_index='ZZ2000',     # 小市值指数名称
        large_indices=['HS300', 'SH50', 'DividendETF']  # 大盘指数对比列表
    )

    def __init__(self):
        super().__init__()
        print('SmallCapStrategy.init')
        # 标记调仓时间与状态
        self.rebalance_date = None
        self.clear_until = None  # 清仓维持到的日期
        self.is_cleared = False  # 当前是否处于清仓状态

    def next(self):
        print('SmallCapStrategy.next')
        dt = self.datas[0].datetime.datetime(0)

        # 判断是否为调仓时间（每周二上午10点）
        if dt.weekday() == self.p.rebalance_weekday and dt.hour * 100 + dt.minute >= self.p.rebalance_time:
            if self.rebalance_date == dt.date():
                return
            self.rebalance_date = dt.date()

            # 快速趋势止损（小市值单日下跌5%）
            if self.check_trend_crash():
                self.sell_all()
                self.clear_until = dt.date() + timedelta(days=7)
                self.is_cleared = True
                return

            # 动量止损（小市值动量不再领先）
            if self.check_momentum_rank() is False:
                self.sell_all()
                self.is_cleared = True
                return

            # 如果处于清仓观察期则跳过调仓
            if self.clear_until and dt.date() < self.clear_until:
                return
            else:
                self.is_cleared = False

            # 选股
            candidates = self.filter_stocks()
            hold_num = self.p.hold_count_high if self.check_momentum_rank() else self.p.hold_count_low
            to_hold = candidates[:hold_num]  # 取前N只小市值股

            # 调仓逻辑：先清掉非目标股票，再买入新股
            for d in self.datas:
                if self.getposition(d).size > 0 and d not in to_hold:
                    self.close(d)
            for d in to_hold:
                if self.getposition(d).size == 0:
                    self.buy(d)

    def sell_all(self):
        print('SmallCapStrategy.sell_all')
        for d in self.datas:
            if self.getposition(d).size > 0:
                self.close(d)

    def get_index_return(self, name, days):
        print('SmallCapStrategy.get_index_return')
        """获取指定指数的N日收益率"""
        d = self.getdatabyname(name)
        if len(d) < days + 1:
            return 0
        return d.close[0] / d.close[-days] - 1

    def check_trend_crash(self):
        print('SmallCapStrategy.check_trend_crash')
        """判断是否触发趋势熔断"""
        return self.get_index_return(self.p.smallcap_index, 1) < self.p.trend_threshold

    def check_momentum_rank(self):
        print('SmallCapStrategy.check_momentum_rank')
        """判断小市值指数是否仍然是动量排名第一"""
        indices = [self.p.smallcap_index] + self.p.large_indices
        returns = {name: self.get_index_return(name, self.p.momentum_days) for name in indices}
        return sorted(returns.items(), key=lambda x: x[1], reverse=True)[0][0] == self.p.smallcap_index

    def filter_stocks(self):
        print('SmallCapStrategy.filter_stocks')
        """选出符合财务和市值要求的小市值股票"""
        candidates = []
        for d in self.datas:
            try:
                if d._name in [self.p.smallcap_index] + self.p.large_indices:
                    continue
                mv = d.mv[0]
                profit = d.profit[0]
                revenue = d.revenue[0]
                is_st = d.is_st[0]
                if mv > self.p.min_mv and profit > 0 and revenue > self.p.min_revenue and not is_st:
                    candidates.append((d, mv))
            except:
                continue
        # 按市值升序排序
        candidates = sorted(candidates, key=lambda x: x[1])
        return [x[0] for x in candidates]