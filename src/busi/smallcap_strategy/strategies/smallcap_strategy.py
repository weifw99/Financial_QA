# strategies/smallcap_strategy.py
# 小市值策略主类，包含调仓逻辑、止损机制与行情判断
import backtrader as bt
from datetime import datetime, timedelta
import numpy as np

from busi.smallcap_strategy.utils.momentum_utils import get_momentum


class SmallCapStrategy(bt.Strategy):
    params = dict(
        min_mv=10e8,                   # 最小市值 10亿
        min_profit=0,                  # 最小净利润
        min_revenue=1e8,              # 最小营业收入
        rebalance_weekday=1,         # 每周调仓日（1 = 周二）
        rebalance_time=1000,         # 调仓时间（上午10点）
        hold_count_high=5,           # 行情好时持股数（集中）
        hold_count_low=10,           # 行情差时持股数（分散）
        momentum_days=20,            # 动量观察窗口
        trend_threshold=-0.05,       # 快速熔断阈值（小市值单日下跌5%）
        smallcap_index='csi932000',     # 小市值指数名称  小市值的动量如何确定 第一种，用中证2000可以近似代替/第二种，用微盘指数可以近似代替
        # large_indices=['HS300', '300etf', 'SH50', '50etf', 'DividendETF'],  # 大盘指数对比列表
        large_indices=['sh.000300', 'etf_SH159300', 'sh.000016', 'etf_SZ510050',  'etf_SZ510880']  # 大盘指数对比列表  沪深300/上证50/红利ETF 510880
    )

    def __init__(self):
        print('✅ 初始化 SmallCapStrategy')
        self.rebalance_date = None
        self.clear_until = None  # 清仓维持到的日期
        self.is_cleared = False  # 当前是否处于清仓状态

        # 设置调仓定时器
        self.add_timer(
            when=bt.Timer.SESSION_START,
            weekdays=[self.p.rebalance_weekday],
            weekcarry=True,
            timername='rebalance_timer',
        )

    def notify_timer(self, timer, when, *args, **kwargs):
        if kwargs.get('timername') == 'rebalance_timer':
            print(f'📅 调仓时间触发: {self.data0.datetime.date(0)}')
            self.rebalance()

    def next(self):
        """
        主逻辑, 每次都会调用
        """
        print('📈 next 执行时间:', self.datetime.datetime(0), '账户净值:', self.broker.getvalue())

    def rebalance(self):
        dt = self.data0.datetime.datetime(0)

        # 若数据不足或缺指数，跳过
        if not self.validate_index_data():
            print("⚠️ 指数数据不足，跳过调仓")
            return

        # 同一天内不重复调仓
        if self.rebalance_date == dt.date():
            return
        self.rebalance_date = dt.date()

        # 快速趋势止损（小市值单日下跌5%）
        if self.check_trend_crash():
            self.sell_all()
            self.clear_until = dt.date() + timedelta(days=7)
            self.is_cleared = True
            return

        # 动量止损
        is_momentum_ok = self.check_momentum_rank()
        if not is_momentum_ok:
            print(f"⚠️ {dt.date()} 动量止损触发")
            self.sell_all()
            self.is_cleared = True
            return

        # 如果处于清仓观察期则跳过调仓
        if self.clear_until and dt.date() < self.clear_until:
            return
        self.is_cleared = False

        # 正常调仓
        candidates = self.filter_stocks()
        hold_num = self.p.hold_count_high if is_momentum_ok else self.p.hold_count_low
        to_hold = candidates[:hold_num]  # 取前N只小市值股

        # 当前持仓股票
        # 清仓：持有但不再目标池中的股票
        current_positions = {d for d, pos in self.positions.items() if pos.size > 0}

        for d in current_positions - set(to_hold):
            print(f"清仓：{d._name}")
            self.close(d)

        # 待买入的新股票
        to_buy = [d for d in to_hold if d not in current_positions]

        # 分配可用现金（留5%冗余）
        available_cash = self.broker.getcash() * 0.95
        cash_per_stock = available_cash / max(len(to_buy), 1)
        for d in to_buy:
            price = d.close[0]
            if price is None or np.isnan(price) or price <= 0:
                continue
            size = int(cash_per_stock // price)
            size = (size // 100) * 100  # ✅ 向下取整为一手
            if size >= 100:
                print(f"买入：{d._name}, size={size}")
                self.buy(d, size=size)


    def validate_index_data(self):
        """检查所有指数数据是否存在且长度够"""
        names = [self.p.smallcap_index] + self.p.large_indices
        for name in names:
            d = self.getdatabyname(name)
            if len(d) < self.p.momentum_days + 1 or np.isnan(d.close[0]):
                return False
        return True

    def get_index_return(self, name, days):
        """获取指定指数的 N 日动量值（可配置动量方法）"""
        print('SmallCapStrategy.get_index_return')

        try:
            d = self.getdatabyname(name)
        except Exception as e:
            print(f"⚠️ 指数 {name} 获取失败: {e}")
            return -999

        if len(d) < days + 1:
            print(f"⚠️ 指数 {name} 长度不足（{len(d)} < {days + 1}）")
            return -999

        # 获取最近 (days + 1) 个收盘价
        prices = d.close.get(size=days + 1)
        if prices is None or len(prices) < days + 1:
            print(f"⚠️ 指数 {name} 获取价格失败或不足")
            return -999

        # 判定异常数据
        if np.any(np.isnan(prices)) or prices[-1] == 0:
            print(f"⚠️ 指数 {name} 存在缺失值或最新价为0")
            return -999

        # 计算动量（可替换方法: "return" / "log" / "slope" / "slope_r2"）
        return get_momentum(prices, method="slope_r2", days=days)


    def check_trend_crash(self):
        r = self.get_index_return(self.p.smallcap_index, 1)
        print(f'🚨 趋势止损判断：{r:.4f}')
        return r < self.p.trend_threshold

    def check_momentum_rank(self):
        print('SmallCapStrategy.check_momentum_rank')
        """判断小市值指数是否仍然是动量排名第一"""
        indices = [self.p.smallcap_index] + self.p.large_indices
        returns = {name: self.get_index_return(name, self.p.momentum_days) for name in indices}
        sorted_returns = sorted(returns.items(), key=lambda x: x[1], reverse=True)
        print(f'📊 动量排名: {sorted_returns}')
        return sorted_returns[0][0] == self.p.smallcap_index

    def filter_stocks(self):
        print('SmallCapStrategy.filter_stocks')
        """选出符合财务和市值要求的小市值股票"""
        candidates = []
        for d in self.datas:
            try:
                if d._name in [self.p.smallcap_index] + self.p.large_indices:
                    continue
                close = d.close[0] # 收盘价
                mv = d.mv[0] # 市值
                profit = d.profit[0] # 净利润
                revenue = d.revenue[0] # 主营营业收入
                is_st = d.is_st[0] # 是否ST
                profit_ttm = d.profit_ttm[0] # 母公司股东净利润
                if (mv > self.p.min_mv  # 市值大于 10亿
                        and mv < self.p.min_mv*10  # 市值小于 100亿
                        and profit > 0  # 净利润大于0
                        and close > 1  # 收盘价大于1
                        and profit_ttm > 0  # 母公司股东净利润大于0
                        and revenue > self.p.min_revenue  # 主营收入大于 1亿
                        and is_st == 0):
                    candidates.append((d, mv))
            except:
                continue
        # 按市值升序排序
        candidates = sorted(candidates, key=lambda x: x[1])
        return [x[0] for x in candidates]

    def sell_all(self):
        print('💰 清仓')
        for data, pos in self.positions.items():
            if pos.size > 0:
                self.close(data)