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
        rebalance_weekday=1,         # 每周调仓日（1 = 周一数据）周二早上开盘买入
        hold_count_high=5,           # 行情好时持股数（集中）
        hold_count_low=5,           # 行情差时持股数（分散）
        hight_price=50,           # 个股最高限价
        momentum_days=15,            # 动量观察窗口
        trend_threshold=-0.05,       # 快速熔断阈值（小市值单日下跌5%）
        null_index='etf_SZ511880',   # 空仓期备选 etf
        # 588000,科创50ETF
        # smallcap_index='sh.000852',     # 中证 1000 sh000852 # 小市值指数名称  小市值的动量如何确定 第一种，用中证2000可以近似代替/第二种，用微盘指数可以近似代替
        smallcap_index='csi932000',     # 中证 1000 sh000852 # 小市值指数名称  小市值的动量如何确定 第一种，用中证2000可以近似代替/第二种，用微盘指数可以近似代替
        # smallcap_index='sz399101',     # 399101 中小综指399101
        # large_indices=['HS300', '300etf', 'SH50', '50etf', 'DividendETF'],  # 大盘指数对比列表
        large_indices=['sh.000300', 'etf_SH159919', 'sh.000016', 'etf_SZ510050',  'etf_SZ510880']  # 大盘指数对比列表  沪深300/上证50/红利ETF 510880
        # large_indices=['sh.000300',  'etf_SZ510050',  'etf_SZ510880']  # 大盘指数对比列表  沪深300/上证50/红利ETF 510880
    )

    def __init__(self):
        print('✅ 初始化 SmallCapStrategy')
        self.clear_until = None  # 清仓维持到的日期
        self.is_cleared = False  # 当前是否处于清仓状态
        self.is_rebalance = False  # 当前时间是否处于调仓状态

        # 设置调仓定时器
        self.add_timer(
            when=bt.Timer.SESSION_START,
            weekdays=[self.p.rebalance_weekday],
            weekcarry=True,
            timername='rebalance_timer',
        )

    # def next_open(self):
    #     dt = self.datetime.datetime(0)
    #     print(f"🕘 next_open: {dt}")
    #
    #     if not self.validate_index_data():
    #         print("⚠️ 指数数据不足，跳过调仓")
    #         return
    #
    #     # 止损逻辑
    #     if self.check_trend_crash():
    #         self.sell_all()
    #         self.clear_until = dt.date() + timedelta(days=7)
    #         self.is_cleared = True
    #         self.ready_to_buy = False
    #         return
    #
    #     if not self.check_momentum_rank():
    #         print(f"⚠️ {dt.date()} 动量止损触发")
    #         self.sell_all()
    #         self.is_cleared = True
    #         self.ready_to_buy = False
    #         return
    #
    #     # 如果在清仓观察期，跳过调仓
    #     if self.clear_until and dt.date() < self.clear_until:
    #         self.ready_to_buy = False
    #         return
    #
    #     self.is_cleared = False
    #
    #     # 正常调仓：选股 & 卖出当前不在 to_hold 中的股票
    #     candidates = self.filter_stocks()
    #     hold_num = self.p.hold_count_high if self.check_momentum_rank() else self.p.hold_count_low
    #     self.to_hold = candidates[:hold_num]
    #
    #     current_positions = {d for d, pos in self.positions.items() if pos.size > 0}
    #     for d in current_positions - set(self.to_hold):
    #         print(f"清仓：{d._name}")
    #         self.close(d)
    #
    #     self.ready_to_buy = True  # 下一轮准备买入

    def notify_timer(self, timer, when, *args, **kwargs):
        '''
        运行时间在 next 方法之前
        :param timer:
        :param when:
        :param args:
        :param kwargs:
        :return:
        '''
        if kwargs.get('timername') == 'rebalance_timer':
            print(f'📅 调仓时间触发: {self.data0.datetime.date(0)}')
            self.rebalance()

    def next(self):
        """
        主逻辑, 每次都会调用
        """
        dt = self.data0.datetime.datetime(0)
        print('📈 next 执行时间:', self.datetime.datetime(0), '账户净值:', self.broker.getvalue())

        if self.is_rebalance: # 跳过调仓， 刚刚notify_timer调过，不需要在执行
            self.is_rebalance = False
            return
        # todo 止损模块应该每天都计算
        # 动量止损
        is_momentum_ok = self.check_momentum_rank()
        is_check_trend = self.check_trend_crash()
        # 快速趋势止损（小市值单日下跌5%） + 动量判断
        print(f'SmallCapStrategy.next stop loss result, is_check_trend：{is_check_trend}, is_momentum_ok： {is_momentum_ok}')
        if is_check_trend or not is_momentum_ok:
            # self.sell_all()
            self.sell_all_not_etf()
            self.is_cleared = True
            return

    def rebalance(self):
        dt = self.data0.datetime.datetime(0)
        self.is_rebalance = True  # 处于调仓状态

        # 若数据不足或缺指数，跳过
        if not self.validate_index_data():
            print("⚠️ 指数数据不足，跳过调仓")
            return

        print("📥 调仓前持仓情况：")
        self.print_positions()

        is_momentum_ok = self.check_momentum_rank()
        is_check_trend = self.check_trend_crash()
        # 快速趋势止损（小市值单日下跌5%） + 动量判断
        print(f'SmallCapStrategy.rebalance stop loss result, is_check_trend：{is_check_trend}, is_momentum_ok： {is_momentum_ok}')
        if is_check_trend or not is_momentum_ok:

            self.clear_until = dt.date() + timedelta(days=7)
            self.is_cleared = True
            # 判断是否 只持有 etf
            if self.is_holding_etf():
                pass
            else:
                self.sell_all()

                #
                # # buy_etf
                # # 待买入的新股票
                # buy_etf = self.getdatabyname(self.p.null_index)
                #
                # # 分配可用现金（留5%冗余）
                # cash_per_stock = self.broker.getcash() * 0.95
                # price = buy_etf.close[0]
                # size = int(cash_per_stock // price)
                # size = (size // 100) * 100  # ✅ 向下取整为一手
                # if size >= 100:
                #     print(f"买入：{buy_etf._name}, size={size}")
                #     self.buy(buy_etf, size=size)

            return


        # 如果处于清仓观察期则跳过调仓
        # if self.clear_until and dt.date() < self.clear_until:
        #     return
        # self.is_cleared = False

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

        print("📤 调仓后持仓情况：")
        self.print_positions()


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
        # print('SmallCapStrategy.get_index_return')

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
        return get_momentum(prices, method="log", days=days)
        # return get_momentum(prices, method="slope_r2", days=days)
        # return get_momentum(prices, method="slope", days=days)


    def check_trend_crash(self):
        """获取指定指数的 N 日动量值（可配置动量方法）"""
        # print('SmallCapStrategy.get_index_return')
        try:
            d = self.getdatabyname(self.p.smallcap_index)
        except Exception as e:
            print(f"⚠️ 指数 {self.p.smallcap_index} 获取失败: {e}")
            return -999
        if len(d) < 2:
            print(f"⚠️ 指数 {self.p.smallcap_index} 长度不足（{len(d)} < 2）")
            return -999
        # 获取最近 (1) 个收盘价
        close_prices = d.close.get(size=1)
        open_prices = d.open.get(size=1)

        # 判定异常数据
        if np.any(np.isnan(close_prices)) or close_prices[-1] == 0 or np.any(np.isnan(open_prices)) or open_prices[-1] == 0:
            print(f"⚠️ 指数 {self.p.smallcap_index} 存在缺失值或最新价为0")
            return -999
        r = close_prices[-1] / open_prices[-1] - 1

        print(f'🚨 趋势止损判断：{r:.4f}')
        return r < self.p.trend_threshold

    def check_momentum_rank(self):
        # print('SmallCapStrategy.check_momentum_rank')
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
                roeAvg = d.roeAvg[0] # 收盘价
                mv = d.mv[0] # 市值
                profit = d.profit[0] # 净利润
                revenue = d.revenue[0] # 主营营业收入
                is_st = d.is_st[0] # 是否ST
                profit_ttm = d.profit_ttm[0] # 母公司股东净利润
                if (mv > self.p.min_mv  # 市值大于 10亿
                        # and mv < self.p.min_mv*10  # 市值小于 100亿
                        and profit > 0  # 净利润大于0
                        and 2 < close < self.p.hight_price  # 收盘价限制
                        and roeAvg > 0  # ROE（净资产收益率，Return on Equity）为正表示公司 盈利，
                        and profit_ttm > 0  # 母公司股东净利润大于0
                        and revenue > self.p.min_revenue  # 主营收入大于 1亿
                        and is_st == 0):
                    candidates.append((d, mv))
            except:
                continue
        # 按市值升序排序
        candidates = sorted(candidates, key=lambda x: x[1])
        return [x[0] for x in candidates]

    def check_stop_conditions(self, dt):
        """统一处理止损逻辑"""
        # 快速趋势止损（小市值单日下跌5%）
        if self.check_trend_crash():
            print(f"🚨 {dt.date()} 触发趋势止损")
            self.sell_all()
            self.clear_until = dt.date() + timedelta(days=7)
            self.is_cleared = True
            return True

        # 动量止损
        is_momentum_ok = self.check_momentum_rank()
        print(f'SmallCapStrategy.check_momentum_rank result is_momentum_ok： {is_momentum_ok}')
        if not is_momentum_ok:
            print(f"⚠️ {dt.date()} 动量止损触发")
            self.sell_all()
            self.is_cleared = True
            return True

        self.is_cleared = False
        return False

    def sell_all(self):
        print('💰 清仓 - sell_all')
        for data, pos in self.positions.items():
            if pos.size > 0:
                self.close(data)

    def sell_all_not_etf(self):
        print('💰 清仓 - sell_all_not_etf')
        for data, pos in self.positions.items():
            if pos.size > 0:
                if data._name == self.p.null_index:
                    print(f"💰 {data._name} 跳过清仓动作")
                    continue
                self.close(data)


    def is_holding_etf(self):
        len_pos = 0
        has_etf = False
        for data, pos in self.positions.items():
            if pos.size > 0:
                len_pos = len_pos + 1
                if data._name == self.p.null_index:
                    has_etf = True
        if has_etf and len_pos == 1:
            return True
        else:
            return False
    def print_positions(self):
        total_value = self.broker.getvalue()
        total_cost = 0.0
        total_market_value = 0.0

        print(f"\n\n📊 当前账户总市值: {total_value:,.2f}")
        print(f"{'股票':<12} {'数量':>6} {'买入价':>10} {'当前价':>10} "
              f"{'市值':>12} {'占比%':>8} {'盈亏¥':>10} {'盈亏%':>8}")

        for d in self.datas:
            pos = self.getposition(d)
            if pos.size > 0:
                buy_price = pos.price
                current_price = d.close[0]
                market_value = pos.size * current_price
                cost = pos.size * buy_price
                profit = market_value - cost
                percent = 100 * market_value / total_value
                pnl_pct = 100 * profit / cost if cost else 0

                total_cost += cost
                total_market_value += market_value

                print(f"{d._name:<12} {pos.size:>6} {buy_price:>10.2f} {current_price:>10.2f} "
                      f"{market_value:>12,.2f} {percent:>8.2f} {profit:>10,.2f} {pnl_pct:>8.2f}")

        # 汇总行
        total_profit = total_market_value - total_cost
        total_profit_pct = 100 * total_profit / total_cost if total_cost else 0
        print("-" * 90)
        print(f"{'合计':<40} {total_market_value:>12,.2f} {'':>8} {total_profit:>10,.2f} {total_profit_pct:>8.2f}\n")