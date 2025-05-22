import backtrader as bt
import pandas as pd

#主策略逻辑
#init里写指标
#next里写买卖逻辑，这里在定时器里写卖的逻辑

class Doublelow2(bt.Strategy):
    params = (
        ('rebal_weekday1',1),
        ('num_volume',10),
    )

    # 日志函数
    def log(self, txt, dt=None):
        """
        :param txt:
        :param dt:
        :return:
        """
        # 以第一个数据data0，即指数作为时间基准
        dt = dt or self.data0.datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.lastRanks = []  # 上次交易股票的列表
        self.order = {}
        self.stocks = self.datas
        self.inds = dict()
        # 定时器
        self.add_timer(
            when=bt.Timer.SESSION_START,
            weekdays=[self.p.rebal_weekday1],
            weekcarry=True,  # if a day isn't there, execute on the next
            timername='rebaltimer1'
        )

    def notify_trade(self, trade):
        """
        记录交易收益情况
        """
        if not trade.isclosed:
            return
        self.log(f"策略收益：\n毛收益 {trade.pnl:.2f}, 净收益 {trade.pnlcomm:.2f}")

    def start(self):
        """
        回测开始前输出结果
        """
        self.log("初始总资金 %.2f" % (self.broker.getvalue()) )

    def stop(self):
        """
        回测结束后输出结果
        """
        self.log("期末总资金 %.2f" % (self.broker.getvalue()) )

    def notify_timer(self, timer, when, *args, **kwargs):
        print('notify_timer:', 'timer:', timer, 'when:', when, args, kwargs)
        timername = kwargs.get('timername', None)
        if timername == 'rebaltimer1':
            self.rebalance_portfolio()  # 执行再平衡
            print('调仓时间：', self.data0.datetime.date(0))


    def next(self):
        """
        主逻辑, 每次都会调用
        """
        print('next 账户总值', self.data0.datetime.datetime(0), self.broker.getvalue())

    def rebalance_portfolio(self):
        print('调仓日')
        # 1 先做排除筛选过程
        self.ranks = [d for d in self.stocks if
                      len(d) > 0
                      # and d.momentum_5[0] <= 0
                      and d.public_date[0] >= 1.5 and d.public_date[0] <= 5
                      and d.volume > 1
                      ]


        # 2 再做排序挑选过程
        self.ranks.sort(key=lambda d: d.double_low2, reverse=False)  # 按双低值从小到大排序
        self.ranks = self.ranks[0:self.p.num_volume]  # 取前num_volume名
        if len(self.ranks) != 0:
            for i, d in enumerate(self.ranks):
                print(f'选股第{i + 1}名,{d._name},momtum5值: {d.momentum_5[0]},双低值: {d.double_low2[0]},')
        else:  # 无债选入
            return


        # 3 以往买入的标的，本次不在标的中，则先平仓
        data_toclose = set(self.lastRanks) - set(self.ranks)
        for d in data_toclose:
            print('不在本次债池里：sell平仓', d._name, self.getposition(d).size)
            o = self.close(data=d)

        # 4 本次标的下单
        # 每只债买入资金百分比，预留2%的资金以应付佣金和计算误差
        buypercentage = (1 - 0.02) / len(self.ranks)

        # 得到目标市值
        targetvalue = buypercentage * self.broker.getvalue()
        # 为保证先卖后买，股票要按持仓市值从大到小排序
        self.ranks.sort(key=lambda d: self.broker.getvalue([d]), reverse=True)
        # self.log('下单, 标的个数 %i, targetvalue %.2f, 当前总市值 %.2f' %
        #          (len(self.ranks), targetvalue, self.broker.getvalue()))

        for d in self.ranks:
            # 按次日开盘价计算下单量，下单量是100的整数倍
            size = int(
                abs((self.broker.getvalue([d]) - targetvalue) / d.open[0] // 100 * 100))
            validday = d.datetime.datetime(1)  # 该股下一实际交易日
            if self.broker.getvalue([d]) > targetvalue:  # 持仓过多，要卖
                # 次日跌停价近似值
                lowerprice = d.close[0] * 0.9 + 0.03

                o = self.sell(data=d, size=size, exectype=bt.Order.Limit, valid=validday, price=lowerprice)
            else:  # 持仓过少，要买
                # 次日涨停价近似值,涨停值过滤不买
                upperprice = d.close[0] * 1.1 - 0.03
                o = self.buy(data=d, size=size, exectype=bt.Order.Limit, valid=validday, price=upperprice)

        self.lastRanks = self.ranks  # 跟踪上次买入的标的





#主策略逻辑
#init里写指标
#next里写买卖逻辑，这里在定时器里写卖的逻辑

class Doublelow1(bt.Strategy):
    params = (
        ('rebal_weekday1',1),
        ('num_volume',10),
    )

    # 日志函数
    def log(self, txt, dt=None):
        """
        :param txt:
        :param dt:
        :return:
        """
        # 以第一个数据data0，即指数作为时间基准
        dt = dt or self.data0.datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.lastRanks = []  # 上次交易股票的列表
        self.order = {}
        self.stocks = self.datas
        self.inds = dict()
        # 定时器
        self.add_timer(
            when=bt.Timer.SESSION_START,
            weekdays=[self.p.rebal_weekday1],
            weekcarry=True,  # if a day isn't there, execute on the next
            timername='rebaltimer1'
        )

    def notify_trade(self, trade):
        """
        记录交易收益情况
        """
        if not trade.isclosed:
            return
        self.log(f"策略收益：\n毛收益 {trade.pnl:.2f}, 净收益 {trade.pnlcomm:.2f}")

    def stop(self):
        """
        回测结束后输出结果
        """
        self.log("期末总资金 %.2f" % (self.broker.getvalue()))

    def notify_timer(self, timer, when, *args, **kwargs):
        timername = kwargs.get('timername', None)
        if timername == 'rebaltimer1':
            self.rebalance_portfolio()  # 执行再平衡
            print('调仓时间：', self.data0.datetime.date(0))



    def next(self):
        """
        主逻辑
        """
        print('next 账户总值', self.data0.datetime.datetime(0), self.broker.getvalue())

    def rebalance_portfolio(self):
        print('调仓日')
        # 1 先做排除筛选过程
        self.ranks = [d for d in self.stocks if
                      len(d) > 0
                      # and d.momentum_5[0] <= 0
                      and d.public_date[0] >= 1 and d.public_date[0] <= 5
                      and d.close[0] >= 98 and d.close[0] <= 150
                      and d.volume > 1
                      ]


        # 2 再做排序挑选过程
        self.ranks.sort(key=lambda d: d.double_low1, reverse=False)  # 按双低值从小到大排序
        self.ranks = self.ranks[0:self.p.num_volume]  # 取前num_volume名
        if len(self.ranks) != 0:
            for i, d in enumerate(self.ranks):
                print(f'选股第{i + 1}名,{d._name},momtum5值: {d.momentum_5[0]},close值: {d.close[0]},双低值: {d.double_low1[0]},')
        else:  # 无债选入
            return


        # 3 以往买入的标的，本次不在标的中，则先平仓
        data_toclose = set(self.lastRanks) - set(self.ranks)
        for d in data_toclose:
            print('不在本次债池里：sell平仓', d._name, self.getposition(d).size)
            o = self.close(data=d)

        # 4 本次标的下单
        # 每只债买入资金百分比，预留2%的资金以应付佣金和计算误差
        buypercentage = (1 - 0.02) / len(self.ranks)

        # 得到目标市值
        targetvalue = buypercentage * self.broker.getvalue()
        # 为保证先卖后买，股票要按持仓市值从大到小排序
        self.ranks.sort(key=lambda d: self.broker.getvalue([d]), reverse=True)
        # self.log('下单, 标的个数 %i, targetvalue %.2f, 当前总市值 %.2f' %
        #          (len(self.ranks), targetvalue, self.broker.getvalue()))

        for d in self.ranks:
            # 按次日开盘价计算下单量，下单量是100的整数倍
            size = int(
                abs((self.broker.getvalue([d]) - targetvalue) / d.open[0] // 100 * 100))
            validday = d.datetime.datetime(1)  # 该股下一实际交易日
            if self.broker.getvalue([d]) > targetvalue:  # 持仓过多，要卖
                # 次日跌停价近似值
                lowerprice = d.close[0] * 0.9 + 0.03

                o = self.sell(data=d, size=size, exectype=bt.Order.Limit, valid=validday, price=lowerprice)
            else:  # 持仓过少，要买
                # 次日涨停价近似值,涨停值过滤不买
                upperprice = d.close[0] * 1.1 - 0.03
                o = self.buy(data=d, size=size, exectype=bt.Order.Limit, valid=validday, price=upperprice)

        self.lastRanks = self.ranks  # 跟踪上次买入的标的



#主策略逻辑
#init里写指标
#next里写买卖逻辑，这里在定时器里写卖的逻辑，定时器能控制买卖的时间点，比如每周二，或者每月1号


class DoublelowTest(bt.Strategy):
    params = (
        ('rebal_weekday1',1),
        ('num_volume',10),
    )

    # 日志函数
    def log(self, txt, dt=None):
        """
        :param txt:
        :param dt:
        :return:
        """
        # 以第一个数据data0，即指数作为时间基准
        dt = dt or self.data0.datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        self.lastRanks = []  # 上次交易股票的列表
        self.order = {}
        self.stocks = self.datas
        self.inds = dict()
        # 定时器
        self.add_timer(
            when=bt.Timer.SESSION_START, # 开始时间
            weekdays=[self.params.rebal_weekday1], # 触发时间，数组，星期几
            weekcarry=True,  # if 如果有一天没有，就在第二天执行
            timername='rebaltimer1' # 定时器名称，用于识别
        )

    def start(self):
        """
        回测开始前输出结果
        """
        self.log("start--初始总资金 %.2f" % (self.broker.getvalue()) )

    def notify_trade(self, trade):
        """
        收到任何开仓/更新/平仓交易的通知
        记录交易收益情况
        """
        if not trade.isclosed:
            return
        self.log(f"notify_trade{self.data0.datetime.date(0)}--策略收益：毛收益 {trade.pnl:.2f}, 净收益 {trade.pnlcomm:.2f}")

    def notify_order(self, order):
        '''
        每当 order 发生更改时接收订单
        :param order:
        :return:
        '''
        if order.status in [order.Completed]:
            if order.isbuy():
                print(f'notify_order{self.data0.datetime.date(0)}--买入成交: {order.size}')
            elif order.issell():
                # 获取当前持仓数量
                print(f'notify_order{self.data0.datetime.date(0)}--卖出成交，持仓数量（size）: {abs(order.size)}')

    def notify_timer(self, timer, when, *args, **kwargs):
        '''
        和 add_timer 结合使用
        接收计时器通知，其中 timer 是 add_timer 返回的计时器，when 是调用时间。 参数 和 kwargs 是传递给 add_timer
        :param timer:
        :param when:
        :param args:
        :param kwargs:
        :return:
        '''
        timername = kwargs.get('timername', None) # 获取计时器名称
        # 获取当前持仓数量
        stake = self.position.size
        if timername == 'rebaltimer1':
            self.rebalance_portfolio()  # 执行再平衡
            print(f'notify_timer{self.data0.datetime.date(0)}--调仓时间：', self.data0.datetime.date(0), f'stake: {stake}')

    def next(self):
        """
        主逻辑, 每次都会调用，获取执行的时间：self.data0.datetime.datetime(0)
        常规使用 创建策略，实现__init__、next
        """

        # 获取佣金参数
        commission = self.broker.getcommissioninfo(self.data).params.commission
        # print(f"next--当前佣金比例: {commission}")
        # 获取持仓规模参数
        stake = self.getsizer().params.stake
        # print(f"next--当前持仓规模: {stake}")

        print(f'next{self.data0.datetime.date(0)}--账户总值', self.data0.datetime.datetime(0), self.broker.getvalue(), f"当前持仓规模: {stake}", f"当前佣金比例: {commission}" )


    def rebalance_portfolio(self):
        print(f'rebalance_portfolio--调仓日: {self.data0.datetime.datetime(0)}')
        # 1 先做排除筛选过程
        self.ranks = [d for d in self.stocks if
                      len(d) > 0
                      # and d.momentum_5[0] <= 0
                      and d.public_date[0] >= 1.5 and d.public_date[0] <= 5
                      # and d.close[0] >= 95 and d.close[0] <= 130
                      and d.volume > 1
                      ]

        # 2 再做排序挑选过程
        self.ranks.sort(key=lambda d: d.double_low2, reverse=False)  # 按双低值从小到大排序
        self.ranks = self.ranks[0:self.p.num_volume]  # 取前num_volume名
        if len(self.ranks) != 0:
            for i, d in enumerate(self.ranks):
                print(f'rebalance_portfolio{self.data0.datetime.date(0)}--选股第{i + 1}名,{d._name},momtum5值: {d.momentum_5[0]},双低值: {d.double_low2[0]},')
        else:  # 无债选入
            return

        # 3 以往买入的标的，本次不在标的中，则先平仓
        data_toclose = set(self.lastRanks) - set(self.ranks)
        for d in data_toclose:
            print(f'rebalance_portfolio{self.data0.datetime.date(0)}--不在本次债池里：sell平仓', d._name, self.getposition(d).size)
            o = self.close(data=d)


        # 4 本次标的下单
        # 每只债买入资金百分比，预留2%的资金以应付佣金和计算误差
        buypercentage = (1 - 0.02) / len(self.ranks)

        # 得到目标市值
        targetvalue = buypercentage * self.broker.getvalue()
        # 为保证先卖后买，股票要按持仓市值从大到小排序
        self.ranks.sort(key=lambda d: self.broker.getvalue([d]), reverse=True)
        self.log(f'rebalance_portfolio{self.data0.datetime.date(0)}--下单, 标的个数 %i, targetvalue %.2f, 当前总市值 %.2f' %
                 (len(self.ranks), targetvalue, self.broker.getvalue()))

        # 5 下单
        for d in self.ranks:
            # 按次日开盘价计算下单量，下单量是100的整数倍
            size = int(
                abs((self.broker.getvalue([d]) - targetvalue) / d.open[0] // 100 * 100))
            validday = d.datetime.datetime(1)  # 该股下一实际交易日
            if self.broker.getvalue([d]) > targetvalue:  # 持仓过多，要卖
                # 次日跌停价近似值
                lowerprice = d.close[0] * 0.9 + 0.03

                o = self.sell(data=d, size=size, exectype=bt.Order.Limit, valid=validday, price=lowerprice)
            else:  # 持仓过少，要买
                # 次日涨停价近似值,涨停值过滤不买
                upperprice = d.close[0] * 1.1 - 0.03
                o = self.buy(data=d, size=size, exectype=bt.Order.Limit, valid=validday, price=upperprice)

        self.lastRanks = self.ranks  # 跟踪上次买入的标的

    def stop(self):
        """
        回测结束后输出结果
        """
        self.log(f"stop{self.data0.datetime.date(0)}--期末总资金 %.2f" % (self.broker.getvalue()) )


