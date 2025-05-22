import backtrader as bt
import pandas as pd
from datetime import datetime


# 自定义策略
class TestStrategy1(bt.Strategy):
    # 全局参数
    params = (
        ('maperiod', 15),
    )

    def log(self, txt, dt=None):
        ''' 记录'''
        dt = dt or self.datas[0].datetime.date(0)
        print('%s, %s' % (dt.isoformat(), txt))

    # 初始化
    def __init__(self):
        # Keep a reference to the "close" line in the data[0] dataseries
        self.dataclose = self.datas[0].close

        # To keep track of pending orders and buy price/commission
        # 跟踪未完成的订单和购买价格 / 佣金
        self.order = None
        self.buyprice = None
        self.buycomm = None

        # 增加移动平均指标
        self.sma = bt.indicators.MovingAverageSimple(self.datas[0], period=self.params.maperiod)

        # 增加划线的指标
        bt.indicators.ExponentialMovingAverage(self.datas[0], period=25)
        bt.indicators.WeightedMovingAverage(self.datas[0], period=25, subplot=True)
        # bt.indicators.StochasticSlow(self.datas[0])
        bt.indicators.MACDHisto(self.datas[0])
        rsi = bt.indicators.RSI(self.datas[0])
        bt.indicators.SmoothedMovingAverage(rsi, period=10)
        bt.indicators.ATR(self.datas[0], plot=False)

    def notify_order(self, order):
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log(
                    'BUY EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                    (order.executed.price,
                     order.executed.value,
                     order.executed.comm))

                self.buyprice = order.executed.price
                self.buycomm = order.executed.comm
            else:  # Sell
                self.log('SELL EXECUTED, Price: %.2f, Cost: %.2f, Comm %.2f' %
                         (order.executed.price,
                          order.executed.value,
                          order.executed.comm))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def notify_trade(self, trade):
        if not trade.isclosed:
            return

        self.log('OPERATION PROFIT, GROSS %.2f, NET %.2f' %
                 (trade.pnl, trade.pnlcomm))

    # 每个 bar 执行的操作
    def next(self):
        # Simply log the closing price of the series from the reference
        self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        if self.order:
            return

        # Check if we are in the market
        if not self.position:

            # Not yet ... we MIGHT BUY if ...
            if self.dataclose[0] > self.sma[0]:
                # BUY, BUY, BUY!!! (with all possible default parameters)
                self.log('BUY CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.buy()

        else:

            if self.dataclose[0] < self.sma[0]:
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell()


# Create a Stratey
class TestStrategy(bt.Strategy):

    def log(self, txt, dt=None):
        ''' 提供记录功能'''
        dt = dt or self.datas[0].datetime.date(0)
        # print(f'{len(self.datas)}, {self.datas[0]}, {self.datas[0]._colmapping}')
        print('%s, %s' % (dt.isoformat(), txt))

    def __init__(self):
        # 引用到输入数据的close价格
        self.dataclose = self.datas[0].close
        # To keep track of pending orders
        # 跟踪未完成的订单
        self.order = None


    def notify_order(self, order):
        """
        订单通知，
        :param order:
        :return:
        """
        self.log('(%s) Order: %s' %('notify_order', self.dataclose[0]))
        if order.status in [order.Submitted, order.Accepted]:
            # Buy/Sell order submitted/accepted to/by broker - Nothing to do
            return

        # Check if an order has been completed
        # Attention: broker could reject order if not enough cash
        if order.status in [order.Completed]:
            if order.isbuy():
                self.log('%s BUY EXECUTED, %.2f' % ('notify_order', order.executed.price))
            elif order.issell():
                self.log('%s SELL EXECUTED, %.2f' % ('notify_order', order.executed.price))

            self.bar_executed = len(self)

        elif order.status in [order.Canceled, order.Margin, order.Rejected]:
            self.log('notify_order, Order Canceled/Margin/Rejected')

        # Write down: no pending order
        self.order = None

    def next(self):
        # 是简单显示下收盘价。
        # self.log('Close, %.2f' % self.dataclose[0])

        # Check if an order is pending ... if yes, we cannot send a 2nd one
        # 检查订单是否在处理中
        if self.order:
            return

        # 检查是否在市场
        if not self.position:

            # 不在，那么连续3天价格下跌就买点
            if self.dataclose[0] < self.dataclose[-1]:
                # 当前价格比上一次低

                if self.dataclose[-1] < self.dataclose[-2]:
                    # 上一次的价格比上上次低

                    # 买入!!!
                    self.log('BUY CREATE, %.2f' % self.dataclose[0])

                    # Keep track of the created order to avoid a 2nd order
                    self.order = self.buy()

        else:

            # 已经在市场，3天后就卖掉。
            if len(self) >= (self.bar_executed + 3):
                # 这里注意，Len(self)返回的是当前执行的bar数量，每次next会加1.而Self.bar_executed记录的最后一次交易执行时的bar位置。
                # SELL, SELL, SELL!!! (with all possible default parameters)
                self.log('SELL CREATE, %.2f' % self.dataclose[0])

                # Keep track of the created order to avoid a 2nd order
                self.order = self.sell



if __name__ == '__main__':
    cerebro = bt.Cerebro()

    # 设置资金池
    cerebro.broker.setcash(100000.0)
    # 设置佣金，0.1% ... 除以100去掉%号。
    cerebro.broker.setcommission(commission=0.001)
    # Add a FixedSize sizer according to the stake 每次买卖的股数量
    cerebro.addsizer(bt.sizers.FixedSize, stake=100)


    # 增加一个策略
    cerebro.addstrategy(TestStrategy1)
    # 增加多参数的策略
    # strats = cerebro.optstrategy(
    #     TestStrategy1,
    #     maperiod=range(10, 31))

    # 获取数据
    file_paths = [
        '/Users/dabai/liepin/study/llm/Financial_QA/src/busi/bond_/data/stock_trading/daily/SH600000.csv',
        '/Users/dabai/liepin/study/llm/Financial_QA/src/busi/bond_/data/stock_trading/daily/SH600004.csv',
    ]
    start_date = datetime(2010, 9, 30)  # 回测开始时间
    end_date = datetime(2021, 9, 30)  # 回测结束时间

    temp_datas = []
    for file_path in file_paths:
        print(file_path)
        stock_hfq_df = pd.read_csv(file_path, index_col='date', parse_dates=True)

        # 'date','open','high','low','close','volume','turn','pctChg'
        stock_hfq_df = stock_hfq_df[['open','high','low','close','volume','pctChg']]
        stock_hfq_df['openinterest'] = 0

        # 删除 null
        # stock_hfq_df.loc[:, ['volume', 'openinterest']] = stock_hfq_df.loc[:, ['volume', 'openinterest']].fillna(0.0001)
        # stock_hfq_df.loc[:, ['open', 'high', 'low', 'close']] = stock_hfq_df.loc[:, ['open', 'high', 'low', 'close']].bfill()
        stock_hfq_df.bfill(inplace=True)
        stock_hfq_df.fillna(0.00001, inplace=True)
        stock_hfq_df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

        stock_hfq_df = stock_hfq_df.sort_index() # 按日期排序 正序(升序, 必须)
        data = bt.feeds.PandasData(dataname=stock_hfq_df, fromdate=start_date, todate=end_date)  # 加载数据
        cerebro.adddata(data)  # 将数据传入回测系统
        temp_datas.append(data)

    print('Starting Portfolio Value: %.2f' % cerebro.broker.getvalue())
    opt_runs = cerebro.run()
    print('Final Portfolio Value: %.2f' % cerebro.broker.getvalue())

    '''
    best_result = None
    best_final_value = float('-inf')

    # 遍历优化结果，找出最佳结果
    for run in opt_runs:
        for strat in run:
            final_value = strat.cerebro.broker.getvalue()
            if final_value > best_final_value:
                best_final_value = final_value
                best_result = strat

    # 输出最佳结果
    if best_result:
        print(f"最佳策略参数: fast_period={best_result.p.fast_period}, slow_period={best_result.p.slow_period}")
        print(f"最终资金: {best_final_value}")
    '''
    cerebro.plot() # 画图 cerebro.optstrategy不可以使用，因为optstrategy返回的是一个列表，而不是一个策略对象。
    # 遍历优化结果并绘图
    # for run in opt_runs:
    #     for strat in run:
    #         print( type(strat), strat.p._getkwargs() )
    #         # 为每个策略实例重新创建 Cerebro 引擎
    #         new_cerebro = bt.Cerebro()
    #         for data_ in temp_datas:
    #             new_cerebro.adddata(data_)
    #         # 添加当前策略实例 strat.strategycls
    #         new_cerebro.addstrategy(strat.strategycls, **strat.p._getkwargs())
    #         # 运行策略
    #         new_cerebro.run()
    #         # 绘图
    #         new_cerebro.plot()