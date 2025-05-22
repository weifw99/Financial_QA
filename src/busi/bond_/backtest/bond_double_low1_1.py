import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime

import quantstats

from busi.bond_.backtest.data import Dailydataextend, Getdata


#主策略逻辑
#init里写指标
#next里写买卖逻辑，这里在定时器里写卖的逻辑


class Doublelow(bt.Strategy):
    params = (
        ('rebal_weekday1',1),
        ('rebal_weekday5',5),
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
        self.add_timer(
            when=bt.Timer.SESSION_START,
            weekdays=[self.p.rebal_weekday5],
            weekcarry=True,  # if a day isn't there, execute on the next
            timername='rebaltimer5'
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
            # self.rebalance_portfolio()  # 执行再平衡
            print('调仓时间：买入', self.data0.datetime.date(0))
            self.buy_all()
        if timername == 'rebaltimer5':
            # self.rebalance_portfolio()  # 执行再平衡
            print('调仓时间：卖出', self.data0.datetime.date(0))
            self.closs_all()



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
                      and d.public_date[0] >= 1.5 and d.public_date[0] <= 5
                      and d.close[0] >= 100 and d.close[0] <= 150
                      and d.volume > 1
                      ]

        # 2 再做排序挑选过程
        self.ranks.sort(key=lambda d: d.double_low1, reverse=False)  # 按双低值从小到大排序
        self.ranks = self.ranks[0:self.p.num_volume]  # 取前num_volume名
        if len(self.ranks) != 0:
            for i, d in enumerate(self.ranks):
                print(
                    f'选股第{i + 1}名,{d._name},momtum5值: {d.momentum_5[0]},close值: {d.close[0]},双低值: {d.double_low1[0]},')
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


    def buy_all(self):
        print('调仓日')
        # 1 先做排除筛选过程
        self.ranks = [d for d in self.stocks if
                      len(d) > 0
                      # and d.momentum_5[0] <= 0
                      and d.public_date[0] >= 1.5 and d.public_date[0] <= 5
                      # and d.close[0] >= 100 and d.close[0] <= 150
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

    def closs_all(self):
        print('调仓日')
        for d in self.lastRanks:
            print('债池里：sell平仓', d._name, self.getposition(d).size)
            o = self.close(data=d)
        self.lastRanks = []  # 清空
















if __name__ == '__main__':

    from_idx = datetime(2024, 1, 1)  # 记录行情数据的开始时间和结束时间
    to_idx = datetime(2025, 1, 1)
    # from_idx = datetime(2025, 1, 1)  # 记录行情数据的开始时间和结束时间
    # to_idx = datetime(2025, 4, 10)
    print(from_idx, to_idx)
    #启动回测

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    #获取数据源
    data = Getdata()
    data_1 = data.dailydata()
    # print(data_1,data_1.columns)

    banks = data_1['symbol'].unique().tolist()


    out=[]
    for stk_code in banks:
        data = pd.DataFrame(index=data_1.index.unique())
        df = data_1[data_1['symbol'] == stk_code]
        df = df.sort_index()
        data_ = pd.merge(data, df, left_index=True, right_index=True, how='left')
        data_.loc[:, ['volume', 'openinterest']] = data_.loc[:, ['volume', 'openinterest']].fillna(0)
        data_.loc[:, ['open', 'high', 'low', 'close']] = data_.loc[:, ['open', 'high', 'low', 'close']].bfill()
        data_.bfill(inplace=True)
        data_.fillna(0, inplace=True)
        data_.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

        data_d = Dailydataextend(dataname=data_,
                                 fromdate=from_idx,
                                 todate=to_idx,
                                 timeframe=bt.TimeFrame.Days,
                                 name=f'1d_'+stk_code)
        cerebro.adddata(data_d)
        out.append(stk_code)

    print('统计数量为{}'.format(len(out)), 'Done !')


    # 载入策略
    cerebro.addstrategy(Doublelow)
    print('add strategy DONE.')

    # 添加分析
    # 添加 PyFolio分析组件
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='_TimeReturn')
    # 计算夏普率
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='mysharpe')

    print('add analyzers DONE.')
    start_portfolio_value = cerebro.broker.getvalue() # 初始的总资产
    results = cerebro.run()
    strat = results[0]
    end_portfolio_value = cerebro.broker.getvalue()  # 最后的总资产
    pnl = end_portfolio_value - start_portfolio_value # 盈亏

    # 输出结果、生成报告、绘制图表
    print(f'初始本金 Portfolio Value: {start_portfolio_value:.2f}')
    print(f'最终本金和 Portfolio Value: {end_portfolio_value:.2f}')
    print(f'利润PnL: {pnl:.2f}')

    # 打印夏普率
    print('Sharpe Ratio:', strat.analyzers.getbyname('mysharpe').get_analysis())

    portfolio_stats = strat.analyzers.getbyname('PyFolio')
    # PyFolio 使用 analyzer 方法 get_pf_items 检索 pyfolio 稍后需要的 4 个组件：
    # 收益率 (returns)、持仓 (positions)、交易记录 (transactions) 和杠杆水平 (gross_lev)
    returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
    # returns.index = returns.index.tz_convert(None)

    # print('-- RETURNS')
    # print(returns)
    # print('-- POSITIONS')
    # print(positions)
    # print('-- TRANSACTIONS')
    # print(transactions)
    # print('-- GROSS LEVERAGE')
    # print(gross_lev)


    pnl = pd.Series(results[0].analyzers._TimeReturn.get_analysis())
    # 计算累计收益
    cumulative = (pnl + 1).cumprod()
    # 计算回撤序列
    max_return = cumulative.cummax()
    drawdown = (cumulative - max_return) / max_return
    # 计算收益评价指标
    import pyfolio as pf

    # 按年统计收益指标
    perf_stats_year = (pnl).groupby(pnl.index.to_period('y')).apply(lambda data: pf.timeseries.perf_stats(data)).unstack()
    # 统计所有时间段的收益指标
    perf_stats_all = pf.timeseries.perf_stats((pnl)).to_frame(name='all')
    perf_stats = pd.concat([perf_stats_year, perf_stats_all.T], axis=0)
    perf_stats_ = round(perf_stats, 4).reset_index()

    print('输出指标:', perf_stats_.columns)
    print('输出指标:', perf_stats_)

    # 绘制图形
    import matplotlib.pyplot as plt

    # 设置字体 用来正常显示中文标签
    # plt.rcParams['font.sans-serif'] = ['SimHei']
    # 用来正常显示负号
    plt.rcParams['axes.unicode_minus'] = False

    import matplotlib.ticker as ticker  # 导入设置坐标轴的模块

    # plt.style.use('seaborn')  # plt.style.use('dark_background')
    plt.style.use('ggplot')  # 或者 plt.style.use('bmh')

    fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [1.5, 4]}, figsize=(20, 8))
    cols_names = ['date', 'Annual\nreturn', 'Cumulative\nreturns', 'Annual\nvolatility',
                  'Sharpe\nratio', 'Calmar\nratio', 'Stability', 'Max\ndrawdown',
                  'Omega\nratio', 'Sortino\nratio', 'Skew', 'Kurtosis', 'Tail\nratio',
                  'Daily value\nat risk']



    # cols_names = ['date', '年回报率', '累积回报', '年度波动',
    #               '夏普比率', 'Calmar比率', '稳定性', '最大回撤',
    #               'Omega\nratio', 'Sortino\nratio', 'Skew', 'Kurtosis', 'Tail\nratio',
    #               'Daily value\nat risk']
    # Calmar比率：表示基金的收益率与基金阶段最大回撤的比率
    # 绘制表格
    ax0.set_axis_off()  # 除去坐标轴
    table = ax0.table(cellText=perf_stats_.values,
                      bbox=(0, 0, 1, 1),  # 设置表格位置， (x0, y0, width, height)
                      rowLoc='right',  # 行标题居中
                      cellLoc='right',
                      colLabels=cols_names,  # 设置列标题
                      colLoc='right',  # 列标题居中
                      edges='open'  # 不显示表格边框
                      )
    table.set_fontsize(13)

    # 绘制累计收益曲线
    ax2 = ax1.twinx()
    ax1.yaxis.set_ticks_position('right')  # 将回撤曲线的 y 轴移至右侧
    ax2.yaxis.set_ticks_position('left')  # 将累计收益曲线的 y 轴移至左侧
    # 绘制回撤曲线
    drawdown.plot.area(ax=ax1, label='drawdown (right)', rot=0, alpha=0.3, fontsize=13, grid=False)
    # 绘制累计收益曲线
    (cumulative).plot(ax=ax2, color='#F1C40F', lw=3.0, label='cumret (left)', rot=0, fontsize=13, grid=False)
    # 不然 x 轴留有空白
    ax2.set_xbound(lower=cumulative.index.min(), upper=cumulative.index.max())
    # 主轴定位器：每 5 个月显示一个日期：根据具体天数来做排版
    ax2.xaxis.set_major_locator(ticker.MultipleLocator(100))
    # 同时绘制双轴的图例
    h1, l1 = ax1.get_legend_handles_labels()
    h2, l2 = ax2.get_legend_handles_labels()
    plt.legend(h1 + h2, l1 + l2, fontsize=12, loc='upper left', ncol=1)

    fig.tight_layout()  # 规整排版
    plt.show()


    print(type(returns))
    print(returns)
    quantstats.reports.html(returns, output='可转债双低策略1.html', title='可转债双低策略1')
