import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime

import quantstats
from busi.etf_.bt_data import Getdata, Dailydataextend
from busi.etf_.etf_momentum_strategy import  MomentumStrategy1, MomentumStrategyV2

if __name__ == '__main__':

    from_idx = datetime(2020, 1, 1)  # 记录行情数据的开始时间和结束时间
    # from_idx = datetime(2025, 3, 12)  # 记录行情数据的开始时间和结束时间
    to_idx = datetime(2025, 11, 15)
    print(from_idx, to_idx)
    #启动回测

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(100000.0)
    cerebro.broker.setcommission(commission=0.0005)
    # print(data_1,data_1.columns)

    pool_file = 'data/etf_strategy/etf_pool_120.csv'
    pool_file = 'data/etf_strategy/etf_pool.csv'
    pool_file = 'data/etf_strategy/etf_pool1.csv'
    pool_file = 'data/etf_strategy/etf_pool_test.csv'
    df = pd.read_csv(pool_file)
    etf_codes = df['代码'].tolist()
    #获取数据源
    datas = Getdata(symbols=etf_codes)
    data_1 = datas.dailydata()

    out=[]
    for stk_code in etf_codes:
        data_date = pd.DataFrame(index=data_1.index.unique())
        df = data_1[data_1['symbol'] == str(stk_code)]
        df = df.sort_index()
        data_ = pd.merge(data_date, df, left_index=True, right_index=True, how='left')
        data_.loc[:, ['volume', 'openinterest']] = data_.loc[:, ['volume', 'openinterest']].fillna(0)
        data_.loc[:, ['open', 'high', 'low', 'close']] = data_.loc[:, ['open', 'high', 'low', 'close']].bfill()
        data_.bfill(inplace=True)
        data_.fillna(0, inplace=True)
        data_.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)

        data_d = Dailydataextend(dataname=data_,
                                 fromdate=from_idx,
                                 todate=to_idx,
                                 timeframe=bt.TimeFrame.Days,
                                 name=f'1d_{stk_code}')
        cerebro.adddata(data_d)
        out.append(stk_code)

    print('统计数量为{}'.format(len(out)), 'Done !')


    # 载入策略
    # cerebro.addstrategy(MomentumStrategy1)
    cerebro.addstrategy(MomentumStrategyV2)
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

    quantstats.reports.html(returns, output='etf.html', title='etf')
