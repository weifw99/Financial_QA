# run_backtest.py - 主回测运行脚本，加载配置、数据，执行策略，记录结果
from datetime import datetime

import backtrader as bt
import pandas as pd
import quantstats
import yaml
import mlflow
import os
from utils.data_loader import load_stock_data
from strategies.smallcap_strategy import SmallCapStrategy

def load_config(path='config/config.yaml'):
    """从 YAML 配置文件加载策略参数"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run():
    config = load_config()
    cerebro = bt.Cerebro()
    # cerebro = bt.Cerebro(cheat_on_open= True)

    # 设置滑点和佣金
    cerebro.broker.set_slippage_perc(perc=0.00015)  # 买卖滑点各 0.015%
    cerebro.broker.setcommission(commission=0.00025)  # 万 2.5 的佣金
    # cerebro.broker.setcash(1000000)  # 初始资金
    cerebro.broker.setcash(100000)  # 初始资金

    from_idx = datetime(2025, 3, 1)  # 记录行情数据的开始时间和结束时间
    to_idx = datetime(2025, 6, 26)

    # from_idx = datetime(2014, 1, 1)  # 记录行情数据的开始时间和结束时间
    # to_idx = datetime(2025, 6, 26)

    print(from_idx, to_idx)
    # 加载所有股票与指数数据
    datafeeds = load_stock_data(from_idx, to_idx)
    for feed in datafeeds:
        cerebro.adddata(feed)
    print('load data DONE.', len(datafeeds))

    # 添加策略及其参数
    # cerebro.addstrategy(SmallCapStrategy, **config['strategy'])
    cerebro.addstrategy(SmallCapStrategy)
    print('add strategy DONE.')

    # 添加 PyFolio分析组件
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='_TimeReturn')
    # 计算夏普率
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='mysharpe')

    print('add analyzers DONE.')

    start_portfolio_value = cerebro.broker.getvalue()
    print(f'初始本金: {start_portfolio_value:.2f}')

    # ✅ 真正执行策略

    print('cerebro.run()')
    # result = cerebro.run(runonce=False, preload=True,stdstats=False)
    results = cerebro.run()
    strat = results[0]
    print("数据长度:", len(results[0].datas[0]))

    # 打印下每个 data 的长度：
    # for d in cerebro.datas:
    #     print(d._name, len(d))

    end_portfolio_value = cerebro.broker.getvalue()  # 最后的总资产
    pnl = end_portfolio_value - start_portfolio_value # 盈亏

    # 输出结果、生成报告、绘制图表
    print(f'初始本金 Portfolio Value: {start_portfolio_value:.2f}')
    print(f'最终本金和 Portfolio Value: {end_portfolio_value:.2f}')
    print(f'利润PnL: {pnl:.2f}')

    port_value = cerebro.broker.getvalue()
    print("Final Portfolio Value: %.2f" % port_value)

    # # 绘制净值图并记录
    # import matplotlib.pyplot as plt
    # cerebro.plot(style='candlestick')
    # # cerebro.plot(strat=result, style='candlestick')
    # plt.savefig("outputs/plots/net_value.png")


    # 打印夏普率
    print('Sharpe Ratio:', strat.analyzers.getbyname('mysharpe').get_analysis())

    portfolio_stats = strat.analyzers.getbyname('PyFolio')
    # PyFolio 使用 analyzer 方法 get_pf_items 检索 pyfolio 稍后需要的 4 个组件：
    # 收益率 (returns)、持仓 (positions)、交易记录 (transactions) 和杠杆水平 (gross_lev)
    returns, positions, transactions, gross_lev = portfolio_stats.get_pf_items()
    # returns.index = returns.index.tz_convert(None)

    print(type(returns))
    print(returns)
    quantstats.reports.html(returns, output=f'data/小市值策略{datetime.now().strftime("%Y-%m-%d %H:%M:%S")}.html', title='小市值策略')

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

    fig, (ax0, ax1) = plt.subplots(2, 1, gridspec_kw={'height_ratios': [3, 4]}, figsize=(30, 15))
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
                      bbox=(0, -0.05, 1, 1),  # 设置表格位置， (x0, y0, width, height)
                      rowLoc='right',  # 行标题居中
                      cellLoc='right',
                      colLabels=cols_names,  # 设置列标题
                      colLoc='right',  # 列标题居中
                      edges='open'  # 不显示表格边框
                      )
    table.auto_set_font_size(False)
    table.set_fontsize(10)  # 手动设置字体大小

    # 绘制累计收益曲线
    ax2 = ax1.twinx()
    ax1.yaxis.set_ticks_position('right')  # 将回撤曲线的 y 轴移至右侧
    ax2.yaxis.set_ticks_position('left')  # 将累计收益曲线的 y 轴移至左侧
    # 绘制回撤曲线
    drawdown.plot.area(ax=ax1, label='drawdown (right)', rot=0, alpha=0.3, fontsize=13, grid=False)
    # 绘制累计收益曲线
    (cumulative).plot(ax=ax2, color='#F1C40F', lw=3.0, label='cumret (left)', rot=0, fontsize=13, grid=False)
    # 设置横坐标标签竖排显示
    for label in ax2.get_xticklabels():
        label.set_rotation(90)
    # 或使用自动美化（建议同时使用）
    fig.autofmt_xdate()

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



if __name__ == '__main__':
    run()