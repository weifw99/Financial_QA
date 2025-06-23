# run_backtest.py - 主回测运行脚本，加载配置、数据，执行策略，记录结果
from datetime import datetime

import backtrader as bt
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

    # 设置滑点和佣金
    cerebro.broker.set_slippage_perc(perc=0.00015)  # 买卖滑点各 0.015%
    cerebro.broker.setcommission(commission=0.00025)  # 万 2.5 的佣金
    cerebro.broker.setcash(1000000)  # 初始资金

    from_idx = datetime(2020, 1, 1)  # 记录行情数据的开始时间和结束时间
    to_idx = datetime(2025, 1, 1)
    print(from_idx, to_idx)
    # 加载所有股票与指数数据
    datafeeds = load_stock_data(from_idx, to_idx)
    for feed in datafeeds:
        cerebro.adddata(feed)
    print('load data DONE.', len(datafeeds))


    # 添加策略及其参数
    cerebro.addstrategy(SmallCapStrategy, **config['strategy'])

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

    result = cerebro.run()
    strat = result[0]

    print('cerebro.run()')

    end_portfolio_value = cerebro.broker.getvalue()  # 最后的总资产
    pnl = end_portfolio_value - start_portfolio_value # 盈亏

    # 输出结果、生成报告、绘制图表
    print(f'初始本金 Portfolio Value: {start_portfolio_value:.2f}')
    print(f'最终本金和 Portfolio Value: {end_portfolio_value:.2f}')
    print(f'利润PnL: {pnl:.2f}')


    port_value = cerebro.broker.getvalue()
    print("Final Portfolio Value: %.2f" % port_value)

    # 绘制净值图并记录
    import matplotlib.pyplot as plt
    cerebro.plot(style='candlestick')
    plt.savefig("outputs/plots/net_value.png")

if __name__ == '__main__':
    run()