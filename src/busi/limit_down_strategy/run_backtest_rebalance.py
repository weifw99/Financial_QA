# run_backtest.py - 主回测运行脚本，加载配置、数据，执行策略，记录结果
from datetime import datetime

import backtrader as bt

from busi.limit_down_strategy.strategies.rebalance_tuesday_strategy import RebalanceTuesdayStrategy, TestStrategy, \
    MeanReversionStrategy
from busi.limit_down_strategy.utils.backtest_util import cerebro_show
from utils.data_loader import load_stock_data


def run():
    # cerebro = bt.Cerebro(cheat_on_open=True)
    cerebro = bt.Cerebro()

    # 设置滑点和佣金
    cerebro.broker.set_slippage_perc(perc=0.00015)  # 买卖滑点各 0.015%
    cerebro.broker.setcommission(commission=0.00025)  # 万 2.5 的佣金
    # cerebro.broker.setcash(1000000)  # 初始资金
    cerebro.broker.setcash(100000)  # 初始资金

    from_idx = datetime(2025, 1, 1)  # 记录行情数据的开始时间和结束时间
    to_idx = datetime(2025, 7, 23)

    # from_idx = datetime(2014, 1, 1)  # 记录行情数据的开始时间和结束时间
    # to_idx = datetime(2025, 6, 26)

    print(from_idx, to_idx)
    # 加载所有股票与指数数据
    datafeeds = load_stock_data(from_idx, to_idx)
    for feed in datafeeds:
        cerebro.adddata(feed)
    print('load data DONE.', len(datafeeds))

    # 添加策略及其参数
    # cerebro.addstrategy(RebalanceTuesdayStrategy)
    # cerebro.addstrategy(TestStrategy)
    # 策略参数：三种卖出方式可选
    cerebro.addstrategy(MeanReversionStrategy,
                        max_stock_num=3,
                        max_hold_num=5,
                        sell_mode="hold_N_days",  # 可切换 open_next / stop_profit_loss / hold_N_days
                        take_profit=0.03,
                        stop_loss=-0.01,
                        hold_days=2)
    print('add strategy DONE.')

    # 添加 PyFolio分析组件
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='PyFolio')
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='_TimeReturn')
    # 计算夏普率
    cerebro.addanalyzer(bt.analyzers.SharpeRatio, _name='mysharpe')

    print('add analyzers DONE.')

    cerebro_show(cerebro)



if __name__ == '__main__':
    run()