# run_backtest.py - 主回测运行脚本，加载配置、数据，执行策略，记录结果
from datetime import datetime

import backtrader as bt

from busi.midcap_strategy.strategies.rebalance_tuesday_strategy import RebalanceTuesdayStrategy
from busi.midcap_strategy.strategies.rebalance_tuesday_strategy1 import RebalanceTuesdayStrategy1
from busi.midcap_strategy.utils.backtest_util import cerebro_show
from utils.data_loader import load_stock_data


def run():
    cerebro = bt.Cerebro(cheat_on_open=True)

    # 设置滑点和佣金
    cerebro.broker.set_slippage_perc(perc=0.00015)  # 买卖滑点各 0.015%
    cerebro.broker.setcommission(commission=0.00025)  # 万 2.5 的佣金
    # cerebro.broker.setcash(1000000)  # 初始资金
    cerebro.broker.setcash(100000)  # 初始资金

    from_idx = datetime(2017, 3, 1)  # 记录行情数据的开始时间和结束时间
    # from_idx = datetime(2010, 1, 15)  # 记录行情数据的开始时间和结束时间
    to_idx = datetime(2025, 12, 20)

    # from_idx = datetime(2014, 1, 1)  # 记录行情数据的开始时间和结束时间
    # to_idx = datetime(2025, 6, 26)

    print(from_idx, to_idx)
    model_result_path = ['/Users/dabai/liepin/study/llm/Financial_QA/src/qlib_/train_test/rolling_train_tree/data/zxzz399101_rec_tree_expanding/pre_result.csv',
                         '/Users/dabai/liepin/study/llm/Financial_QA/src/qlib_/train_test/rolling_train_tree/data/zxzz399101_tree_all_expanding/pre_result.csv',
                         '/Users/dabai/liepin/study/llm/Financial_QA/src/qlib_/train_test/rolling_train_tree/data/zxzz399101_tree_import_expanding/pre_result.csv',
                         '/Users/dabai/liepin/study/llm/Financial_QA/src/qlib_/train_test/rolling_train_tree/data/zxzz399101_tree_select1_expanding/pre_result.csv',
                         ]


    # 加载所有股票与指数数据
    datafeeds = load_stock_data(from_idx, to_idx, model_result_path)
    for feed in datafeeds:
        cerebro.adddata(feed)
    print('load data DONE.', len(datafeeds))

    # 添加策略及其参数
    cerebro.addstrategy(RebalanceTuesdayStrategy1)
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