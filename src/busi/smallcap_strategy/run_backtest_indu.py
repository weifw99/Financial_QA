# run_backtest.py - 主回测运行脚本，加载配置、数据，执行策略，记录结果
from datetime import datetime

import backtrader as bt

from busi.smallcap_strategy.run_backtest import load_config
from busi.smallcap_strategy.strategies.indu_rebalance_tuesday_strategy import InduRebalanceTuesdayStrategy
from busi.smallcap_strategy.utils.backtest_util import cerebro_show
from busi.smallcap_strategy.utils.selected_industries_util import get_indu_data, load_stock_industry_map
from utils.data_loader import load_stock_data


def run():
    config = load_config()
    cerebro = bt.Cerebro(cheat_on_open=True)

    # 设置滑点和佣金
    cerebro.broker.set_slippage_perc(perc=0.00015)  # 买卖滑点各 0.015%
    cerebro.broker.setcommission(commission=0.00025)  # 万 2.5 的佣金
    # cerebro.broker.setcash(1000000)  # 初始资金
    cerebro.broker.setcash(100000)  # 初始资金

    from_idx = datetime(2025, 4, 1)  # 记录行情数据的开始时间和结束时间
    to_idx = datetime(2025, 7, 4)

    # from_idx = datetime(2014, 1, 1)  # 记录行情数据的开始时间和结束时间
    # to_idx = datetime(2025, 6, 26)

    print(from_idx, to_idx)
    # 加载所有股票与指数数据
    datafeeds = load_stock_data(from_idx, to_idx)
    for feed in datafeeds:
        cerebro.adddata(feed)
    print('load data DONE.', len(datafeeds))


    # 添加策略及其参数
    cerebro.addstrategy(InduRebalanceTuesdayStrategy,
                        selected_industries=get_indu_data(),
                        indu_type="E1", # 0.0032
                        # indu_type="E", # 0.0032
                        # indu_type="D", # 0.1357
                        # indu_type="B", # 0.0664
                        # indu_type="A1", # 0.1221
                        # indu_type="A", # -0.0028
                        stock_industry_map=load_stock_industry_map())
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