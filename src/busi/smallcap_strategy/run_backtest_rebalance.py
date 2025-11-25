# run_backtest.py - 主回测运行脚本，加载配置、数据，执行策略，记录结果
from datetime import datetime

import backtrader as bt

from busi.smallcap_strategy.run_backtest import load_config
from busi.smallcap_strategy.strategies.rebalance_tuesday_strategy import RebalanceTuesdayStrategy
from busi.smallcap_strategy.utils.backtest_util import cerebro_show
from utils.data_loader import load_stock_data


def run():
    config = load_config()
    cerebro = bt.Cerebro(cheat_on_open=True)

    # 设置滑点和佣金
    cerebro.broker.set_slippage_perc(perc=0.00015)  # 买卖滑点各 0.015%
    cerebro.broker.setcommission(commission=0.00025)  # 万 2.5 的佣金
    # cerebro.broker.setcash(1000000)  # 初始资金
    cerebro.broker.setcash(100000)  # 初始资金

    from_idx = datetime(2015, 2, 15)  # 记录行情数据的开始时间和结束时间
    from_idx = datetime(2024, 2, 10)  # 记录行情数据的开始时间和结束时间
    # from_idx = datetime(2025, 2, 15)  # 记录行情数据的开始时间和结束时间
    to_idx = datetime(2025, 11, 30)

    # from_idx = datetime(2014, 1, 1)  # 记录行情数据的开始时间和结束时间
    # to_idx = datetime(2025, 6, 26)

    print(from_idx, to_idx)
    # 加载所有股票与指数数据
    datafeeds = load_stock_data(from_idx, to_idx)
    for feed in datafeeds:
        cerebro.adddata(feed)
    print('load data DONE.', len(datafeeds))


    '''
    from busi.smallcap_strategy.utils.selected_industries_util import load_industry_price, load_industry_fundflow

    base_price_path = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/industry/industry_price"
    base_path = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/industry"
    # 加载数据
    df_price = load_industry_price(base_price_path)
    df_flow = load_industry_fundflow(f'{base_path}/industry_flow.csv')

    # 1. 初始化
    from busi.smallcap_strategy.test.industry_factor_research import IndustryFactorResearch

    research = IndustryFactorResearch(df_price, df_flow)
    code_industry_dict = research.build_code_industry_dict()

    # {'RPS周期': 5, 'future_day': 20, 'w_hot': 0.05, 'w_advanced': 0.95, 'IC_mean': 0.13631835749094978, 'window_trend': 7}
    research.build_hot_factors()
    research.build_advanced_flow_features(window_trend=7)

    research.build_rps(5)
    research.compute_future_ret(20)
    research.build_combo_score_advanced(w_hot=0.05, w_advanced=0.95)

    print(research.get_daily_quantile_details())

    # 添加策略及其参数
    # cerebro.addstrategy(RebalanceTuesdayStrategy)
    from busi.smallcap_strategy.strategies.in_rebalance_tuesday_strategy import InRebalanceTuesdayStrategy
    cerebro.addstrategy(InRebalanceTuesdayStrategy, ind_dict=research.get_daily_quantile_details(), stock_ind=code_industry_dict)
    
    '''
    cerebro.addstrategy(RebalanceTuesdayStrategy)
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