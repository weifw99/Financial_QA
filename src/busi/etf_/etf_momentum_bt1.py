import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime
import quantstats
from busi.etf_.bt_data import Getdata, Dailydataextend
from busi.etf_.etf_momentum_strategy import MomentumStrategy, MomentumStrategy1
from itertools import product
from multiprocessing import Pool, cpu_count
import traceback

def evaluate_single_combo(datafeeds, strategy_cls, momentum_params, initial_cash=100000):
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(initial_cash)

    # 添加数据
    for data in datafeeds:
        cerebro.adddata(data)

    # 添加策略
    cerebro.addstrategy(strategy_cls, momentum_params=momentum_params)

    # 添加分析器
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')
    result = cerebro.run()[0]

    # 计算结果指标
    final_value = cerebro.broker.getvalue()
    return_ratio = (final_value - initial_cash) / initial_cash

    # 计算夏普比率
    sharpe = None
    try:
        returns = result.analyzers.pyfolio.get_analysis()['returns']
        returns = pd.Series(returns)
        # 移除无效值
        returns = returns.replace([np.inf, -np.inf], np.nan).dropna()
        if len(returns) > 0 and returns.std() > 0:
            sharpe = returns.mean() / returns.std() * np.sqrt(252)
    except Exception as e:
        print(f"计算夏普比率时出错：{e}")

    # 返回结果
    result_dict = {
        'final_value': final_value,
        'return_ratio': return_ratio,
        'sharpe': sharpe
    }
    
    # 添加动量参数
    result_dict.update(momentum_params)
    
    return result_dict

def prepare_datafeeds(datas, etf_codes, momentum_params, from_idx, to_idx):
    """准备数据源"""
    # 获取数据
    data_1 = datas.dailydata1(momentum_params=momentum_params)
    
    # 准备数据源
    datafeeds = []
    for stk_code in etf_codes:
        # 获取该ETF的数据
        df = data_1.xs(stk_code, level='symbol', drop_level=False)
        if df.empty:
            print(f"警告：未找到ETF {stk_code} 的数据")
            continue
            
        # 重置索引，将日期作为列
        df = df.reset_index(level='date')
        
        # 确保日期列是datetime类型
        df['date'] = pd.to_datetime(df['date'])
        
        # 确保数据按日期排序
        df = df.sort_values('date')
        
        # 处理缺失值
        df.loc[:, ['volume', 'openinterest']] = df.loc[:, ['volume', 'openinterest']].fillna(0)
        df.loc[:, ['open', 'high', 'low', 'close']] = df.loc[:, ['open', 'high', 'low', 'close']].bfill()
        df.bfill(inplace=True)
        df.fillna(0, inplace=True)
        df.dropna(subset=['open', 'high', 'low', 'close'], inplace=True)
        
        # 创建数据源
        data_d = Dailydataextend(
            dataname=df,
            fromdate=from_idx,
            todate=to_idx,
            timeframe=bt.TimeFrame.Days,
            name=f'1d_{stk_code}'
        )
        datafeeds.append(data_d)
    
    return datafeeds

def evaluate_combo_with_args(args):
    strategy_cls, datas, etf_codes, param_keys, param_values, from_idx, to_idx = args
    combo_dict = dict(zip(param_keys, param_values))
    try:
        # 为每个参数组合准备数据
        datafeeds = prepare_datafeeds(datas, etf_codes, combo_dict, from_idx, to_idx)
        result = evaluate_single_combo(datafeeds, strategy_cls, combo_dict)
        combo_dict.update(result)
        return combo_dict
    except Exception as e:
        print(f"组合失败: {combo_dict}, 错误: {e}\n{traceback.format_exc()}")
        return None

def run_grid_search_parallel(strategy_cls, datas, etf_codes, param_grid, from_idx, to_idx, n_jobs=None):
    param_keys = list(param_grid.keys())
    param_combinations = list(product(*param_grid.values()))
    n_jobs = n_jobs or cpu_count()

    # 构建任务参数
    task_args = [(strategy_cls, datas, etf_codes, param_keys, values, from_idx, to_idx) 
                for values in param_combinations]

    print(f"启动并行网格搜索，共 {len(task_args)} 个组合，使用 {n_jobs} 核心, task_args: {task_args}")

    with Pool(processes=n_jobs) as pool:
        result_list = pool.map(evaluate_combo_with_args, task_args)

    result_list = [r for r in result_list if r is not None]
    return pd.DataFrame(result_list)

if __name__ == '__main__':
    from_idx = datetime(2020, 1, 1)
    to_idx = datetime(2025, 1, 1)
    print(from_idx, to_idx)

    pool_file = 'data/etf_strategy/etf_pool.csv'
    df = pd.read_csv(pool_file)
    etf_codes = df['代码'].tolist()

    # 获取数据源
    datas = Getdata(symbols=etf_codes)
    
    # 定义不同动量计算方式的参数网格
    param_grids = {
        'linear': {
            'linear_window': [20, 30, 40, 50, 60, 90, 120],
        },
        # 'simple': {
        #     'simple_window': [20, 30, 40, 50, 60, 90, 120],
        # },
        # 'log_simple': {
        #     'log_simple_window': [20, 30, 40, 50, 60, 90, 120],
        # },
        # 'log_r2': {
        #     'log_r2_window': [20, 30, 40, 50, 60, 90, 120],
        # },
        # 'line_log_r2': {
        #     'line_log_r2_window': [20, 30, 40, 50, 60, 90, 120],
        # },
        # 'dual': {
        #     'long_window': [60, 90, 120, 150],
        #     'short_window': [20, 30, 40, 50],
        #     'smooth_long': [10, 20],
        #     'smooth_short': [3, 5],
        #     'min_long_return': [0.01, 0.02],
        #     'min_short_return': [0.005, 0.01],
        #     'long_weight': [0.6, 0.7, 0.8],
        #     'short_weight': [0.2, 0.3, 0.4],
        # },
        # 'dual_v2': {
        #     'long_window': [60, 90, 120, 150],
        #     'short_window': [20, 30, 40, 50],
        #     'smooth_long': [10, 20],
        #     'smooth_short': [3, 5],
        #     'min_long_return': [0.01, 0.02],
        #     'min_short_return': [0.005, 0.01],
        #     'slope_positive_filter': [True, False],
        #     'weight_long': [0.6, 0.7, 0.8],
        #     'weight_short': [0.2, 0.3, 0.4],
        # }
        # 'dual': {
        #     'long_window': [ 150],
        #     'short_window': [ 50],
        #     'smooth_long': [10, 20],
        #     'smooth_short': [3, 5],
        #     'min_long_return': [0.01, 0.02],
        #     'min_short_return': [0.005, 0.01],
        #     'long_weight': [0.6, 0.8],
        #     'short_weight': [0.2, 0.4],
        # },
        # 'dual_v2': {
        #     'long_window': [150],
        #     'short_window': [ 50],
        #     'smooth_long': [10, 20],
        #     'smooth_short': [3, 5],
        #     'min_long_return': [0.01, 0.02],
        #     'min_short_return': [0.005, 0.01],
        #     'slope_positive_filter': [True ],
        #     'weight_long': [0.6,  0.8],
        #     'weight_short': [0.2, 0.4],
        # }
    }

    # 对每种动量计算方式进行网格搜索
    all_results = []
    for momentum_type, param_grid in param_grids.items():
        print(f"\n开始测试 {momentum_type} 动量计算方式")
        
        # 运行网格搜索
        df_results = run_grid_search_parallel(
            MomentumStrategy1, 
            datas, 
            etf_codes, 
            param_grid, 
            from_idx, 
            to_idx, 
            n_jobs=1
        )
        df_results['momentum_type'] = momentum_type
        all_results.append(df_results)

    # 合并所有结果
    final_results = pd.concat(all_results, ignore_index=True)
    final_results.sort_values(by='sharpe', ascending=False).to_csv("momentum_comparison_results.csv", index=False)


