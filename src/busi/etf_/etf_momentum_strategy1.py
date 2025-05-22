import backtrader as bt
import os
import pandas as pd
import argparse
from datetime import datetime
import matplotlib.pyplot as plt


class MomentumStrategy(bt.Strategy):
    params = (
        ('etf_list', []),
        ('rebalance_day', 1),
        ('topk', 5),
        ('stoploss', 0.05),
    )

    def __init__(self):
        self.order_dict = {}
        self.last_prices = {}
        self.initial_prices = {}
        self.rebalance_flag = False

    def next(self):
        dt = self.datas[0].datetime.date(0)
        if dt.day == self.p.rebalance_day:
            self.rebalance_flag = True
        else:
            self.rebalance_flag = False

        if self.rebalance_flag:
            self.rebalance()


        self.check_stoploss()

    def rebalance(self):
        self.log("Rebalancing Portfolio...")
        for d in self.datas:
            self.order_dict[d._name] = self.close(d)
            self.initial_prices[d._name] = d.close[0]

        weights = 1.0 / self.p.topk
        for i, d in enumerate(self.datas):
            if i < self.p.topk:
                self.order_target_percent(d, weights)
            else:
                self.order_target_percent(d, 0)

    def check_stoploss(self):
        for d in self.datas:
            name = d._name
            if self.getposition(d).size > 0 and name in self.initial_prices:
                current_price = d.close[0]
                initial_price = self.initial_prices[name]
                if current_price < initial_price * (1 - self.p.stoploss):
                    self.log(f"{name} hit stoploss, closing position.")
                    self.order_target_percent(d, 0)

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(f'{dt.isoformat()} {txt}')


def load_data_from_csv(code, fromdate, todate):
    df = pd.read_csv(f'data/{code}.csv', parse_dates=['date'], index_col='date')
    df = df[(df.index >= fromdate) & (df.index <= todate)]
    data = bt.feeds.PandasData(dataname=df, name=code)
    return data

def load_benchmark_data(code, fromdate, todate):
    df = pd.read_csv(f'data/{code}.csv', parse_dates=['date'], index_col='date')
    df = df[(df.index >= fromdate) & (df.index <= todate)]
    data = bt.feeds.PandasData(dataname=df, name=code)
    return data


def run_backtest(pool_file, topk, rebalance_day, stoploss):
    df = pd.read_csv(pool_file)
    etf_codes = df['基金代码'].tolist()
    start = datetime(2023, 1, 1)
    end = datetime(2025, 1, 1)

    cerebro = bt.Cerebro()
    cerebro.broker.setcash(1_000_000)
    cerebro.addanalyzer(bt.analyzers.PyFolio, _name='pyfolio')

    for code in etf_codes:
        file_path = f'data/{code}.csv'
        if os.path.exists(file_path):
            data = load_data_from_csv(code, start, end)
            cerebro.adddata(data)
        else:
            print(f"[⚠️] 缺少数据文件：{file_path}")

    # 加载沪深300指数作为benchmark
    benchmark_code = '000300.SH'
    benchmark_data = load_benchmark_data(benchmark_code, start, end)
    cerebro.adddata(benchmark_data)

    cerebro.addstrategy(
        MomentumStrategy,
        etf_list=etf_codes,
        rebalance_day=rebalance_day,
        topk=topk,
        stoploss=stoploss
    )

    print(f"\n🚀 启动回测（Top{topk}，调仓日：{rebalance_day}，止损：{stoploss*100:.1f}%）...\n")
    result = cerebro.run()
    strat = result[0]

    # 结果可视化
    cerebro.plot(style='candlestick', volume=False)

    # 输出分析指标
    portfolio_stats = strat.analyzers.getbyname('pyfolio').get_analysis()
    if portfolio_stats:
        returns = portfolio_stats['returns']
        print("\n✅ 回测完成，回报分析可通过 pyfolio 分析器查看（或后续导出）。")
