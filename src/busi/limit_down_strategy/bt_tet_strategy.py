# bt_tet_strategy.py
# 运行前: pip install backtrader pandas numpy
# 使用: python bt_tet_strategy.py
import os
import glob
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime

# ========== 配置 ==========
DATA_DIR = "results"           # 已有每股指标 CSV 的目录
CSV_PATTERN = "*_TET_full.csv" # 文件模式
SYMBOLS = None                 # None => 读取 DATA_DIR 下所有匹配文件；或指定 ['600519.SH','000001.SZ']
REBALANCE_DAYS = 5
TOP_N = 30
INITIAL_CASH = 1_000_000
COMMISSION = 0.0008            # 万8 单边比例
SLIPPAGE_PERC = 0.0005
OUTPUT_DIR = "bt_results"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ========== 自定义 PandasData 扩展以载入额外列 ==========
class PandasDataTET(bt.feeds.PandasData):
    # 下面列出你 CSV 中的列名，让 backtrader 把它们映射成 lines
    lines = ('trend','joint_trend','emotion','anchored','timing',
             'RoC_24','RoC_32','RoC_48','RoC_64','RoC_96','RoC_128','RoC_192','RoC_256','RoC_384','RoC_512',
             'SMA_24','SMA_32','SMA_48','SMA_64','SMA_96','SMA_128','SMA_192','SMA_256','SMA_384','SMA_512',
             'X_20_50','X_20_100','X_20_200','X_20_400','X_50_100','X_50_200','X_50_400','X_100_200','X_100_400','X_200_400',
             'LR_63','LR_84','LR_105','LR_126','LR_147','LR_168','LR_189','LR_252','LR_315','LR_378',
             'RSI_5','RSI_8','RSI_11','RSI_14','RSI_17','RSI_20',
             'CR_3','CR_6','CR_9','CR_12','CR_15','CR_18'
            )
    params = (
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),
        # map each extra column
        ('trend','trend'),('joint_trend','joint_trend'),('emotion','emotion'),('anchored','anchored'),('timing','timing'),
        ('RoC_24','RoC_24'),('RoC_32','RoC_32'),('RoC_48','RoC_48'),('RoC_64','RoC_64'),('RoC_96','RoC_96'),
        ('RoC_128','RoC_128'),('RoC_192','RoC_192'),('RoC_256','RoC_256'),('RoC_384','RoC_384'),('RoC_512','RoC_512'),
        ('SMA_24','SMA_24'),('SMA_32','SMA_32'),('SMA_48','SMA_48'),('SMA_64','SMA_64'),('SMA_96','SMA_96'),
        ('SMA_128','SMA_128'),('SMA_192','SMA_192'),('SMA_256','SMA_256'),('SMA_384','SMA_384'),('SMA_512','SMA_512'),
        ('X_20_50','X_20_50'),('X_20_100','X_20_100'),('X_20_200','X_20_200'),('X_20_400','X_20_400'),
        ('X_50_100','X_50_100'),('X_50_200','X_50_200'),('X_50_400','X_50_400'),('X_100_200','X_100_200'),
        ('X_100_400','X_100_400'),('X_200_400','X_200_400'),
        ('LR_63','LR_63'),('LR_84','LR_84'),('LR_105','LR_105'),('LR_126','LR_126'),('LR_147','LR_147'),
        ('LR_168','LR_168'),('LR_189','LR_189'),('LR_252','LR_252'),('LR_315','LR_315'),('LR_378','LR_378'),
        ('RSI_5','RSI_5'),('RSI_8','RSI_8'),('RSI_11','RSI_11'),('RSI_14','RSI_14'),('RSI_17','RSI_17'),('RSI_20','RSI_20'),
        ('CR_3','CR_3'),('CR_6','CR_6'),('CR_9','CR_9'),('CR_12','CR_12'),('CR_15','CR_15'),('CR_18','CR_18'),
    )

# ========== 策略 ==========
class TETStrategy(bt.Strategy):
    params = dict(
        rebalance_days=REBALANCE_DAYS,
        top_n=TOP_N
    )

    def __init__(self):
        self.datadict = {d._name:d for d in self.datas}  # name -> data
        self.days = 0
        self.order_records = []

    def next(self):
        self.days += 1
        if (self.days - 1) % self.p.rebalance_days != 0:
            return  # 非调仓日跳过

        dt = self.datas[0].datetime.date(0)
        # 收集每只股票当日的 timing
        timing_list = []
        for name, data in self.datadict.items():
            # 如果数据不足则忽略
            try:
                t = data.lines.timing[0]
            except Exception:
                t = None
            if t is None or np.isnan(t):
                continue
            timing_list.append((name, float(t)))
        if len(timing_list) == 0:
            return

        # 选 top-N by timing
        timing_list.sort(key=lambda x: x[1], reverse=True)
        selected = [x[0] for x in timing_list[:self.p.top_n]]

        # 建仓/调仓：把当前仓位调整成等权 selected；其余全部平
        # 1) 计算 target percents
        target_pct = {name: 0.0 for name in self.datadict.keys()}
        w = 1.0 / max(1, len(selected))
        for s in selected:
            target_pct[s] = w

        # 2) 对每个数据下达 order_target_percent
        for name, data in self.datadict.items():
            current_value = self.getposition(data).size * data.close[0] if data.close[0] > 0 else 0
            target = target_pct.get(name, 0.0)
            # backtrader expects order_target_percent with data arg
            self.order_target_percent(data=data, target=target)
            # 记录
            self.order_records.append({'date': dt, 'code': name, 'target_pct': target})

    def notify_order(self, order):
        if order.status in [order.Completed]:
            dt = self.datas[0].datetime.date(0)
            symbol = order.data._name
            side = 'BUY' if order.isbuy() else 'SELL'
            logging_line = {'date': dt, 'symbol': symbol, 'side': side,
                            'size': order.executed.size, 'price': order.executed.price,
                            'value': order.executed.value}
            self.order_records.append(logging_line)

# ========== 主流程 ==========
def run_backtest():
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION)
    cerebro.broker.set_slippage_perc(SLIPPAGE_PERC)

    # load files
    files = glob.glob(os.path.join(DATA_DIR, CSV_PATTERN))
    symbols = []
    for f in files:
        basename = os.path.basename(f)
        code = basename.split("_TET_full")[0]
        symbols.append((code, f))
    if SYMBOLS:
        symbols = [s for s in symbols if s[0] in SYMBOLS]

    # add datafeeds
    for code, fp in symbols:
        df = pd.read_csv(fp, parse_dates=['date'], index_col='date')
        # ensure ascending index
        df = df.sort_index()
        # remove rows with nonpositive close
        df = df[df['close'] > 0]
        if df.empty:
            continue
        datafeed = PandasDataTET(dataname=df, name=code)
        cerebro.adddata(datafeed)

    cerebro.addstrategy(TETStrategy, rebalance_days=REBALANCE_DAYS, top_n=TOP_N)

    print("Starting Portfolio Value:", cerebro.broker.getvalue())
    results = cerebro.run()
    print("Final Portfolio Value:", cerebro.broker.getvalue())

    # 导出净值（用 observer 的方式较复杂，这里用简单回放：broker.pnl not directly available）
    # 我推荐用户在策略中把每天的净值写入一个全局 list，但为简单起见，这里我们 rerun with analyzers
    # 使用 TimeReturn analyzer
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION)
    cerebro.broker.set_slippage_perc(SLIPPAGE_PERC)
    for code, fp in symbols:
        df = pd.read_csv(fp, parse_dates=['date'], index_col='date').sort_index()
        df = df[df['close'] > 0]
        if df.empty: continue
        datafeed = PandasDataTET(dataname=df, name=code)
        cerebro.adddata(datafeed)
    cerebro.addstrategy(TETStrategy, rebalance_days=REBALANCE_DAYS, top_n=TOP_N)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timeret')
    res = cerebro.run()
    analyzer = res[0].analyzers.timeret.get_analysis()
    # analyzer: dict date->return
    tr = pd.Series(analyzer).sort_index()
    cum = (1 + tr).cumprod()
    cum.to_csv(os.path.join(OUTPUT_DIR, "portfolio_equity.csv"), header=['cum_ret'])
    # orders_log: try to collect from strategy instances
    orders = []
    for r in res:
        strat = r
        if hasattr(strat, 'order_records'):
            orders.extend(strat.order_records)
    pd.DataFrame(orders).to_csv(os.path.join(OUTPUT_DIR, "orders_log.csv"), index=False)
    print("Saved equity and orders to", OUTPUT_DIR)

if __name__ == "__main__":
    run_backtest()