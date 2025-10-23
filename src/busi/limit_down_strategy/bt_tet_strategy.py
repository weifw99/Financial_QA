# filename: bt_tet_strategy_pro.py
# 功能：使用 Backtrader 回测 Trend-Emotion-Timing (TET) 策略，支持基本面过滤与动态调仓
# 运行: python bt_tet_strategy_pro.py

import os
import glob
import pandas as pd
import numpy as np
import backtrader as bt
from datetime import datetime, date, timedelta

# ================= 参数配置 =================
DATA_DIR = "results"
OUTPUT_DIR = "bt_results"
CSV_PATTERN = "*_TET_full.csv"
REBALANCE_DAYS = 5
INITIAL_CASH = 1_000_000
COMMISSION = 0.0008
SLIPPAGE_PERC = 0.0005
TOP_N = 20
MIN_MV = 10e8
MIN_REVENUE = 1e8
MIN_ROE = 0
MAX_PRICE = 100
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ================= 自定义 PandasData =================
class PandasDataTET(bt.feeds.PandasData):
    lines = ('mv','lt_mv','roeAvg_y','profit_y','revenue_y','is_st',
             'trend','joint_trend','emotion','anchored','timing')
    params = (
        ('datetime', None),
        ('open','open'),
        ('high','high'),
        ('low','low'),
        ('close','close'),
        ('volume','volume'),
        ('mv','mv'),
        ('lt_mv','lt_mv'),
        ('roeAvg_y','roeAvg_y'),
        ('profit_y','profit_y'),
        ('revenue_y','revenue_y'),
        ('is_st','is_st'),
        ('trend','trend'),
        ('joint_trend','joint_trend'),
        ('emotion','emotion'),
        ('anchored','anchored'),
        ('timing','timing'),
    )

# ================= 策略定义 =================
class TETFilterRebalanceStrategy(bt.Strategy):
    params = dict(
        rebalance_days=REBALANCE_DAYS,
        top_n=TOP_N,
        min_mv=MIN_MV,
        min_revenue=MIN_REVENUE,
        min_roe=MIN_ROE,
        max_price=MAX_PRICE,
        cash_per_stock_ratio=1.0 / TOP_N
    )

    def __init__(self):
        self.datadict = {d._name: d for d in self.datas}
        self.days = 0
        self.to_buy = []
        self.order_records = []

    def log(self, txt):
        dt = self.datas[0].datetime.date(0)
        print(f"{dt} | {txt}")

    # 选股过滤器
    def filter_stocks(self):
        filtered = []
        for name, d in self.datadict.items():
            try:
                if d.is_st[0] == 1:  # ST股过滤
                    continue
                if np.isnan(d.timing[0]) or np.isnan(d.close[0]):
                    continue
                if d.lt_mv[0] < self.p.min_mv:
                    continue
                if d.revenue_y[0] < self.p.min_revenue:
                    continue
                if d.roeAvg_y[0] < self.p.min_roe:
                    continue
                if d.close[0] <= 1 or d.close[0] > self.p.max_price:
                    continue
                filtered.append((name, float(d.timing[0])))
            except Exception as e:
                continue

        # 按 timing 排序选 top-N
        filtered.sort(key=lambda x: x[1], reverse=True)
        selected = [x[0] for x in filtered[:self.p.top_n]]
        return selected

    def next(self):
        self.days += 1
        if (self.days - 1) % self.p.rebalance_days != 0:
            return  # 非调仓日跳过

        dt = self.datas[0].datetime.date(0)
        self.log(f"调仓日 {dt}")

        selected = self.filter_stocks()
        if not selected:
            self.log("❌ 无可交易股票，清空所有持仓")
            for data in self.datas:
                pos = self.getposition(data)
                if pos.size > 0:
                    self.close(data)
            return

        hold_now = {d for d, pos in self.positions.items() if pos.size > 0}
        to_sell = hold_now - {self.getdatabyname(s) for s in selected}
        to_buy = [self.getdatabyname(s) for s in selected if s not in [d._name for d in hold_now]]

        # 卖出不在名单内的股票
        for d in to_sell:
            self.log(f"💸 卖出 {d._name}")
            self.close(d)

        # 买入新股票
        if to_buy:
            total_cash = self.broker.getcash()
            cash_per_stock = total_cash * self.p.cash_per_stock_ratio
            for d in to_buy:
                price = d.close[0]
                if price <= 0 or np.isnan(price):
                    continue
                size = int(cash_per_stock // price // 100 * 100)
                if size >= 100:
                    self.buy(data=d, size=size)
                    self.log(f"🟢 买入 {d._name} size={size}")
                    self.order_records.append({'date': dt, 'code': d._name, 'action': 'buy', 'price': price, 'size': size})
                else:
                    self.log(f"⚠️ {d._name} 资金不足跳过")

    def stop(self):
        value = self.broker.getvalue()
        print(f"回测结束 | 最终净值: {value:,.2f}")
        pd.DataFrame(self.order_records).to_csv(os.path.join(OUTPUT_DIR, "orders_log.csv"), index=False)
        print("已保存交易记录到 orders_log.csv")

# ================= 主流程 =================
def run_backtest():
    cerebro = bt.Cerebro()
    cerebro.broker.setcash(INITIAL_CASH)
    cerebro.broker.setcommission(commission=COMMISSION)
    cerebro.broker.set_slippage_perc(SLIPPAGE_PERC)

    files = glob.glob(os.path.join(DATA_DIR, CSV_PATTERN))
    for f in files:
        print(f"加载文件 {f}")
        code = os.path.basename(f).split("_TET_full")[0]
        df = pd.read_csv(f, parse_dates=['date'], index_col='date').sort_index()
        if df.empty or 'close' not in df.columns:
            continue
        for col in ['open','high','low','close','volume']:
            if col not in df.columns:
                df[col] = df['close']
        data = PandasDataTET(dataname=df, name=code)
        cerebro.adddata(data)

    cerebro.addstrategy(TETFilterRebalanceStrategy)
    cerebro.addanalyzer(bt.analyzers.TimeReturn, _name='timeret')
    print("初始资金:", INITIAL_CASH)

    result = cerebro.run()
    strat = result[0]
    analyzer = strat.analyzers.timeret.get_analysis()
    tr = pd.Series(analyzer).sort_index()
    cum = (1 + tr).cumprod()
    cum.to_csv(os.path.join(OUTPUT_DIR, "portfolio_equity.csv"), header=['cum_ret'])
    print("✅ 已保存净值文件到 portfolio_equity.csv")

if __name__ == "__main__":
    run_backtest()