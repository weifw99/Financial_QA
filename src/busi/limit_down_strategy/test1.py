from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

from busi.limit_down_strategy.utils.data_loader import load_stock_data_df

# === 参数设置 ===
initial_cash = 1_000_000
commission_rate = 0.00025
min_commission = 5
max_stocks = 3
risk_free_rate = 0.03  # 年化无风险利率

cash = initial_cash
positions = {}  # {股票代码: 持仓数量}
portfolio_records = []

# === 假设 stock_list 已经存在，每个元素 {'code': '000001', 'data': df} ===

# === 股票数据 ===
# df = pd.read_csv("stock_data.csv", parse_dates=["date"])
from_idx = datetime(2025, 1, 1)  # 记录行情数据的开始时间和结束时间
to_idx = datetime(2025, 7, 23)

# from_idx = datetime(2014, 1, 1)  # 记录行情数据的开始时间和结束时间
# to_idx = datetime(2025, 6, 26)

print(from_idx, to_idx)
# 加载所有股票与指数数据
stock_list = load_stock_data_df(from_idx, to_idx)
for stock in stock_list:
    df = stock["data"]
    df['date1'] = pd.to_datetime(df['date1'])
    # 确保日期排序
    df = df.sort_values("date1").reset_index(drop=True)
    stock["data"] = df

    # 计算指标
    df["ma20"] = df["close"].rolling(20).mean()
    df["ma60"] = df["close"].rolling(60).mean()


    # 可选：计算 RSI
    def calc_rsi(series, period=14):
        delta = series.diff()
        gain = np.where(delta > 0, delta, 0)
        loss = np.where(delta < 0, -delta, 0)
        roll_up = pd.Series(gain).rolling(period).mean()
        roll_down = pd.Series(loss).rolling(period).mean()
        rs = roll_up / (roll_down + 1e-9)
        return 100 - (100 / (1 + rs))


    df["rsi"] = calc_rsi(df["close"])

# 收集所有交易日期
# all_dates = sorted({date for stock in stock_list for date in stock["data"]["date1"]})
all_dates = sorted({pd.to_datetime(date) for stock in stock_list for date in stock["data"]["date1"]})

trade_records = []  # 交易日志

# === 回测过程 ===
for date in all_dates:
    daily_candidates = []

    # --- 筛选候选股票 ---
    for stock in stock_list:
        code = stock["code"]
        df = stock["data"]
        row = df[df["date1"] == date]
        if row.empty: continue
        row = row.iloc[0]

        if row["is_st"] == 1: continue
        if row["amount"] < 1e8: continue
        if row["close"] <= row["ma60"]: continue
        daily_return = (row["close"] - row["open"]) / row["open"]
        if not (-0.095 <= daily_return <= -0.05): continue
        if np.isnan(row["ma20"]) or abs(row["close"] - row["ma20"]) / row["ma20"] > 0.05: continue

        daily_candidates.append((code, daily_return, row["volume"], row["close"]))

    # --- 候选排序取 topN ---
    daily_candidates = sorted(daily_candidates, key=lambda x: (-x[1], -x[2]))[:max_stocks]
    candidate_codes = {c[0] for c in daily_candidates}

    # === 收盘统一调仓 ===
    # 1. 卖出不在候选名单的持仓
    for code in list(positions.keys()):
        if code not in candidate_codes:
            stock = next((s for s in stock_list if s["code"] == code), None)
            if stock is None: continue
            df = stock["data"]
            row = df[df["date1"] == date]
            if row.empty: continue
            row = row.iloc[0]

            close_price = row["close"]
            qty = positions.pop(code)
            proceeds = qty * close_price
            fee = max(min_commission, proceeds * commission_rate)
            cash += proceeds - fee

            trade_records.append({
                "date": date,
                "code": code,
                "action": "SELL",
                "price": close_price,
                "qty": qty,
                "amount": proceeds,
                "fee": fee
            })

    # 2. 买入在候选名单里但不在持仓的股票
    available_cash = cash
    new_buys = [c for c in daily_candidates if c[0] not in positions]
    if new_buys:
        alloc_cash = available_cash / len(new_buys)

        for code, ret, vol, close_price in new_buys:
            qty = int(alloc_cash // close_price // 100) * 100
            if qty <= 0:
                continue

            cost = qty * close_price
            fee = max(min_commission, cost * commission_rate)
            total_cost = cost + fee
            if cash >= total_cost:
                cash -= total_cost
                positions[code] = qty

                trade_records.append({
                    "date": date,
                    "code": code,
                    "action": "BUY",
                    "price": close_price,
                    "qty": qty,
                    "amount": cost,
                    "fee": fee
                })

    # === 记录资金池 ===
    market_value = 0
    for code, qty in positions.items():
        stock = next((s for s in stock_list if s["code"] == code), None)
        if stock is None: continue
        df = stock["data"]
        row = df[df["date1"] == date]
        if not row.empty:
            price = row["close"].values[0]
            market_value += qty * price

    total_value = cash + market_value
    portfolio_records.append([date, cash, market_value, total_value])

# === 保存资金曲线和交易记录 ===
result = pd.DataFrame(portfolio_records, columns=["date1", "cash", "market_value", "total_value"])
print(result.tail())
result.to_csv("backtest_result.csv", index=False)

# === 保存交易记录 ===
trades = pd.DataFrame(trade_records)
trades.to_csv("trades.csv", index=False)

print("✅ 回测完成，结果已保存到 backtest_result.csv 和 trades.csv")


result.set_index("date1", inplace=True)

# === 基准指数 ===
benchmark = [ data["data"] for data in stock_list if data["code"] in ["sh.000300"] ] [0]

benchmark = benchmark.set_index("date1").sort_index()
benchmark["benchmark"] = benchmark["close"] / benchmark["close"].iloc[0] * initial_cash

# === 绘制资金曲线 ===
plt.figure(figsize=(12, 6))
plt.plot(result.index, result["total_value"], label="策略净值", linewidth=2)
plt.plot(benchmark.index, benchmark["benchmark"], label="沪深300", linewidth=2, linestyle="--")
plt.axhline(initial_cash, color="gray", linestyle="--", label="初始资金")
plt.title("策略净值 vs 基准指数")
plt.xlabel("日期")
plt.ylabel("账户总资产")
plt.legend()
plt.grid(True)
plt.show()

# === 绩效指标计算 ===
# 日收益率
result["daily_ret"] = result["total_value"].pct_change().fillna(0)
benchmark["daily_ret"] = benchmark["benchmark"].pct_change().fillna(0)

# 累计收益率
cumulative_return = result["total_value"].iloc[-1] / initial_cash - 1
annualized_return = (1 + cumulative_return) ** (252 / len(result)) - 1  # 252 个交易日
annualized_vol = result["daily_ret"].std() * np.sqrt(252)
max_drawdown = (result["total_value"] / result["total_value"].cummax() - 1).min()
sharpe_ratio = (annualized_return - risk_free_rate) / annualized_vol

print(f"累计收益率: {cumulative_return:.2%}")
print(f"年化收益率: {annualized_return:.2%}")
print(f"最大回撤: {max_drawdown:.2%}")
print(f"年化波动率: {annualized_vol:.2%}")
print(f"夏普比率: {sharpe_ratio:.2f}")
