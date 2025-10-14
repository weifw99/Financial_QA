"""
=========================================================
  小市值指数构建器（Small Cap Index Builder）
  ------------------------------------------------------
  功能：
    - 从 A 股日行情数据（dict结构）构建小市值指数
    - 自动过滤 ST、流动性差、市值过小股票
    - 支持等权/市值加权
    - 支持自定义调仓周期
    - 自动输出调仓记录与可视化结果
=========================================================
"""
from datetime import datetime

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os
from typing import Dict, Tuple

from busi.smallcap_strategy.task.data_loader import load_recent_data


def build_smallcap_index(
    data_dict: Dict[str, pd.DataFrame],
    rebalance_freq: str = 'M',
    n_small: int = 300,
    weight_type: str = 'equal',
    base_value: float = 1000.0,
    compare_index_dict: Dict[str, pd.DataFrame] = None,
    liquidity_window: int = 20,
    min_turnover: float = 1e7,
    min_mv: float = 1e9,
    save_path: str = "./output"
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    构建小市值指数
    --------------------------------
    参数：
        data_dict: dict, key=股票代码, value=DataFrame（含 date, mv, close, volume, amount, is_st 等）
        rebalance_freq: 调仓频率 ('M' 月度 / 'Q' 季度)
        n_small: 每期选取的小市值股票数量
        weight_type: 'equal' 或 'mv'
        base_value: 指数基期
        compare_index_dict: 可选，用于绘图对比 {name: DataFrame[date, close]}
        liquidity_window: int，流动性计算窗口
        min_turnover: float，近20日平均成交额下限（元）
        min_mv: float，市值下限（元）
        save_path: 输出结果保存目录
    返回：
        index_df: DataFrame，指数行情
        rebalance_df: DataFrame，调仓记录
    """

    os.makedirs(save_path, exist_ok=True)

    # ========== 1️⃣ 整理输入数据 ==========
    all_dates = sorted({d for df in data_dict.values() for d in df['date']})
    all_dates = pd.to_datetime(all_dates)
    base_df = pd.DataFrame(index=sorted(all_dates))

    mv_df, close_df, open_df, high_df, low_df = [base_df.copy() for _ in range(5)]
    volume_df, amount_df, turn_df, st_df = [base_df.copy() for _ in range(4)]

    for code, df in data_dict.items():
        df = df.copy()
        df['date'] = pd.to_datetime(df['date'])
        df = df.set_index('date').sort_index()
        mv_df[code] = df['mv']
        close_df[code] = df['close']
        open_df[code] = df['open']
        high_df[code] = df['high']
        low_df[code] = df['low']
        volume_df[code] = df['volume']
        amount_df[code] = df['amount']
        turn_df[code] = df['turn']
        st_df[code] = df['is_st']

    # ========== 2️⃣ 确定调仓日期 ==========
    rebalance_dates = mv_df.resample(rebalance_freq).last().index
    index_ohlc = pd.DataFrame(index=base_df.index)
    rebalance_records = []

    # ========== 3️⃣ 循环调仓 ==========
    for i, date in enumerate(rebalance_dates):
        if date not in mv_df.index:
            continue

        mv_today = mv_df.loc[date]
        valid_codes = mv_today.dropna().index.tolist()
        if len(valid_codes) < n_small:
            continue

        # -------- 🔍 多重过滤 --------
        valid = pd.Series(True, index=valid_codes)

        # ① ST过滤
        st_today = st_df.loc[date, valid_codes]
        valid &= (st_today == 0)

        # ② 市值过滤
        mv_today = mv_today.loc[valid.index]
        valid &= (mv_today > min_mv)

        # ③ 流动性过滤
        date_idx = amount_df.index.get_loc(date)
        if date_idx >= liquidity_window:
            recent_window = amount_df.iloc[date_idx - liquidity_window:date_idx]
            mean_turn = recent_window.mean()
            valid &= (mean_turn > min_turnover)

        valid_codes = [c for c, ok in valid.items() if ok]
        if len(valid_codes) < n_small:
            continue

        # -------- 📊 选取最小市值股票 --------
        mv_filtered = mv_today.loc[valid_codes]
        smallcaps = mv_filtered.nsmallest(n_small).index.tolist()

        # 保存调仓信息
        rebalance_records.append({
            "date": date,
            "n_stocks": len(smallcaps),
            "stocks": smallcaps
        })

        # -------- ⚖️ 计算区间内权重 --------
        if i < len(rebalance_dates) - 1:
            next_date = rebalance_dates[i + 1]
            mask = (index_ohlc.index > date) & (index_ohlc.index <= next_date)
        else:
            mask = (index_ohlc.index > date)
        period_dates = index_ohlc.index[mask]
        if len(period_dates) == 0:
            continue

        if weight_type == 'equal':
            weights = pd.Series(1 / len(smallcaps), index=smallcaps)
        else:
            weights = mv_filtered.loc[smallcaps] / mv_filtered.loc[smallcaps].sum()

        # -------- 🧮 计算加权OHLC --------
        index_ohlc.loc[period_dates, 'open'] = open_df.loc[period_dates, smallcaps].mul(weights, axis=1).sum(axis=1)
        index_ohlc.loc[period_dates, 'high'] = high_df.loc[period_dates, smallcaps].mul(weights, axis=1).sum(axis=1)
        index_ohlc.loc[period_dates, 'low'] = low_df.loc[period_dates, smallcaps].mul(weights, axis=1).sum(axis=1)
        index_ohlc.loc[period_dates, 'close'] = close_df.loc[period_dates, smallcaps].mul(weights, axis=1).sum(axis=1)
        index_ohlc.loc[period_dates, 'volume'] = volume_df.loc[period_dates, smallcaps].sum(axis=1)
        index_ohlc.loc[period_dates, 'amount'] = amount_df.loc[period_dates, smallcaps].sum(axis=1)
        index_ohlc.loc[period_dates, 'turn'] = turn_df.loc[period_dates, smallcaps].mean(axis=1)

    # ========== 4️⃣ 计算指数 ==========
    index_ohlc = index_ohlc.dropna(subset=['close'])
    index_ohlc['pctChg'] = index_ohlc['close'].pct_change().fillna(0)
    index_ohlc['index'] = (1 + index_ohlc['pctChg']).cumprod() * base_value
    index_ohlc = index_ohlc.reset_index().rename(columns={'index': 'date'})

    rebalance_df = pd.DataFrame(rebalance_records)

    # ========== 5️⃣ 绘图比较 ==========
    if compare_index_dict:
        plt.figure(figsize=(12,6))
        plt.plot(index_ohlc['date'], index_ohlc['index'], label='SmallCap_Custom', linewidth=2)
        for name, idx_df in compare_index_dict.items():
            idx_df = idx_df.copy()
            idx_df['date'] = pd.to_datetime(idx_df['date'])
            idx_df = idx_df.set_index('date').sort_index()
            idx_df['index'] = idx_df['close'] / idx_df['close'].iloc[0] * base_value
            plt.plot(idx_df.index, idx_df['index'], label=name)
        plt.legend()
        plt.title("小市值指数 vs 基准指数")
        plt.grid(True)
        plt.show()

    # ========== 6️⃣ 保存文件 ==========
    index_path = os.path.join(save_path, "smallcap_index.csv")
    rebalance_path = os.path.join(save_path, "rebalance_log.csv")
    index_ohlc.to_csv(index_path, index=False)
    rebalance_df.to_csv(rebalance_path, index=False)
    print(f"✅ 指数数据已保存：{index_path}")
    print(f"✅ 调仓记录已保存：{rebalance_path}")

    return index_ohlc, rebalance_df


if __name__ == "__main__":
    # 1. 加载最近30日的数据（指数 + 个股）
    today = datetime.today()
    stock_data_dict, data_date = load_recent_data()

    print("该文件为模块文件，请在项目中 import 使用。示例：")
    print("from smallcap_index_builder import build_smallcap_index")