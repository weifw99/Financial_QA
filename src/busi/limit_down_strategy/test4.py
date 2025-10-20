"""
Trend-Emotion-Timing 策略 (Reiss, 2025)
可切换：离散投票模式 / 连续趋势模式
支持 A股数据结构 [{'code': code, 'data': df}, ...]
"""

import os
import logging
from datetime import datetime
import pandas as pd
import numpy as np
from scipy import stats
from busi.limit_down_strategy.utils.data_loader import load_stock_data_df

# ========= 参数 =========
REBALANCE_DAYS = 5
TOP_N = 30
TRANSACTION_COST = 0.0008
SLIPPAGE = 0.0005
SAVE_FULL_DETAIL = True
USE_CONTINUOUS_TREND = False   # 🔁 True = 连续趋势模式；False = 离散投票模式
USE_CONTINUOUS_TREND = True   # 🔁 True = 连续趋势模式；False = 离散投票模式
MAX_STOCKS = 50

# ========= 日志配置 =========
os.makedirs("results", exist_ok=True)
log_filename = f"trend_emotion_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log"
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[
        logging.FileHandler(log_filename, mode='w', encoding='utf-8'),
        logging.StreamHandler()
    ]
)
logging.info(f"📋 日志记录至 {log_filename}")
logging.info(f"运行模式：{'连续趋势' if USE_CONTINUOUS_TREND else '离散投票'}")

# ========= 基础函数 =========
def roc(series, n): return (series / series.shift(n) - 1)
def sma(series, n): return series.rolling(window=n, min_periods=1).mean()

def linear_regression_slope(series, window_days):
    def slope(x):
        if np.isnan(x).any() or len(x) < 3: return np.nan
        y = np.array(x); xs = np.arange(len(y))
        slope, _, _, _, _ = stats.linregress(xs, y)
        return slope
    return series.rolling(window=window_days, min_periods=3).apply(slope, raw=True)

def rsi(series, n):
    delta = series.diff()
    up = delta.clip(lower=0).rolling(n).mean()
    down = -delta.clip(upper=0).rolling(n).mean()
    rs = up / (down + 1e-12)
    return 100 - (100 / (1 + rs))

def candle_range_index(close, high, low, n):
    high_n = high.rolling(window=n, min_periods=1).max()
    low_n = low.rolling(window=n, min_periods=1).min()
    pos = (close - low_n) / (high_n - low_n + 1e-12)
    return 2.0 * pos - 1.0

# ========= 趋势指标 =========
def compute_trend_score(df):
    close = df['close']
    roc_periods = [24,32,48,64,96,128,192,256,384,512]
    sma_periods = roc_periods
    crossover_pairs = [(20,50),(20,100),(20,200),(20,400),
                       (50,100),(50,200),(50,400),(100,200),(100,400),(200,400)]
    lr_months = [3,4,5,6,7,8,9,12,15,18]
    lr_days = [int(m*21) for m in lr_months]

    components = {}

    # === RoC ===
    for n in roc_periods:
        if USE_CONTINUOUS_TREND:
            val = (close / close.shift(n) - 1).clip(-0.1, 0.1) * 10  # normalize to [-1,1]
        else:
            val = np.sign(close - close.shift(n)).fillna(0)
        components[f"RoC_{n}"] = val

    # === SMA ===
    for n in sma_periods:
        s = sma(close, n)
        if USE_CONTINUOUS_TREND:
            val = ((close - s) / s).clip(-0.1, 0.1) * 10
        else:
            val = np.sign(close - s).fillna(0)
        components[f"SMA_{n}"] = val

    # === Crossovers ===
    for (s_short, s_long) in crossover_pairs:
        s1, s2 = sma(close, s_short), sma(close, s_long)
        if USE_CONTINUOUS_TREND:
            val = ((s1 - s2) / s2).clip(-0.1, 0.1) * 10
        else:
            val = np.sign(s1 - s2).fillna(0)
        components[f"X_{s_short}_{s_long}"] = val

    # === Linear Regression ===
    for w in lr_days:
        slope = linear_regression_slope(close, w)
        if USE_CONTINUOUS_TREND:
            vol = close.pct_change().rolling(w).std()
            val = (slope / (vol + 1e-8)).clip(-1, 1)
        else:
            val = slope.apply(lambda x: 1 if x>1e-8 else (-1 if x<-1e-8 else 0)).fillna(0)
        components[f"LR_{w}"] = val

    votes_df = pd.DataFrame(components)
    trend_score = votes_df.mean(axis=1)
    return trend_score, votes_df

# ========= 情绪指标 =========
def compute_emotion_index(df):
    close, high, low = df['close'], df['high'], df['low']
    rsi_periods = [5,8,11,14,17,20]
    cr_periods = [3,6,9,12,15,18]

    components = {}
    for n in rsi_periods:
        r = rsi(close, n)
        components[f"RSI_{n}"] = 2.0*(r/100.0)-1.0
    for n in cr_periods:
        components[f"CR_{n}"] = candle_range_index(close, high, low, n)

    osc_df = pd.DataFrame(components)
    emotion = osc_df.mean(axis=1)
    return emotion, osc_df

# ========= 锚定趋势 =========
def compute_anchored_trend(trend_score, emotion):
    anchored = pd.Series(index=trend_score.index, dtype=float)
    prev_anchor = trend_score.iloc[0]
    prev_sign = np.sign(emotion.iloc[0]) if not np.isnan(emotion.iloc[0]) else 0
    anchored.iloc[0] = prev_anchor
    for i in range(1, len(trend_score)):
        cur_sign = np.sign(emotion.iloc[i]) if not np.isnan(emotion.iloc[i]) else 0
        if cur_sign != prev_sign and cur_sign != 0:
            prev_anchor = trend_score.iloc[i]; prev_sign = cur_sign
        anchored.iloc[i] = prev_anchor
    return anchored

# ========= 单股计算 =========
def process_single_stock(code, df, index_close=None):
    try:
        logging.info(f"▶ 处理 {code} 数据行数={len(df)}")
        if df is None or df.empty:
            logging.warning(f"⚠️ {code} 数据为空，跳过。"); return None
        if not all(col in df.columns for col in ['open','high','low','close']):
            logging.warning(f"⚠️ {code} 缺少关键列，跳过。"); return None

        # df = df.copy().sort_values('date')
        # df.index = pd.to_datetime(df['date'])
        df = df[['open','high','low','close','volume']].dropna()

        trend_score, trend_components = compute_trend_score(df)

        if index_close is not None:
            idx_aligned = index_close.reindex(df.index).fillna(method='ffill')
            ratio_close = df['close'] / idx_aligned
            ratio_df = pd.DataFrame({
                'close': ratio_close,
                'high': df['high']/idx_aligned,
                'low': df['low']/idx_aligned
            })
            ratio_trend, _ = compute_trend_score(ratio_df)
            joint = [np.sign(t)*min(abs(t),abs(r)) if np.sign(t)==np.sign(r)!=0 else 0.0 for t,r in zip(trend_score, ratio_trend)]
            joint = pd.Series(joint, index=trend_score.index)
        else:
            joint = trend_score

        emotion, emotion_components = compute_emotion_index(df)
        anchored = compute_anchored_trend(joint, emotion)
        timing = anchored - emotion

        main_df = pd.DataFrame({
            'trend': trend_score, 'joint_trend': joint,
            'emotion': emotion, 'anchored': anchored,
            'timing': timing, 'close': df['close']
        }, index=df.index)

        if SAVE_FULL_DETAIL:
            full_df = pd.concat([main_df, trend_components, emotion_components], axis=1)
        else:
            full_df = main_df

        filepath = os.path.join("results", f"{code}_TET_full.csv")
        full_df.to_csv(filepath)
        logging.info(f"💾 已保存 {code} 指标结果至 {filepath} ({len(full_df)} 行, {len(full_df.columns)} 列)")
        return {'code': code, 'result': main_df}
    except Exception as e:
        logging.error(f"❌ {code} 计算出错: {e}", exc_info=True)
        return None

# ========= 回测函数 =========
def backtest(stocks_processed, rebalance_days=REBALANCE_DAYS, top_n=TOP_N):
    logging.info("开始回测 ...")
    all_codes, dfs = [], {}
    for s in stocks_processed:
        if not s or 'result' not in s: continue
        df = s['result']
        if df is None or df.empty:
            logging.warning(f"⚠️ {s.get('code')} 无有效数据，跳过。"); continue
        all_codes.append(s['code']); dfs[s['code']] = df

    if len(dfs) == 0:
        raise ValueError("❌ 没有有效的股票结果可用于回测。")

    all_dates = sorted(list(set().union(*[df.index for df in dfs.values()])))
    timing_df = pd.DataFrame({c: dfs[c]['timing'] for c in all_codes})
    close_df = pd.DataFrame({c: dfs[c]['close'] for c in all_codes})

    positions = pd.DataFrame(0.0, index=all_dates, columns=all_codes)
    for i in range(0, len(all_dates), rebalance_days):
        date = all_dates[i]
        row = timing_df.loc[date].dropna()
        if row.empty: continue
        top = row.sort_values(ascending=False).head(top_n).index
        w = 1.0 / len(top)
        positions.loc[date, top] = w
    positions = positions.replace(0.0, np.nan).ffill().fillna(0.0)

    ret = close_df.pct_change().replace([np.inf, -np.inf], 0).fillna(0)
    port_ret = (positions.shift(1) * ret).sum(axis=1)
    tx_cost = positions.diff().abs().sum(axis=1) * TRANSACTION_COST
    net_ret = port_ret - tx_cost - SLIPPAGE

    large_moves = net_ret[(net_ret > 0.5) | (net_ret < -0.5)]
    if not large_moves.empty:
        logging.warning(f"⚠️ 检测到 {len(large_moves)} 天异常收益，将截断。")
        for dt, val in large_moves.items():
            logging.warning(f"    异常收益 {dt.date()} = {val:.2%}")
        net_ret = net_ret.clip(-0.5, 0.5)

    net_ret = net_ret.replace([np.inf, -np.inf], 0).fillna(0)
    cum = (1 + net_ret).cumprod()
    if np.isinf(cum).any():
        logging.error("❌ 检测到 cum_ret 出现 inf，自动修正")
        cum = cum.replace([np.inf, -np.inf], np.nan).fillna(method='ffill').fillna(method='bfill')
        cum = cum / cum.iloc[0]

    pd.DataFrame({'cum_ret': cum}).to_csv("portfolio_cum_ret.csv")
    net_ret.to_csv("portfolio_daily_ret.csv", header=['daily_ret'])
    positions.to_csv("positions.csv")
    logging.info("📁 已保存 portfolio_cum_ret.csv / portfolio_daily_ret.csv / positions.csv")
    logging.info(f"✅ 回测完成，总天数={len(all_dates)}，最终累计收益={cum.iloc[-1]:.3f}")
    return {'daily_ret': net_ret, 'cum_ret': cum, 'positions': positions}

# ========= 主流程 =========
if __name__ == "__main__":
    from_idx = datetime(2015, 1, 1)
    to_idx = datetime(2025, 7, 23)
    index_code = 'sh.000300'
    logging.info(f"加载数据范围: {from_idx.date()} - {to_idx.date()}")

    stocks_raw = load_stock_data_df(from_idx, to_idx)
    stocks, index_series = [], None
    for stock in stocks_raw:
        if stock["code"] == index_code:
            index_series = stock["data"]['close']
        else:
            stocks.append(stock)

    logging.info(f"股票数量={len(stocks)}, 指数数据={'已加载' if index_series is not None else '无'}")

    results = []
    for s in stocks[:MAX_STOCKS]:
        res = process_single_stock(s['code'], s['data'], index_close=index_series)
        if res: results.append(res)

    bt = backtest(results)

    # 汇总输出
    summary = []
    for r in results:
        df = r['result']
        last = df.iloc[-1]
        summary.append({
            'code': r['code'],
            'final_timing': last['timing'],
            'final_trend': last['trend'],
            'final_emotion': last['emotion'],
            'last_close': last['close']
        })
    pd.DataFrame(summary).to_csv("trend_emotion_results_summary.csv", index=False)
    logging.info("📁 已保存 trend_emotion_results_summary.csv")