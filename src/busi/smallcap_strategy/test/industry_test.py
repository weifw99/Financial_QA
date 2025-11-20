import pandas as pd
import numpy as np

from busi.smallcap_strategy.utils.selected_industries_util import load_industry_price, load_industry_fundflow

base_price_path = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/industry/industry_price"
base_path = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/industry"
# 加载数据
df_price = load_industry_price(base_price_path)
df_flow = load_industry_fundflow(f'{base_path}/industry_flow.csv')

# ================
# 1. 数据预处理
# ================
dfp = df_price.copy()
dff = df_flow.copy()

dfp['日期'] = pd.to_datetime(dfp['日期'])
dff['日期'] = pd.to_datetime(dff['日期'])

dfp = dfp.sort_values(['行业代码', '日期'])
dff = dff.sort_values(['行业代码', '日期'])

# 合并行情 + 资金流
df = pd.merge(dfp, dff, on=['日期', '行业代码', '行业名称'], how='inner')
df = df.sort_values(['行业代码', '日期']).reset_index(drop=True)

# =========================
# 2. 计算行业 RPS（以55日为例）
# =========================
N = 55
df['return_N'] = df.groupby('行业代码')['收盘'].pct_change(N)
df['RPS'] = df.groupby('日期')['return_N'].rank(pct=True)

# =========================
# 3. 计算资金热度指标 hot_score
# =========================

print( df.columns)
print( df.head())
# 主力连续净流入天数（5 日窗口）
df['主力连续净流入天数'] = df.groupby('行业代码')['主力净流入-净额'].transform(
    lambda x: x.gt(0).rolling(5).sum()
)

# 超大单爆发度（相对20日均值）
df['超大单爆发'] = df.groupby('行业代码')['超大单净流入-净额'].transform(
    lambda x: x / x.rolling(20).mean()
)

# 主力占比变化
df['主力占比变化'] = df.groupby('行业代码')['主力净流入-净占比'].diff()

# 资金流 RPS
df['flow_RPS'] = df.groupby('日期')['主力净流入-净额'].rank(pct=True)

# 综合热度 hot_score
df['hot_score'] = (
    df['flow_RPS'] * 0.5 +
    (df['主力净流入-净额'] > 0).astype(int) * 0.2 +
    (df['超大单爆发'] > 2).astype(int) * 0.2 +
    (df['主力占比变化'] > 0).astype(int) * 0.1
)

# ================
# 4. 定义预测信号
# ================
df['signal_hot'] = (df['hot_score'] > 0.7).astype(int)
df['signal_rps'] = (df['RPS'] > 0.9).astype(int)

# ================
# 5. 未来收益验证（后验）
# ================
future_day = 5

df['future_ret'] = (
    df.groupby('行业代码')['收盘']
    .shift(-future_day) / df['收盘'] - 1
)

# ================
# 6. 评估 signal → future_ret
# ================

def evaluate_signal(df, signal_col):
    """统计某信号的后验表现"""
    if signal_col is None:
        # 自动组合信号
        sig = ((df['hot_score'] > 0.7) & (df['RPS'] > 0.9)).astype(int)
    else:
        sig = df[signal_col]

    sub = df[sig == 1]

    return {
        '样本数': len(sub),
        '胜率': (sub['future_ret'] > 0).mean(),
        '平均收益': sub['future_ret'].mean(),
        '平均上涨幅度': sub[sub['future_ret'] > 0]['future_ret'].mean(),
        '平均下跌幅度': sub[sub['future_ret'] <= 0]['future_ret'].mean(),
        '最大回撤': sub['future_ret'].min()
    }


result_hot = evaluate_signal(df, 'signal_hot')
result_rps = evaluate_signal(df, 'signal_rps')
# 组合信号：热度 + RPS 同时满足
df['signal_both'] = ((df['hot_score'] > 0.7) & (df['RPS'] > 0.9)).astype(int)
# 评估组合信号
result_both = evaluate_signal(df, 'signal_both')
# print(result_both)


print("🔥 热度信号表现：", result_hot)
print("📈 RPS 信号表现：", result_rps)
print("📈 both 信号表现：", result_both)
