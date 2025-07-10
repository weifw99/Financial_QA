import pandas as pd
import matplotlib.pyplot as plt

import pandas as pd

def load_industry_fundflow(filepath):
    df = pd.read_csv(filepath, encoding='utf-8')
    df['日期'] = pd.to_datetime(df['日期'])
    df.sort_values(by=['日期', '主力净流入-净额'], ascending=[True, False], inplace=True)
    return df

def strategy_A_avg_net_inflow(df, top_n=10, ma_window=5):
    # 🅰️ 策略 A：行业主力净流入 3 日均值 TopN
    df = df.copy()
    result = []

    for name, group in df.groupby('行业名称'):
        group = group.set_index('日期').sort_index()
        group[f'{ma_window}日均净流入'] = group['主力净流入-净额'].rolling(window=ma_window).mean()
        group['行业名称'] = name
        result.append(group.reset_index())

    merged = pd.concat(result)
    merged.sort_values(['日期', f'{ma_window}日均净流入'], ascending=[True, False], inplace=True)

    top_industries = merged.groupby('日期').head(top_n)
    return top_industries[['日期', '行业名称', f'{ma_window}日均净流入']]

def strategy_B_recent_topk(df, top_k=3, window=5):
    """
    策略 B：过去 N 天中入选 TopK 次数最多的行业。
    例如：今天是第6天，过去第1~5天，每天取净流入前top_k的行业，累计出现频次，排名靠前者作为今天的Top行业。

    参数：
        df: 包含 日期、行业名称、主力净流入-净额 的 DataFrame
        top_k: 每天取的前K行业
        window: 回顾的窗口长度（过去多少天）

    返回：
        DataFrame，列包含：日期、Top行业列表（list）
    """
    df = df.copy()
    df = df.sort_values(['日期', '主力净流入-净额'], ascending=[True, False])
    date_list = sorted(df['日期'].unique())

    # 每天 top_k 行业
    daily_top = (
        df.groupby('日期')
        .apply(lambda x: x.sort_values('主力净流入-净额', ascending=False).head(top_k)['行业名称'].tolist())
        .reset_index(name='Top行业')
    )

    result = []

    for i in range(window, len(daily_top)):
        today = daily_top.loc[i, '日期']
        past_top_list = daily_top.loc[i - window:i - 1, 'Top行业']
        flat_list = [item for sublist in past_top_list for item in sublist]

        counts = pd.Series(flat_list).value_counts()
        top_industries = counts.head(top_k).index.tolist()

        result.append({
            '日期': today,
            'Top行业': top_industries
        })

    return pd.DataFrame(result)


def strategy_C_avg_net_ratio(df, top_n=5, ma_window=10):
    # 🅲 策略 C：主力净占比 3 日均值 TopN
    df = df.copy()
    result = []

    for name, group in df.groupby('行业名称'):
        group = group.set_index('日期').sort_index()
        group[f'{ma_window}日均净占比'] = group['主力净流入-净占比'].rolling(window=ma_window).mean()
        group['行业名称'] = name
        result.append(group.reset_index())

    merged = pd.concat(result)
    merged.sort_values(['日期', f'{ma_window}日均净占比'], ascending=[True, False], inplace=True)

    top_industries = merged.groupby('日期').head(top_n)
    return top_industries[['日期', '行业名称', f'{ma_window}日均净占比']]

import matplotlib.pyplot as plt

def strategy_D_plot_industry_trend(df, industries, start_date=None, end_date=None):
    # 🅳 策略 D：主力净流入趋势分析（行业趋势可视化）
    df = df.copy()
    df = df[df['行业名称'].isin(industries)]
    df = df.sort_values(['行业名称', '日期'])

    if start_date:
        df = df[df['日期'] >= pd.to_datetime(start_date)]
    if end_date:
        df = df[df['日期'] <= pd.to_datetime(end_date)]

    plt.figure(figsize=(10, 6))
    for name, group in df.groupby('行业名称'):
        group = group.set_index('日期').sort_index()
        group['主力净流入-净额'].rolling(3).mean().plot(label=name)

    plt.title('主力净流入3日均值趋势图')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

if __name__ == '__main__':
    base_path = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/industry"

    # 加载数据
    df = load_industry_fundflow(f'{base_path}/industry_flow.csv')
    # 执行 A
    res_a = strategy_A_avg_net_inflow(df, top_n=5, ma_window=5)
    # 执行 B
    res_b = strategy_B_recent_topk(df)
    # 执行 C
    res_c = strategy_C_avg_net_ratio(df)
    # 执行 D（示例：选择前几名）
    strategy_D_plot_industry_trend(df, industries=['软件开发', '证券', '光伏设备'], start_date='2025-06-01')

    res_a.to_csv('result_strategy_A.csv', index=False)
    res_b.to_csv('result_strategy_B.csv', index=False)
    res_c.to_csv('result_strategy_C.csv', index=False)


