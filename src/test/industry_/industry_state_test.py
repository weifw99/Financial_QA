import pandas as pd
import matplotlib.pyplot as plt


def get_daily_top_industries(df, top_n=5):
    df['日期'] = pd.to_datetime(df['日期'])
    grouped = df.groupby('日期')
    top_industry_list = []

    for date, daily_df in grouped:
        top_daily = daily_df.sort_values('主力净流入-净额', ascending=False).head(top_n)
        for _, row in top_daily.iterrows():
            top_industry_list.append({
                '日期': date,
                '行业名称': row['行业名称'],
                '主力净流入-净额': row['主力净流入-净额']
            })

    return pd.DataFrame(top_industry_list)


def identify_industry_switch(top_df, lookback_days=10):
    top_df = top_df.sort_values('日期')
    top_df['is_new'] = False

    history = {}
    for date in sorted(top_df['日期'].unique()):
        today_industries = top_df[top_df['日期'] == date]['行业名称'].tolist()

        new_entries = []
        for ind in today_industries:
            if ind not in history or (date - history[ind]).days > lookback_days:
                new_entries.append(ind)
            history[ind] = date

        top_df.loc[(top_df['日期'] == date) & (top_df['行业名称'].isin(new_entries)), 'is_new'] = True

    return top_df[top_df['is_new']]


def plot_industry_flow(df, industry_list):
    df['日期'] = pd.to_datetime(df['日期'])
    df_pivot = df[df['行业名称'].isin(industry_list)].pivot(index='日期', columns='行业名称', values='主力净流入-净额')
    df_pivot.plot(figsize=(14, 6), title='行业主力净流入趋势')
    plt.axhline(0, color='gray', linestyle='--')
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    base_path = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/industry"

    # 示例：加载行业资金数据
    df = pd.read_csv(f'{base_path}/industry_flow.csv')  # 包含日期、行业名称、主力净流入-净额

    # 计算每日资金流入前 N 的行业
    top_df = get_daily_top_industries(df, top_n=20)
    top_df.to_csv('top_industries.csv', index=False)

    # 识别轮换行业
    switch_df = identify_industry_switch(top_df, lookback_days=5)
    switch_df.to_csv('industry_switch.csv', index=False)

    # 画图观察几个典型行业的主力资金走势
    sample_industries = ['贵金属', '农药兽药', '有色金属']
    plot_industry_flow(df, sample_industries)