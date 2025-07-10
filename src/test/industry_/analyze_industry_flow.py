import pandas as pd
import matplotlib.pyplot as plt
from matplotlib import font_manager


import matplotlib
print('path: ', matplotlib.matplotlib_fname())


# 指定字体文件路径
# font_path = "/System/Library/Fonts/STHeiti Medium.ttc"
# my_font = font_manager.FontProperties(fname=font_path)
# 
# plt.rcParams['font.family'] = 'STHeiti Medium'
# plt.rcParams['axes.unicode_minus'] = False  # 正确显示负号

def analyze_industry_flow(df: pd.DataFrame, target_industry=None, days=20):
    """
    分析行业资金流入趋势和当前强势行业
    参数:
        df: 行业资金流数据（含主力/中小单净额）
        target_industry: 若指定，则绘制该行业资金流图
        days: 趋势分析使用的窗口天数
    """
    df['日期'] = pd.to_datetime(df['日期'])
    df = df.sort_values(['行业名称', '日期']).copy()

    # 计算散户净流入
    df['散户净流入'] = df['中单净流入-净额'] + df['小单净流入-净额']

    # 分析目标行业资金趋势
    if target_industry:
        plot_industry_trend(df, target_industry, days)

    # 筛选今日主力流入最多行业
    latest_date = df['日期'].max()
    today_df = df[df['日期'] == latest_date]
    top_main_inflow = today_df.sort_values(by='主力净流入-净额', ascending=False).head(10)

    print(f"\n📌 {latest_date.strftime('%Y-%m-%d')} 主力资金净流入Top10行业：")
    print(top_main_inflow[['行业名称', '主力净流入-净额', '主力净流入-净占比']])

    return top_main_inflow


def plot_industry_trend(df, industry, days):
    """
    可视化资金流趋势
    """
    df_ind = df[df['行业名称'] == industry].set_index('日期').tail(days)

    plt.figure(figsize=(12, 6))
    plt.plot(df_ind.index, df_ind['主力净流入-净额'], label='主力净流入', color='red', )
    plt.plot(df_ind.index, df_ind['散户净流入'], label='散户净流入', color='green', )
    plt.title(f'{industry} 行业资金流趋势（近{days}日）', )
    plt.xlabel('日期', )
    plt.ylabel('资金（元）', )
    plt.axhline(0, color='black', linestyle='--', linewidth=0.8)
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()


if __name__ == '__main__':
    base_path = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/industry"
    df = pd.read_csv(f"{base_path}/industry_flow.csv")  # 你的行业资金数据
    top10 = analyze_industry_flow(df, target_industry='文化传媒', days=30)

