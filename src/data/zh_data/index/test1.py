"""
批量抓取 A 股十大流通股东数据，并统计社保持仓减持指标
数据来源: 东方财富 (https://data.eastmoney.com/gdfx/HoldingAnalyse.html)
依赖: akshare pandas openpyxl
安装: pip install akshare pandas openpyxl
"""

import akshare as ak
import pandas as pd
import re
import os
from datetime import datetime

def fetch_quarter_data(report_date: str, save_prefix: str = "社保持仓"):
    """抓取单季度数据，如果文件存在则直接读取"""
    excel_file = f"{save_prefix}_{report_date}.xlsx"
    if os.path.exists(excel_file):
        print(f"[跳过] {report_date} 已存在，直接读取")
        return pd.read_excel(excel_file)

    print(f"抓取 {report_date} 十大流通股东数据...")
    df = ak.stock_gdfx_free_holding_analyse_em(date=report_date)
    print(f"原始数据总条数: {len(df)}")

    # 匹配全国社保基金/社保基金/基本养老保险基金
    keyword_pattern = re.compile(r"(社保|全国社保|基本养老保险基金)", re.IGNORECASE)
    ss_df = df[df['股东名称'].str.contains(keyword_pattern, na=False)]

    # 去重（股票代码+股东名称）
    ss_df = ss_df.drop_duplicates(subset=['股票代码', '股东名称'])

    # 重命名字段
    ss_df = ss_df.rename(columns={
        '股票代码': '代码',
        '股票简称': '名称',
        '股东名称': '股东',
        '股东类型': '股东类型',
        '期末持股-数量': '持股数',
        '期末持股-数量变化': '持股变化',
        '期末持股-数量变化比例': '变化比例(%)',
        '期末持股-流通市值': '持股市值(元)',
        '公告日': '公告日'
    })

    # 保存单季度文件
    ss_df.to_excel(excel_file, index=False)
    print(f"[保存] {excel_file}，记录数: {len(ss_df)}")
    return ss_df

def calc_reduce_stats_per_stock(df, report_dates, last_n_quarters):
    """
    计算每个股票在最近 N 个季度的减持次数和减持占比
    df: 包含所有季度的社保持仓数据
    report_dates: 所有季度日期列表（按时间升序）
    last_n_quarters: 最近多少季度
    返回: DataFrame ['代码','减持次数','减持占比']
    """
    last_quarters = report_dates[-last_n_quarters:]
    df_last = df[df['季度'].isin(last_quarters)].copy()

    # 对每个股票统计减持次数
    reduce_count_df = df_last.groupby("代码")["减持标记"].sum().reset_index()
    reduce_count_df.rename(columns={"减持标记": f"最近{last_n_quarters}季度减持次数"}, inplace=True)

    # 减持占比 = 减持次数 / N
    reduce_count_df[f"最近{last_n_quarters}季度减持占比"] = reduce_count_df[f"最近{last_n_quarters}季度减持次数"] / last_n_quarters

    return reduce_count_df

def aggregate_multi_quarters(report_dates, save_prefix="社保持仓11"):
    """聚合多个季度的数据，计算减持次数和减持占比"""
    all_quarters = {}
    for d in report_dates:
        all_quarters[d] = fetch_quarter_data(d, save_prefix=save_prefix)

    # 合并所有季度数据
    combined_df = pd.concat(all_quarters.values(), keys=all_quarters.keys(), names=["季度", "行号"])
    combined_df = combined_df.reset_index(level=0).rename(columns={"level_0": "季度"})

    # 增加减持标记
    combined_df["减持标记"] = combined_df["变化比例(%)"].apply(lambda x: 1 if pd.notna(x) and x < 0 else 0)

    # 最新季度数据
    latest_quarter = report_dates[-1]
    latest_df = all_quarters[latest_quarter]

    # 聚合最新季度：持股市值、组合数
    agg_latest = latest_df.groupby("代码").agg({
        "持股市值(元)": "sum",
        "股东": "nunique"
    }).rename(columns={"股东": "社保组合数"}).reset_index()

    # 计算减持指标
    reduce_4_df = calc_reduce_stats_per_stock(combined_df, report_dates, 4)
    reduce_2_df = calc_reduce_stats_per_stock(combined_df, report_dates, 2)
    reduce_1_df = calc_reduce_stats_per_stock(combined_df, report_dates, 1)

    # 合并到最新季度表
    agg_latest = agg_latest.merge(reduce_4_df, on="代码", how="left")
    agg_latest = agg_latest.merge(reduce_2_df, on="代码", how="left")
    agg_latest = agg_latest.merge(reduce_1_df, on="代码", how="left")

    return agg_latest, combined_df

if __name__ == "__main__":
    # 选择要抓取的季度，按时间顺序排列
    quarters = ["20240930", "20241231", "20250331", "20250630"]

    final_df, raw_df = aggregate_multi_quarters(quarters)

    # 保存汇总结果
    today = datetime.today().strftime("%Y%m%d")
    final_df.to_excel(f"社保持仓汇总_减持指标_{today}.xlsx", index=False)
    print(f"汇总结果股票数量: {len(final_df)}")
    print(final_df.head())
