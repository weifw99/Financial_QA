"""
批量抓取 A 股十大流通股东数据，并筛选社保/养老基金持仓
数据来源: 东方财富 (https://data.eastmoney.com/gdfx/HoldingAnalyse.html)
依赖: akshare pandas openpyxl
安装: pip install akshare pandas openpyxl
"""

import akshare as ak
import pandas as pd
import re


def fetch_social_security_holdings(report_date: str, save_prefix: str = "社保持仓"):
    """
    获取指定财报日的所有股票十大流通股东，并筛选包含社保/养老基金的记录

    :param report_date: str, 报告期，例如 "20240630"
    :param save_prefix: str, 保存文件前缀
    """
    print(f"正在抓取 {report_date} 十大流通股东数据...")
    df = ak.stock_gdfx_free_holding_analyse_em(date=report_date)
    print(f"原始数据总条数: {len(df)}")

    # 正则匹配关键字：全国社保基金、社保基金、基本养老保险基金
    keyword_pattern = re.compile(r"(社保|全国社保|基本养老保险基金)", re.IGNORECASE)
    ss_df = df[df['股东名称'].str.contains(keyword_pattern, na=False)]

    # 去重（同一股票、同一股东名称只保留一条）
    ss_df = ss_df.drop_duplicates(subset=['股票代码', '股东名称'])

    # 选取关键信息并重命名字段
    ss_df = ss_df.rename(columns={
        '股票代码': '代码',
        '股票简称': '名称',
        '股东名称': '股东',
        '股东类型': '股东类型',
        '期末持股-数量': '持股数',
        '期末持股-数量变化': '持股变化',
        '期末持股-数量变化比例': '变化比例(%)',
        '期末持股-流通市值': '持股市值(元)',
        '公告日': '公告日',
        '公告日后涨跌幅-10个交易日': '公告后10日涨跌幅(%)',
        '公告日后涨跌幅-30个交易日': '公告后30日涨跌幅(%)',
        '公告日后涨跌幅-60个交易日': '公告后60日涨跌幅(%)'
    })

    # 排序：按持股市值降序
    ss_df = ss_df.sort_values(by='持股市值(元)', ascending=False)

    # 保存到文件
    excel_file = f"{save_prefix}_{report_date}.xlsx"
    csv_file = f"{save_prefix}_{report_date}.csv"
    ss_df.to_excel(excel_file, index=False)
    ss_df.to_csv(csv_file, index=False, encoding="utf-8-sig")

    print(f"筛选出 {len(ss_df)} 条社保/养老基金持仓记录")
    print(f"已保存到: {excel_file} 和 {csv_file}")
    print("前 5 条结果预览：")
    print(ss_df.head())


if __name__ == "__main__":
    # 修改这里的日期即可切换季度，如 20230930, 20231231, 20240630
    fetch_social_security_holdings(report_date="20240630")
