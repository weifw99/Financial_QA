import random

import akshare as ak
from datetime import datetime
import time

import pandas as pd
import os
import glob


def compute_industry_rise_fall() -> pd.DataFrame:
    """
    参数：
    - industry_file: 股票行业归属 CSV 文件路径（包含 code, industry_name, date）
    - quote_dir: 股票行情数据文件夹（每个文件如 20250710.csv）
    - output_file: 保存统计结果的路径
    """

    # 替换为你需要的目录路径
    directory = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/industry/board_industry"
    # 使用 glob 获取所有 csv 文件
    files = glob.glob(os.path.join(directory, "*.csv"))
    # 按照修改时间排序，取最新的一个
    latest_file = max(files, key=os.path.getmtime)

    print(f"📁 最新文件: {latest_file}")

    # 读取最新文件到 DataFrame
    industry_df = pd.read_csv(latest_file, dtype={'code': str})
    industry_df = industry_df[['code', 'name', 'industry_name', 'industry_code']]

    quote_dir: str = '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/market'
    all_results = []

    for fname in sorted(os.listdir(quote_dir)):
        if '.' not in fname:
            print(f"⚠️ 文件 {fname} 无效")
            continue
        code_str = fname.split('.')[1]

        # 加载当日行情数据
        quote_df = pd.read_csv(os.path.join(quote_dir, f'{fname}/daily_a.csv') )
        quote_df['code'] = code_str
        quote_df = quote_df[['code', 'open', 'close', 'date']]
        quote_df = quote_df[quote_df['open'] > 0]  # 排除开盘为0的无效数据
        quote_df['pct'] = (quote_df['close'] - quote_df['open']) / quote_df['open'] * 100

        # 合并行业数据
        merged = pd.merge(quote_df, industry_df, on=['code'], how='inner')

        # 去除没有行业信息的
        merged = merged.dropna(subset=['industry_name', 'code'])
        all_results.append( merged )

    all_data = pd.concat(all_results, ignore_index=True)
    all_results = []
    # 分组统计
    for group_index, group in all_data.groupby([ 'date', 'industry_name',]):
        date1, ind_name = group_index[0], group_index[1]
        total = len(group)
        up = (group['pct'] > 0).sum()
        down = (group['pct'] < 0).sum()
        limit_up = (group['pct'] > 9.8).sum()
        limit_down = (group['pct'] < -9.8).sum()
        up_1 = (group['pct'] > 1).sum()
        up_2 = (group['pct'] > 2).sum()
        down_2 = (group['pct'] < -2).sum()
        up_5 = (group['pct'] > 5).sum()
        down_5 = (group['pct'] < -5).sum()
        all_results.append({
            'date': date1,
            'industry': ind_name,
            'total': total,
            'up_rate': up/total,
            'up': up,
            'down': down,
            'limit_up': limit_up,
            'limit_down': limit_down,
            'up_1pct': up_1,
            'up_2pct': up_2,
            'up_5pct': up_5,
            'up_1pct_rate': up_1 / total,
            'up_2pct_rate': up_2 / total,
            'up_5pct_rate': up_5 / total,
            'down_2pct': down_2,
            'down_5pct': down_5,
        })


    # 保存结果
    result_df = pd.DataFrame(all_results)
    # result_df.to_csv(output_file, index=False, encoding='utf-8-sig')
    # print(f"✅ 行业涨跌统计已保存：{output_file}")
    return result_df


def save_all_stock_industry_map():
    base_path = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/industry"
    if not os.path.exists(base_path):
        os.makedirs(base_path)
    board_industry_base_path = f"{base_path}/industry_flow"
    if not os.path.exists(board_industry_base_path):
        os.makedirs(board_industry_base_path)
    # 当前日期
    today_str = datetime.today().strftime("%Y%m%d")

    # 获取行业列表
    industry_df = ak.stock_board_industry_name_em()
    industry_list = industry_df.to_dict('records')

    result = []

    for item in industry_list:
        industry_name = item['板块名称']
        industry_code = item['板块代码']
        try:

            # 获取行业历史资金流
            cons_df = ak.stock_sector_fund_flow_hist(symbol=industry_name)

            cons_df['行业代码'] = industry_code
            cons_df['行业名称'] = industry_name
            cons_df.to_csv(f"{board_industry_base_path}/{industry_name}.csv", index=False, encoding='utf-8-sig')
            result.append(cons_df)
            print(f"✅ {industry_name} 获取成功，共 {len(cons_df)} 条")
            time.sleep(random.randint(1, 5))  # 防止请求太快被封
        except Exception as e:
            print(f"⚠️ 获取 {industry_name} 失败：{e}")

    result_df = pd.concat(result, ignore_index=True)
    result_df.to_csv(f"{base_path}/industry_flow.csv", index=False, encoding='utf-8-sig')
    print(f"\n📁 已保存到文件: {base_path}/industry_flow.csv，共 {len(result_df)} 条")

    industry_rise_df = compute_industry_rise_fall()
    industry_rise_df.to_csv(f"{base_path}/industry_stock_stat.csv", index=False, encoding='utf-8-sig')
    print(f"\n📁 已保存到文件: {base_path}/industry_stock_stat.csv，共 {len(industry_rise_df)} 条")

    # 确保用于关联的字段是字符串类型
    result_df['日期'] = result_df['日期'].astype(str)
    result_df['行业名称'] = result_df['行业名称'].astype(str)

    industry_rise_df['date'] = industry_rise_df['date'].astype(str)
    industry_rise_df['industry'] = industry_rise_df['industry'].astype(str)

    # 再进行合并
    merge_result = pd.merge(
        result_df,
        industry_rise_df,
        left_on=['日期', '行业名称'],
        right_on=['date', 'industry'],
        how='left'
    )

    merge_result.to_csv(f"{base_path}/industry_flow_merge.csv", index=False, encoding='utf-8-sig')
    print(f"\n📁 已保存到文件: {base_path}/industry_flow_merge.csv，共 {len(merge_result)} 条")


if __name__ == '__main__':

    save_all_stock_industry_map()
    # compute_industry_rise_fall()

