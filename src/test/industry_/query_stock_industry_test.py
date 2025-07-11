import baostock as bs
import pandas as pd

import akshare as ak
import pandas as pd
from datetime import datetime
import time, os


def save_all_stock_industry_map():
    board_industry_base_path = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/board_industry"
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
            cons_df = ak.stock_board_industry_cons_em(symbol=industry_name)
            for _, row in cons_df.iterrows():
                result.append({
                    "code": row["代码"],
                    "name": row["名称"],
                    "industry_code": industry_code,
                    "industry_name": industry_name,
                    "date": today_str
                })
            print(f"✅ {industry_name} 获取成功，共 {len(cons_df)} 条")
            time.sleep(0.5)  # 防止请求太快被封
        except Exception as e:
            print(f"⚠️ 获取 {industry_name} 失败：{e}")

    result_df = pd.DataFrame(result)
    result_df.to_csv(f"{board_industry_base_path}/{today_str}.csv", index=False, encoding='utf-8-sig')
    print(f"\n📁 已保存到文件: board_industry_{today_str}.csv，共 {len(result_df)} 条")



if __name__ == '__main__':
    '''
    # 登陆系统
    lg = bs.login()
    # 显示登陆返回信息
    print('login respond error_code:'+lg.error_code)
    print('login respond  error_msg:'+lg.error_msg)

    # 获取行业分类数据
    rs = bs.query_stock_industry()
    # rs = bs.query_stock_basic(code_name="浦发银行")
    print('query_stock_industry error_code:'+rs.error_code)
    print('query_stock_industry respond  error_msg:'+rs.error_msg)

    # 打印结果集
    industry_list = []
    while (rs.error_code == '0') & rs.next():
        # 获取一条记录，将记录合并在一起
        industry_list.append(rs.get_row_data())
    result = pd.DataFrame(industry_list, columns=rs.fields)
    # 结果集输出到csv文件
    result.to_csv("stock_industry.csv", index=False)
    print(result)

    # 登出系统
    bs.logout()
    '''
    import akshare as ak

    # 获取东方财富行业指数的历史 K 线数据
    # 示例：获取“煤炭行业”（板块代码 BK0421）的历史行情
    df = ak.stock_board_industry_hist_em(symbol="BK0421", start_date="20100101", end_date="20990701")

    # 字段说明：
    # 日期,开盘,收盘,最高,最低,成交量,成交额,振幅,涨跌幅,涨跌额,换手率

    print(df.head())



    import akshare as ak
    import pandas as pd

    # 1. 获取行业板块基本情况
    df_ind = ak.stock_board_industry_name_em()

    # 2. 获取行业资金流（如今日流入）
    df_fund = ak.stock_sector_fund_flow_rank(indicator="5日", sector_type="行业资金流")

    # 3. 合并数据
    df = pd.merge(df_ind, df_fund[['名称', '5日主力净流入-净额', '5日主力净流入-净占比']],
                  left_on='板块名称', right_on='名称', how='left')

    # 4. 统计趋势：涨跌家数、涨停/跌停
    # df['涨停家数'] = df['板块名称'].apply(lambda x: ...)  # 若API无，可另行爬取或估算
    # 注：API 本身提供“上涨家数”和“下跌家数”。

    # 5. 排序或筛选分析
    # df_sorted = df.sort_values('主力净流入-净额', ascending=False)
    # print(df_sorted[['板块名称', '涨跌幅', '上涨家数', '下跌家数', '主力净流入-净额']])

    import akshare as ak
    # 获取行业历史资金流
    stock_sector_fund_flow_hist_df = ak.stock_sector_fund_flow_hist(symbol="汽车服务")
    print(stock_sector_fund_flow_hist_df)

    save_all_stock_industry_map()

