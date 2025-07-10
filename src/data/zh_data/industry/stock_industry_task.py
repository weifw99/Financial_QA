import baostock as bs
import pandas as pd

import akshare as ak
import pandas as pd
from datetime import datetime
import time, os


def save_all_stock_industry_map():
    board_industry_base_path = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/industry/board_industry"
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
    print(f"\n📁 已保存到文件: {board_industry_base_path}/{today_str}.csv，共 {len(result_df)} 条")


if __name__ == '__main__':

    save_all_stock_industry_map()

