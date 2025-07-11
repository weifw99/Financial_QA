import akshare as ak
import pandas as pd
import os
import time
from datetime import datetime


def save_all_industry_history(start_date="20200101", end_date=None, save_dir=None):
    """
    获取所有东方财富行业指数的历史行情，并保存为 CSV 文件。

    参数：
        start_date (str): 起始日期，格式如 "20200101"
        end_date (str): 结束日期（默认当天）
        save_dir (str): 保存目录（默认当前目录下的 ./industry_price_data）
    """
    if end_date is None:
        end_date = datetime.today().strftime("%Y%m%d")

    if save_dir is None:
        save_dir = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/industry/industry_price"

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    industry_list = ak.stock_board_industry_name_em().to_dict("records")
    success_count = 0

    for item in industry_list:
        name = item["板块名称"]
        code = item["板块代码"]
        try:
            df = ak.stock_board_industry_hist_em(
                symbol=code,
                start_date=start_date,
                end_date=end_date,
                adjust="qfq",  # 可选：qfq/hfq/None
            )

            # df.rename(columns={"日期": "date", "开盘": "open", "收盘": "close", "最高": "high", "最低": "low", "成交量": "volume", "成交额": "amount", "振幅": "pctChg", "换手率": "turn",}, inplace=True)

            df['行业代码'] =  code
            df['行业名称'] =  name
            file_path = os.path.join(save_dir, f"{code}_{name}.csv")
            df.to_csv(file_path, index=False, encoding="utf-8-sig")
            print(f"✅ {name}（{code}）保存成功，{len(df)} 行")
            success_count += 1
            time.sleep(0.5)  # 防止被封
        except Exception as e:
            print(f"⚠️ {name}（{code}）获取失败：{e}")
            continue

    print(f"\n📊 已成功获取 {success_count}/{len(industry_list)} 个行业数据。")


if __name__ == "__main__":
    save_all_industry_history(start_date="20220101")