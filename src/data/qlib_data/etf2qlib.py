
## 将zh_data中的数据转换为qlib的格式
# zh_data 目录结构 data/zh_data/    
# ├── market/
# ├── raw/
#   ├── index/ # 指数数据
#   ├── stock_list.csv # 股票列表


# qlib_data 目录结构 data/qlib_data/
# 每个股票保存为一个csv文件。其中日期列命名为'date'。文件名为股票代码，如股票600000的价格数据，保存在'SH600000.csv'

import os
import pandas as pd
import argparse

from src.data.qlib_data.scripts.dump_bin import DumpDataAll

if __name__ == '__main__':

    out_base_path = '/Users/dabai/liepin/study/llm/Financial_QA/data/qlib_data'
    csv_dir_base = '/Users/dabai/liepin/study/llm/Financial_QA/src/busi/etf_/data'
    qlib_csv_dir = csv_dir_base + '/etf_trading/'
    qlib_data_dir = out_base_path + '/cn_data_etf'

    # python scripts/dump_bin.py dump_all --csv_path ~/dev/stock_price_data_wind --qlib_dir ~/dev/qlib_data/cn_data_wind

    include_fields = "open,high,low,close,volume,amount,turn,pctChg,factor"
    DumpDataAll(csv_path=qlib_csv_dir + 'daily',
                qlib_dir=qlib_data_dir,
                freq="day",
                max_workers=4,
                date_field_name="date",
                file_suffix=".csv",
                symbol_field_name="symbol",
                include_fields=include_fields,).dump()
