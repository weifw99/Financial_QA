import os
from pathlib import Path

# 获取当前 Python 文件所在的目录
# current_directory = Path(__file__).resolve().parent

# print(current_directory)

class DataCons:
    DATA_DIR = f'{Path(__file__).resolve().parent}/data'
    # etf 数据列表，地址
    ETF_INFO_DIR = f'{DATA_DIR}/etf_info'
    ETF_INFO_CAT = f'{DATA_DIR}/etf_info/list_concat.csv'
    ETF_INFO_CAT_DAY = f'{DATA_DIR}/'+ 'etf_info/daily/{}.csv'
    # etf交易数据
    ETF_TRADING_DAY_DIR = f'{DATA_DIR}/etf_trading'
    # etf列表明细
    ETF_INFO_FILE_PATH = f'{DATA_DIR}/etf_list_info.csv'
    ETF_INFO_HIS_FILE_PATH = f'{DATA_DIR}/etf_list_info_his.csv'
    ETF_SINA_INFO_FILE_PATH = f'{DATA_DIR}/etf_list_info_sina.csv'
    ETF_HS_DAILY_FILE_PATH = f'{ETF_TRADING_DAY_DIR}/' + 'daily/{}.csv'


