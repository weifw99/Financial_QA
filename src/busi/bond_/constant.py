import os
from pathlib import Path

# 获取当前 Python 文件所在的目录
# current_directory = Path(__file__).resolve().parent

# print(current_directory)

class DataCons:
    DATA_DIR = f'{Path(__file__).resolve().parent}/data'
    # 可转债 数据列表，地址
    BOND_INFO_DIR = f'{DATA_DIR}/bond_info'
    BOND_INFO_CAT = f'{DATA_DIR}/bond_info/list_concat.csv'
    BOND_INFO_CAT_DAY = f'{DATA_DIR}/' + 'bond_info/daily/{}.csv'
    # 可转债交易数据
    BOND_TRADING_DAY_DIR = f'{DATA_DIR}/bond_trading'
    # 可转债对应的股票数据
    STOCK_TRADING_DIR = f'{DATA_DIR}/stock_trading'

    # 可转债列表明细
    BOND_INFO_FILE_PATH = f'{DATA_DIR}/bond_list_info.csv'
    # 可转债比价表
    BOND_COV_COMPARISON_FILE_PATH = f'{DATA_DIR}/bond_cov_comparison.csv'
    BOND_ZH_COV_VALUE_ANALYSIS_FILE_PATH = f'{BOND_TRADING_DAY_DIR}/' + 'bond_zh_cov_value_analysis/{}.csv'
    BOND_ZH_HS_DAILY_PATH = f'{BOND_TRADING_DAY_DIR}/' + 'daily/'
    BOND_ZH_HS_DAILY_FILE_PATH = f'{BOND_ZH_HS_DAILY_PATH}/' + '{}.csv'

    STOCK_TRADING_DAY_PATH = f'{STOCK_TRADING_DIR}/' + 'daily/'
    STOCK_TRADING_DAY_FILE_PATH = f'{STOCK_TRADING_DAY_PATH}/' + '{}.csv'

