# utils/data_loader.py
# 封装股票与指数的 CSV 数据加载，注入自定义字段：市值、利润、营收、ST
import os
import pandas as pd
import backtrader as bt

class CustomCSV(bt.feeds.GenericCSVData):
    """
    自定义数据类，包含：市值、市盈率、利润、营收、是否ST标记等基本面数据
    需要保证CSV中有以下字段：datetime, open, high, low, close, volume, mv, profit, revenue, is_st
    """
    lines = ('mv', 'profit', 'revenue', 'is_st',)
    params = (
        ('datetime', 0),
        ('open', 1),
        ('high', 2),
        ('low', 3),
        ('close', 4),
        ('volume', 5),
        ('mv', 6),
        ('profit', 7),
        ('revenue', 8),
        ('is_st', 9),  # 0 or 1 表示是否ST
        ('dtformat', '%Y-%m-%d'),
        ('timeframe', bt.TimeFrame.Days),
        ('compression', 1),
        ('openinterest', -1),
    )

def load_stock_data(data_dir):
    """
    批量加载 data_dir 下的所有 CSV 文件，返回数据列表
    文件名将作为数据名称注入，如 '600000.csv' -> data._name = '600000'
    :param data_dir: 包含CSV的路径
    :return: list of data feeds
    """
    datas = []
    for fname in os.listdir(data_dir):
        if not fname.endswith('.csv'):
            continue
        fpath = os.path.join(data_dir, fname)
        df = pd.read_csv(fpath)

        # 检查并填充关键列
        required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'mv', 'profit', 'revenue', 'is_st']
        for col in required_cols:
            if col not in df.columns:
                raise ValueError(f"缺失字段：{col} in {fname}")

        data = CustomCSV(dataname=fpath)
        data._name = fname.replace('.csv', '')  # 设置数据名称（用于后续匹配指数名等）
        datas.append(data)
    return datas