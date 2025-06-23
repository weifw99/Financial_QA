# utils/data_loader.py
# 封装股票与指数的 CSV 数据加载，注入自定义字段：市值、利润、营收、ST
import os
from pathlib import Path

import pandas as pd
import backtrader as bt

class CustomPandasData(bt.feeds.PandasData):
    """
    自定义数据类，包含：市值、市盈率、利润、营收、是否ST标记等基本面数据
    需要保证df中有以下字段：datetime, open, high, low, close, volume, mv, profit, revenue, is_st
    """
    lines = ('mv', 'profit', 'revenue', 'is_st',)
    params = (# 'datetime', 'open', 'high', 'low', 'close', 'volume', 'mv', 'profit', 'revenue', 'is_st'
        ('datetime', None),
        ('open', 'open'),
        ('high', 'high'),
        ('low', 'low'),
        ('close', 'close'),
        ('volume', 'volume'),

        ('mv', 'mv'),
        ('profit', 'profit'),
        ('revenue', 'revenue'),
        ('is_st', 'is_st'),  # 0 or 1 表示是否ST
        ('dtformat', '%Y-%m-%d'),
        ('timeframe', bt.TimeFrame.Days),
        ('compression', 1),
        ('openinterest', -1),
    )

def load_stock_data(from_idx, to_idx):
    """
    批量加载 data_dir 下的所有 CSV 文件，返回数据列表
    文件名将作为数据名称注入，如 '600000.csv' -> data._name = '600000'
    :param data_dir: 包含CSV的路径
    :return: list of data feeds
    """
    datas = []

    base_data_path = '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data'
    zh_data_dir = Path(base_data_path) / 'market'
    financial_data_dir = Path(base_data_path).parent / 'zh_data/financial'

    for i, stock_file in enumerate(os.listdir(zh_data_dir)):
        # if i > 10:
        #     break
        print(f'{i}/{stock_file}')
        file_path = f'{zh_data_dir}/{stock_file}/daily.csv'
        file_path_a = f'{zh_data_dir}/{stock_file}/daily_a.csv'

        # 获取财务盈利信息
        financial_path = f'{financial_data_dir}/{stock_file}/income.csv'
        if os.path.exists(file_path) and os.path.exists(financial_path):
            df = pd.read_csv(file_path)
            df_a = pd.read_csv(file_path_a)[['date','close']]
            df_a.rename(columns={'close': 'close_1'}, inplace=True)

            df = pd.merge(df, df_a, on='date', how='inner')

            # 使用后复权价格，factor均设置为1， 回测使用该因子
            df['factor'] = 1.0

            financial_df = pd.read_csv(financial_path)
            financial_df['date'] = financial_df['pubDate']

            financial_df = financial_df[['date', 'netProfit', 'MBRevenue', 'totalShare', 'liqaShare']]
            # 确保 date 列为 datetime 类型并排序
            df['date'] = pd.to_datetime(df['date'])
            financial_df['date'] = pd.to_datetime(financial_df['date'])

            df_sorted = df.sort_values('date')
            df2_sorted = financial_df.sort_values('date').ffill().dropna()


            # pubDate	公司发布财报的日期
            # statDate	财报统计的季度的最后一天, 比如2017-03-31, 2017-06-30
            # netProfit	净利润(元)
            # MBRevenue	主营营业收入(元)
            # totalShare	总股本(股)
            # 使用 pd.merge_asof 实现按时间向前填充匹配
            df = pd.merge_asof(df_sorted, df2_sorted, on='date', direction='backward')

            df.rename(columns={'netProfit': 'profit', 'MBRevenue': 'revenue', 'isST': 'is_st', 'date': 'datetime'}, inplace=True)

            df['mv'] = df['totalShare'] * df['close_1'] # 市值 = 总股本 * 收盘价（不复权）

            # 检查并填充关键列
            required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'mv', 'profit', 'revenue', 'is_st']
            for col in required_cols:
                if col not in df.columns:
                    raise ValueError(f"缺失字段：{col} in {stock_file}")

            df = df[required_cols]
            df['openinterest'] = 0
            df['datetime'] = pd.to_datetime(df['datetime'])
            df.set_index('datetime', inplace=True)  # 设置 datetime 为索引

            data_ = df.sort_index()
            data_.loc[:, ['volume', 'openinterest']] = data_.loc[:, ['volume', 'openinterest']].fillna(0)
            data_.loc[:, ['open', 'high', 'low', 'close']] = data_.loc[:, ['open', 'high', 'low', 'close']].bfill()
            data_.bfill(inplace=True)
            data_.fillna(0, inplace=True)
            rsub_cols = [ 'open', 'high', 'low', 'close', 'volume', 'mv', 'profit', 'revenue', 'is_st']

            data_.dropna(subset=rsub_cols, inplace=True)

            if data_.empty or len(data_) < 100:
                continue

            data = CustomPandasData(dataname=df,
                                    fromdate=from_idx,
                                    todate=to_idx,
                                    dtformat='%Y-%m-%d',
                                    timeframe=bt.TimeFrame.Days,
                                    )
            data._name = stock_file.replace('.csv', '')  # 设置数据名称（用于后续匹配指数名等）
            # print(f'添加数据源：{data._name}，数据日期范围：{df["datetime"].min()} ~ {df["datetime"].max()}，共 {len(df)} 条记录')
            datas.append(data)
    return datas

