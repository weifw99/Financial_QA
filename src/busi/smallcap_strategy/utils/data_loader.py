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
        # ('datetime', None),
        # ('open', 'open'),
        # ('high', 'high'),
        # ('low', 'low'),
        # ('close', 'close'),
        # ('volume', 'volume'),

        # ('mv', 'mv'),
        # ('profit', 'profit'),
        # ('revenue', 'revenue'),
        # ('is_st', 'is_st'),  # 0 or 1 表示是否ST
        # ('dtformat', '%Y-%m-%d'),
        # ('timeframe', bt.TimeFrame.Days),
        # ('compression', 1),
        # ('openinterest', -1),
        ('mv', -1),
        ('profit', -1),
        ('revenue', -1),
        ('is_st', -1),  # 0 or 1 表示是否ST
        ('dtformat', '%Y-%m-%d'),
    )

    '''
    def __init__(self, **kwargs):
        # 确保数据格式正确
        if 'dataname' in kwargs:
            df = kwargs['dataname']
            if isinstance(df.index, pd.MultiIndex):
                # 重置索引，将日期和symbol作为列
                df = df.reset_index()
                # 设置日期为索引
                df.set_index('date', inplace=True)
                kwargs['dataname'] = df

        super().__init__(**kwargs)

        # 验证数据
        if self.p.dataname is None or self.p.dataname.empty:
            print("警告：输入数据为空")
            return

        print(f"数据源基本信息：\n{self.p.dataname.info()}")
        # print(f"数据源前5行：\n{self.p.dataname.head()}")

        # 检查价格数据
        # price_cols = ['open', 'high', 'low', 'close']
        price_cols = ['open', 'high', 'low', 'close',  'mv', 'profit', 'revenue',]

        for col in price_cols:
            if col not in self.p.dataname.columns:
                print(f"警告：缺少价格列 {col}")
            else:
                zero_prices = self.p.dataname[self.p.dataname[col] == 0]
                if not zero_prices.empty:
                    print(f"警告：{col}列中有{len(zero_prices)}个零值")
                    print(f"零值记录：\n{zero_prices[[ col]].head()}")

    '''


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

    # 获取所有时间数据， 使用000001.csv
    pdf = pd.read_csv(f'{zh_data_dir}/sh.000001/daily.csv')
    pdf['date'] = pd.to_datetime(pdf['date'])

    for i, stock_file in enumerate(os.listdir(zh_data_dir)):
        if i > 500:
            break

        data = pd.DataFrame(index=pdf['date'].unique())
        data = data.sort_index()


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

            # df.rename(columns={'netProfit': 'profit', 'MBRevenue': 'revenue', 'isST': 'is_st', 'date': 'datetime'}, inplace=True)
            df.rename(columns={'netProfit': 'profit', 'MBRevenue': 'revenue', 'isST': 'is_st', }, inplace=True)

            df['mv'] = df['totalShare'] * df['close_1'] # 市值 = 总股本 * 收盘价（不复权）

            df['openinterest'] = 0
            df['date'] = pd.to_datetime(df['date'])

            # 选择需要的列
            df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'mv', 'profit', 'revenue', 'is_st', 'openinterest',]]

            df.set_index('date', inplace=True)  # 设置 datetime 为索引
            df = df.sort_index()

            data_ = pd.merge(data, df, left_index=True, right_index=True, how='left')
            data_ = data_.sort_index()  # ✅ 强制升序
            # 检查并填充关键列
            # required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'mv', 'profit', 'revenue', 'is_st']
            # for col in required_cols:
            #     if col not in df.columns:
            #         raise ValueError(f"缺失字段：{col} in {stock_file}")
            # df = df[required_cols]

            # data_ = df.sort_index()
            data_.loc[:, ['volume', 'openinterest']] = data_.loc[:, ['volume', 'openinterest']].fillna(0)
            data_.loc[:, ['open', 'high', 'low', 'close']] = data_.loc[:, ['open', 'high', 'low', 'close']].bfill()
            data_.bfill(inplace=True)
            data_.fillna(0, inplace=True)
            rsub_cols = [ 'open', 'high', 'low', 'close', ]

            data_.dropna(subset=rsub_cols, inplace=True)

            # print("最终合并后的 data_ 形状:", data_.shape)
            # print("缺失字段统计:\n", data_.isnull().sum())
            # print("close 列前5行:\n", data_['close'].head())

            # if df.empty or len(df) < 100:
            #     continue
            data = CustomPandasData(dataname=data_,
                                    fromdate=from_idx,
                                    todate=to_idx,
                                    timeframe=bt.TimeFrame.Days,
                                    name=stock_file.replace('.csv', ''))

            # data._name = stock_file.replace('.csv', '')  # 设置数据名称（用于后续匹配指数名等）
            # print(f'添加数据源：{data._name}，数据日期范围：{df["datetime"].min()} ~ {df["datetime"].max()}，共 {len(df)} 条记录')
            datas.append(data)
    return datas

