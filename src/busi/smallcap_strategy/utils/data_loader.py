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
    lines = ('mv', 'profit', 'revenue', 'is_st', 'profit_ttm', 'roeAvg',)
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
        ('profit_ttm', -1),  #
        ('roeAvg', -1),  #
        ('dtformat', '%Y-%m-%d'),
    )



def load_stock_data(from_idx, to_idx):
    """
    批量加载 data_dir 下的所有 CSV 文件，返回数据列表
    文件名将作为数据名称注入，如 '600000.csv' -> data._name = '600000'
    :param data_dir: 包含CSV的路径
    :return: list of data feeds
    """
    zz_code_data_paths = [
        '/Users/dabai/liepin/study/llm/Financial_QA/src/test/中证1000-000852.csv',
        '/Users/dabai/liepin/study/llm/Financial_QA/src/test/中证2000-932000.csv',
    ]
    # zz_code_data_path = '/Users/dabai/liepin/study/llm/Financial_QA/src/test/中证1000-000852.csv'
    # zz_code_data_path = '/Users/dabai/liepin/study/llm/Financial_QA/src/test/中证2000-932000.csv'
    zz_code_list = []
    for zz_code_data_path in zz_code_data_paths:
        zz_code_df = pd.read_csv(zz_code_data_path)
        zz_code_list += zz_code_df['type'].tolist()

    datas = []

    base_data_path = '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data'
    zh_data_dir = Path(base_data_path) / 'market'
    financial_data_dir = Path(base_data_path).parent / 'zh_data/financial'

    # 获取所有时间数据， 使用000001.csv
    pdf = pd.read_csv(f'{zh_data_dir}/sh.000001/daily.csv')
    pdf['date'] = pd.to_datetime(pdf['date'])

    data = pd.DataFrame(index=pdf['date'].unique())
    data = data.sort_index()

    select_cols = ['date', 'open', 'high', 'low', 'close', 'volume', ]
    add_cols = ['mv', 'profit', 'revenue', 'is_st', 'profit_ttm', 'roeAvg', 'openinterest', ]
    # 加载 SZ510880 SH159300
    etf_list = ['SZ510880', 'SH159919', 'SZ510050', 'SZ588000', 'SZ511880']
    etf_path = '/Users/dabai/liepin/study/llm/Financial_QA/src/busi/etf_/data/etf_trading/daily'
    for etf_code in etf_list:
        etf_df = pd.read_csv(f'{etf_path}/{etf_code}.csv')
        # 选择需要的列
        etf_df = etf_df[select_cols]
        for col in add_cols:
            etf_df[col] = 0
        etf_df['date'] = pd.to_datetime(etf_df['date'])
        etf_df.set_index('date', inplace=True)  # 设置 datetime 为索引
        etf_df = etf_df.sort_index()
        data_ = pd.merge(data, etf_df, left_index=True, right_index=True, how='left')
        data_.fillna(0, inplace=True)
        data_ = data_.sort_index()  # ✅ 强制升序
        pandas_data = CustomPandasData(dataname=data_,
                                       fromdate=from_idx,
                                       todate=to_idx,
                                       timeframe=bt.TimeFrame.Days,
                                       name=f'etf_{etf_code}')
        datas.append(pandas_data)


    index_list =['csi932000', 'sz399101' ]
    # 获取指数数据
    zz_path = '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/index'

    for index_code in index_list:

        zz_df = pd.read_csv(f'{zz_path}/{index_code}.csv')
        # 选择需要的列
        zz_df = zz_df[select_cols]
        for col in add_cols:
            zz_df[col] = 0
        zz_df['date'] = pd.to_datetime(zz_df['date'])
        zz_df.set_index('date', inplace=True)  # 设置 datetime 为索引
        zz_df = zz_df.sort_index()
        data_ = pd.merge(data, zz_df, left_index=True, right_index=True, how='left')
        data_.fillna(0, inplace=True)
        data_ = data_.sort_index()  # ✅ 强制升序
        pandas_data = CustomPandasData(dataname=data_,
                                       fromdate=from_idx,
                                       todate=to_idx,
                                       timeframe=bt.TimeFrame.Days,
                                       name=index_code)
        datas.append(pandas_data)

    temp_stock_list = ['sh.000300',  'sh.000016', 'sh.000852' ]
    for i, stock_file in enumerate(os.listdir(zh_data_dir)):
        # if i > 500:
        #     break

        # 测试
        # if len(datas) >100 and stock_file  not in temp_stock_list:
        #     continue

        # 使用中证1000或则中证2000股票回测
        if stock_file not in zz_code_list and stock_file not in temp_stock_list:
            continue

        print(f'{i}/{stock_file}')
        file_path = f'{zh_data_dir}/{stock_file}/daily.csv'
        file_path_a = f'{zh_data_dir}/{stock_file}/daily_a.csv'

        # 获取财务盈利信息
        financial_path = f'{financial_data_dir}/{stock_file}/income.csv'
        if os.path.exists(file_path):
            df = pd.read_csv(file_path)
            df_a = pd.read_csv(file_path_a)[['date','close']]
            df_a.rename(columns={'close': 'close_1'}, inplace=True)

            df = pd.merge(df, df_a, on='date', how='inner')

            # 使用后复权价格，factor均设置为1， 回测使用该因子
            df['factor'] = 1.0
            # 确保 date 列为 datetime 类型并排序
            df['date'] = pd.to_datetime(df['date'])
            df_sorted = df.sort_values('date')
            if os.path.exists(financial_path):

                financial_df = pd.read_csv(financial_path)
                financial_df['date'] = financial_df['pubDate']

                financial_df = financial_df[['date', 'netProfit', 'MBRevenue', 'totalShare', 'liqaShare', 'epsTTM', 'roeAvg', ]]

                # 确保 date 列为 datetime 类型并排序
                financial_df['date'] = pd.to_datetime(financial_df['date'])


                df2_sorted = financial_df.sort_values('date').ffill().dropna()

                # 归属母公司股东的净利润TTM
                # epsTTM	每股收益	归属母公司股东的净利润TTM/最新总股本
                df2_sorted['profit_ttm'] = df2_sorted['totalShare'] * df2_sorted['epsTTM']

                # pubDate	公司发布财报的日期
                # roeAvg	净资产收益率(平均)(%)	归属母公司股东净利润/[(期初归属母公司股东的权益+期末归属母公司股东的权益)/2]*100%
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
                df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'mv', 'profit', 'revenue', 'is_st', 'profit_ttm', 'roeAvg', 'openinterest',]]

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
                pandas_data = CustomPandasData(dataname=data_,
                                               fromdate=from_idx,
                                               todate=to_idx,
                                               timeframe=bt.TimeFrame.Days,
                                               name=stock_file.replace('.csv', ''))

                # data._name = stock_file.replace('.csv', '')  # 设置数据名称（用于后续匹配指数名等）
                # print(f'添加数据源：{data._name}，数据日期范围：{df["datetime"].min()} ~ {df["datetime"].max()}，共 {len(df)} 条记录')
                datas.append(pandas_data)
            else:
                print(f'{stock_file} 缺少财务信息')
                # 选择需要的列
                df_sorted = df_sorted[select_cols]
                for col in add_cols:
                    df_sorted[col] = 0

                df_sorted.set_index('date', inplace=True)  # 设置 datetime 为索引
                df_sorted = df_sorted.sort_index()
                data_ = pd.merge(data, df_sorted, left_index=True, right_index=True, how='left')
                data_.fillna(0, inplace=True)
                data_ = data_.sort_index()  # ✅ 强制升序
                pandas_data = CustomPandasData(dataname=data_,
                                               fromdate=from_idx,
                                               todate=to_idx,
                                               timeframe=bt.TimeFrame.Days,
                                               name=stock_file.replace('.csv', ''))
                datas.append(pandas_data)

    return datas

