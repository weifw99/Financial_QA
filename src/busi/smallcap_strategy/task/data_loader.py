import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta


def load_recent_data():
    """
    加载最近30天的数据（用于每日信号任务）
    - 返回 dict[str, pd.DataFrame]
    - 股票名称为 key，DataFrame 为值
    """
    base_data_path = '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data'
    zh_data_dir = Path(base_data_path) / 'market'
    financial_data_dir = Path(base_data_path).parent / 'zh_data/financial'
    etf_path = '/Users/dabai/liepin/study/llm/Financial_QA/src/busi/etf_/data/etf_trading/daily'
    index_path = '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/index'

    today = datetime.today()
    from_date = today - timedelta(days=40)

    result = {}

    # 统一时间索引基准（用上证指数）
    calendar_df = pd.read_csv(f'{zh_data_dir}/sh.000001/daily.csv')
    calendar_df['date'] = pd.to_datetime(calendar_df['date'])
    calendar_df = calendar_df[calendar_df['date'] >= from_date]
    data_index = pd.DataFrame(index=calendar_df['date'].sort_values().unique())

    select_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turn']
    add_cols = ['amount', 'turn', 'mv', 'profit', 'revenue', 'is_st', 'profit_ttm', 'roeAvg', 'openinterest', ]

    def process_dataframe(df):
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= from_date]
        df.set_index('date', inplace=True)
        df = df.sort_index()
        df = pd.merge(data_index, df, left_index=True, right_index=True, how='left')
        df.fillna(0, inplace=True)
        return df

    # 加载 ETF
    etf_list = ['SZ510880', 'SH159919', 'SZ510050', 'SZ588000', 'SZ511880']
    for code in etf_list:
        f = f'{etf_path}/{code}.csv'
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)[select_cols]
        for col in add_cols:
            if col not in df.columns:
                df[col] = 0
        result[f'etf_{code}'] = process_dataframe(df)

    # 加载指数
    index_list = ['csi932000', 'sz399101']
    index_list =['csi932000', 'sz399101' , 'sh000905', 'sh000852', 'sh000046', 'sz399005', 'sz399401']

    for code in index_list:
        f = f'{index_path}/{code}.csv'
        if not os.path.exists(f):
            continue
        df = pd.read_csv(f)[select_cols[:-1]]
        for col in add_cols:
            if col not in df.columns:
                df[col] = 0
        result[code] = process_dataframe(df)

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

    temp_stock_list = ['sh.000300', 'sh.000016', 'sh.000852']
    for i, stock_code in enumerate(os.listdir(zh_data_dir)):
        # file_path = f'{zh_data_dir}/{stock_code}/daily.csv'
        file_path_a = f'{zh_data_dir}/{stock_code}/daily_a.csv'
        income_path = f'{financial_data_dir}/{stock_code}/income.csv'

        # 过滤创业板/科创板/北交所股票
        if ('.30' in stock_code
                or '.68' in stock_code
                or '.8' in stock_code
                or '.4' in stock_code):
            print(f'过滤创业板/科创板/北交所股票: {stock_code}')
            continue
        # 使用中证1000或则中证2000股票回测
        # if stock_code not in zz_code_list and stock_code not in temp_stock_list:
        #     continue

        if not os.path.exists(file_path_a):
            continue

        df = pd.read_csv(file_path_a)
        df['close_1'] = df['close']
        # df_a = pd.read_csv(file_path_a)[['date', 'close']].rename(columns={'close': 'close_1'})
        # df = pd.merge(df, df_a, on='date', how='inner')
        df['factor'] = 1.0
        df['date'] = pd.to_datetime(df['date'])
        df = df[df['date'] >= from_date]
        df = df.sort_values('date')

        if os.path.exists(income_path):
            income_df = pd.read_csv(income_path)
            income_df['date'] = pd.to_datetime(income_df['pubDate'])
            income_df = income_df[['date', 'netProfit', 'MBRevenue', 'totalShare', 'epsTTM', 'roeAvg']]
            income_df = income_df.sort_values('date').ffill()
            income_df['profit_ttm'] = income_df['totalShare'] * income_df['epsTTM']
            df = pd.merge_asof(df, income_df, on='date', direction='backward')
        else:
            df['netProfit'] = 0
            df['MBRevenue'] = 0
            df['totalShare'] = 0
            df['roeAvg'] = 0
            df['profit_ttm'] = 0

        df['profit'] = df['netProfit']
        df['revenue'] = df['MBRevenue']
        df['mv'] = df['totalShare'] * df['close_1']
        df['is_st'] = df.get('isST', 0)
        df['openinterest'] = 0

        try:
            df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'mv', 'profit', 'revenue',
                     'is_st', 'profit_ttm', 'roeAvg', 'openinterest']]
            df = process_dataframe(df)
            result[stock_code] = df
        except Exception as e:
            print(f'❌ 处理 {stock_code} 失败: {e}')

    return result, calendar_df['date'].sort_values().unique()[-1]