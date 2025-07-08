import os
from pathlib import Path
import pandas as pd
import numpy as np
from datetime import datetime, timedelta

from src.busi.smallcap_strategy.utils.data_loader import process_financial_data, merge_with_stock


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
    add_cols = ['amount', 'turn', 'mv', 'is_st', 'profit_ttm_y', 'profit_y', 'revenue_y', 'roeAvg_y', 'profit_ttm_q', 'profit_q', 'revenue_single_q', 'roeAvg_q', 'openinterest', ]

    def process_dataframe(df):
        df.loc[:, 'date'] = pd.to_datetime(df['date'])
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
    index_list =['csi932000', 'sz399101' , 'sh000905', 'sh000852', 'sh000046', 'sz399005', 'sz399008', 'sz399401',
                 'sz399649','sz399663','sz399377','sh000046','sz399408','sz399401','sh000991' ,
                 'sh000852', 'sz399004', 'sh000905', 'sz399006',
                 'sz399693']
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
        '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/raw/index/中小板指数-中小100-399005.csv',
        # '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/raw/index/中小综指-399101.csv',
        # '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/raw/index/中证1000-000852.csv',
        # '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/raw/index/中证2000-932000.csv',
        # '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/raw/index/微盘股-BK1158.csv',
    ]
    zz_code_list = []
    for zz_code_data_path in zz_code_data_paths:
        if not os.path.exists(zz_code_data_path):
            print(f'{zz_code_data_path} 不存在')
            continue
        zz_code_df = pd.read_csv(zz_code_data_path)
        zz_code_list += zz_code_df['type'].tolist()

    temp_stock_list = ['sh.000300', 'sh.000016', 'sh.000852']
    for i, stock_code in enumerate(os.listdir(zh_data_dir)):
        # file_path = f'{zh_data_dir}/{stock_code}/daily.csv'
        file_path_a = f'{zh_data_dir}/{stock_code}/daily_a.csv'
        income_path = f'{financial_data_dir}/{stock_code}/income.csv'
        income_gbjg_path = f'{financial_data_dir}/{stock_code}/income_gbjg.csv'


        # 过滤创业板/科创板/北交所股票
        if ('.30' in stock_code
                or '.68' in stock_code
                or '.8' in stock_code
                or '.4' in stock_code):
            print(f'过滤创业板/科创板/北交所股票: {stock_code}')
            continue
        # 使用指数成分股股票回测
        # if stock_code not in zz_code_list and stock_code not in temp_stock_list:
        #     print(f'过滤非指数成分股股票: {stock_code}')
        #     continue

        if not os.path.exists(file_path_a):
            continue

        print(file_path_a)
        df = pd.read_csv(file_path_a)
        df.rename(columns={'isST': 'is_st', }, inplace=True)
        df['close_1'] = df['close']
        # df_a = pd.read_csv(file_path_a)[['date', 'close']].rename(columns={'close': 'close_1'})
        # df = pd.merge(df, df_a, on='date', how='inner')
        df['factor'] = 1.0
        df['date'] = pd.to_datetime(df['date'])
        # df = df[df['date'] >= from_date]
        df = df.sort_values('date')

        # 过滤上市时间太短的股票 （A 股一年交易时间243天），取上市一年多的股票
        if len(df) < 275:
            print(f'{stock_code} 上市交易时间太短，交易的天数: {len(df)}，忽略该股票')
            continue

        if os.path.exists(income_path):

            financial_df = pd.read_csv(income_path)
            if os.path.exists(income_gbjg_path):
                income_gbjg_df = pd.read_csv(income_gbjg_path)[['变更日期', '总股本']]
                income_gbjg_df.rename(columns={'变更日期': 'date', '总股本': 'totalShare_new'}, inplace=True)
            else:
                income_gbjg_df = None

            quarterly_df, annual_df = process_financial_data(financial_df)
            df_temp = merge_with_stock(df, quarterly_df, annual_df, income_gbjg_df)
            if 'totalShare_new' not in df_temp.columns:
                df_temp['totalShare_new'] = df_temp['totalShare_y']
            df2_sorted = df_temp.sort_values('date').ffill().dropna()

            df = df2_sorted

            df['mv'] = df['totalShare_new'] * df['close_1']  # 市值 = 总股本 * 收盘价（不复权）

            df['openinterest'] = 0
            df['date'] = pd.to_datetime(df['date'])

        else:
            for col in add_cols:
                if col not in df.columns:
                    df[col] = 0

        try:
            # 选择需要的列
            df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'mv', 'is_st', 'profit_ttm_y',
                     'profit_y', 'revenue_y', 'roeAvg_y', 'profit_ttm_q', 'profit_q', 'revenue_single_q', 'roeAvg_q',
                     'openinterest', ]]
            df = process_dataframe(df)
            print(f'✅ 处理 {stock_code} 成功 {len(df)} 行数据')
            result[stock_code] = df
        except Exception as e:
            print(f'❌ 处理 {stock_code} 失败: {e}')

    return result, calendar_df['date'].sort_values().unique()[-1]