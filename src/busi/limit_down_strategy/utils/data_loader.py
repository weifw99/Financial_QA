# utils/data_loader.py
# 封装股票与指数的 CSV 数据加载，注入自定义字段：市值、利润、营收、ST
import os
from datetime import timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import backtrader as bt

class CustomPandasData(bt.feeds.PandasData):
    """
    自定义数据类，包含：市值、市盈率、利润、营收、是否ST标记等基本面数据
    需要保证df中有以下字段：datetime, open, high, low, close, volume, mv, profit, revenue, is_st
    """

    lines = ('amount', 'turn', 'mv', 'lt_mv', 'lt_share_rate',  'is_st', 'profit_ttm_y', 'profit_y', 'revenue_y', 'roeAvg_y', 'profit_ttm_q', 'profit_q', 'revenue_single_q', 'roeAvg_q', 'price_limit')
    params = (
        ('amount', -1),
        ('turn', -1),
        ('mv', -1),
        ('lt_mv', -1),
        ('lt_share_rate', -1),
        ('is_st', -1),# 0 or 1 表示是否ST
        ('profit_ttm_y', -1),
        ('profit_y', -1),
        ('revenue_y', -1),
        ('roeAvg_y', -1),  #

        ('profit_ttm_q', -1),
        ('profit_q', -1),
        ('revenue_single_q', -1),
        ('roeAvg_q', -1),  #
        ('price_limit', -1),  #

        ('dtformat', '%Y-%m-%d'),
    )
import pandas as pd
import numpy as np

def calc_MBRevenue_single_via_rule(df: pd.DataFrame) -> pd.DataFrame:
    """
    按照固定规则补充 MBRevenue_single：
    - Q1（3月）：Q2值 / 2
    - Q2（6月）：Q2值 / 2
    - Q3（9月）：(Q4 - Q2) / 2
    - Q4（12月）：(Q4 - Q2) / 2

    要求 df 包含 ['statDate', 'MBRevenue']，每行是一个季度记录。
    """

    df = df.copy()
    df['statDate'] = pd.to_datetime(df['statDate'])
    df = df.sort_values('statDate').reset_index(drop=True)
    df['year'] = df['statDate'].dt.year
    df['month'] = df['statDate'].dt.month
    df['MBRevenue'] = pd.to_numeric(df['MBRevenue'], errors='coerce')
    df['MBRevenue_single'] = np.nan

    for year, group in df.groupby('year'):
        group = group.set_index('month')

        try:
            q2_val = group.at[6, 'MBRevenue']
            q4_val = group.at[12, 'MBRevenue']

            # Q1（3月）：若存在06-30，Q2值 / 2
            if 3 in group.index:
                idx = df[(df['year'] == year) & (df['month'] == 3)].index[0]
                if pd.notna(q2_val):
                    df.at[idx, 'MBRevenue_single'] = q2_val / 2

            # Q2（6月）：Q2值 / 2
            if 6 in group.index:
                idx = df[(df['year'] == year) & (df['month'] == 6)].index[0]
                if pd.notna(q2_val):
                    df.at[idx, 'MBRevenue_single'] = q2_val / 2

            # Q3/Q4：如果 Q2/Q4 都有值，则 Q3/Q4 = (Q4 - Q2) / 2
            if pd.notna(q2_val) and pd.notna(q4_val):
                delta = q4_val - q2_val

                if 9 in group.index:
                    idx = df[(df['year'] == year) & (df['month'] == 9)].index[0]
                    df.at[idx, 'MBRevenue_single'] = delta / 2

                if 12 in group.index:
                    idx = df[(df['year'] == year) & (df['month'] == 12)].index[0]
                    df.at[idx, 'MBRevenue_single'] = delta / 2

        except KeyError:
            continue  # 本年季度数据不全，跳过

    return df.drop(columns=['year', 'month'])
def process_financial_data(financial_df: pd.DataFrame):
    """
    输入原始财报数据，输出：
    - 年度财报（添加 apply_year 字段）
    - 季度财报（添加 apply_quarter 字段）
    """
    df = calc_MBRevenue_single_via_rule( financial_df )
    df.rename(columns={'netProfit': 'profit', 'MBRevenue': 'revenue','MBRevenue_single': 'revenue_single', }, inplace=True)
    df['revenue'] = df['revenue'].ffill()
    df['pubDate'] = pd.to_datetime(df['pubDate'])
    df['statDate'] = pd.to_datetime(df['statDate'])

    # 归属母公司股东的净利润TTM
    # epsTTM	每股收益	归属母公司股东的净利润TTM/最新总股本
    df['profit_ttm'] = df['totalShare'] * df['epsTTM']
    # pubDate	公司发布财报的日期
    # roeAvg	净资产收益率(平均)(%)	归属母公司股东净利润/[(期初归属母公司股东的权益+期末归属母公司股东的权益)/2]*100%
    # statDate	财报统计的季度的最后一天, 比如2017-03-31, 2017-06-30
    # netProfit	净利润(元)
    # MBRevenue	主营营业收入(元)
    # mv 市值
    # 使用 pd.merge_asof 实现按时间向前填充匹配
    # profit_ttm 归属母公司股东的净利润TTM

    # 年度判断：12月31日的 statDate 为年报
    is_annual = df['statDate'].dt.month == 12

    annual_df = df[is_annual].copy()
    quarterly_df = df.copy()

    # 添加年度财报适用年
    annual_df['apply_year'] = annual_df['statDate'].dt.year + 1

    # 添加季度财报适用季度（季度末日期）
    def get_next_quarter(stat_date):
        y, m = stat_date.year, stat_date.month
        if m <= 3:
            return pd.Timestamp(f"{y}-06-30")
        elif m <= 6:
            return pd.Timestamp(f"{y}-09-30")
        elif m <= 9:
            return pd.Timestamp(f"{y}-12-31")
        else:
            return pd.Timestamp(f"{y+1}-03-31")

    quarterly_df['apply_quarter'] = quarterly_df['statDate'].apply(get_next_quarter)

    # 后缀字段名
    suffix_q = {col: f"{col}_q" for col in df.columns if col not in ['code', 'apply_quarter', 'statDate', 'pubDate']}
    suffix_y = {col: f"{col}_y" for col in df.columns if col not in ['code', 'apply_year', 'statDate', 'pubDate']}

    # 重命名字段（便于后续合并时区分）
    quarterly_df = quarterly_df.rename(columns=suffix_q)
    annual_df = annual_df.rename(columns=suffix_y)

    return quarterly_df, annual_df


def merge_with_stock(stock_df: pd.DataFrame, quarterly_df: pd.DataFrame, annual_df: pd.DataFrame, df_shares: pd.DataFrame):
    """
    stock_df 包含字段 ['code', 'trade_date']（datetime 类型）
    合并年度与季度财报数据，生成完整对齐 DataFrame
    """

    df = stock_df.copy()
    df['date'] = pd.to_datetime(df['date'])

    # 获取季度末日期
    def get_quarter_end(date):
        y, m = date.year, date.month
        if m <= 3:
            return pd.Timestamp(f"{y}-03-31")
        elif m <= 6:
            return pd.Timestamp(f"{y}-06-30")
        elif m <= 9:
            return pd.Timestamp(f"{y}-09-30")
        else:
            return pd.Timestamp(f"{y}-12-31")

    df['quarter_end'] = df['date'].apply(get_quarter_end)
    df['year'] = df['date'].dt.year

    # 1. 合并季度数据
    df = df.merge(
        quarterly_df,
        how='left',
        left_on=['code', 'quarter_end'],
        right_on=['code', 'apply_quarter']
    )

    # 2. 合并年度数据
    df = df.merge(
        annual_df,
        how='left',
        left_on=['code', 'year'],
        right_on=['code', 'apply_year']
    )

    # 假设 df_stock 是股票行情数据，有 'date' 列
    # 假设 df_shares 是总股本数据，有 'date' 和 'totalShare' 列

    if df_shares is not None:
        # 确保日期列是 datetime 类型
        df['date'] = pd.to_datetime(df['date'])
        df_shares['date'] = pd.to_datetime(df_shares['date'])

        # 按日期排序（merge_asof 要求先排序）
        df = df.sort_values('date')
        df_shares = df_shares.sort_values('date')

        # 向前填充最近的总股数（不晚于该交易日）
        df_merged = pd.merge_asof(
            df,
            df_shares,
            on='date',
            direction='backward'  # 向前查找最近的股本数据
        )

        return df_merged
    else:
        return df


def merge_stock_with_industry(stock_df: pd.DataFrame, industry_df: pd.DataFrame) -> pd.DataFrame:
    """
    将股票交易数据与行业分类数据进行左关联（保留原始 code，并创建匹配用的 code）

    参数：
        stock_df: 股票交易数据，包含 code（格式如 sh.000001）
        industry_dir: 行业数据 CSV 文件所在目录，文件结构为 code,name,industry_code,industry_name

    返回：
        合并后的 DataFrame（附加字段：industry_code, industry_name）
    """

    # 提取股票原始 code，并转为 6 位匹配值
    stock_df = stock_df.copy()
    stock_df['stock_code_raw'] = stock_df['code']  # 原始如 sh.000002

    stock_df['stock_code_raw'] = stock_df['stock_code_raw'].apply(lambda x: str(x)[-6:])  # 提取 000002 用于匹配

    industry_df.rename(columns={'code': 'code_indu'}, inplace=True)
    # 4. 合并行业数据（左连接）
    merged_df = pd.merge(
        stock_df,
        industry_df[['code_indu', 'industry_code', 'industry_name']],
        how='left',
        left_on='stock_code_raw',
        right_on='code_indu'
    )

    merged_df.fillna(0, inplace=True)

    return merged_df


def load_stock_data(from_idx, to_idx):
    """
    批量加载 data_dir 下的所有 CSV 文件，返回数据列表
    文件名将作为数据名称注入，如 '600000.csv' -> data._name = '600000'
    :param data_dir: 包含CSV的路径
    :return: list of data feeds
    """
    zz_code_data_paths = [
        # '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/raw/index/中小板指数-中小100-399005.csv',
        '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/raw/index/中小综指-399101.csv',
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

    datas = []

    base_data_path = '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data'
    zh_data_dir = Path(base_data_path) / 'market'
    financial_data_dir = Path(base_data_path).parent / 'zh_data/financial'
    board_industry_dir = Path(base_data_path).parent / 'zh_data/industry/board_industry'


    # 1. 找到行业目录中最新的CSV文件
    files = [f for f in os.listdir(board_industry_dir) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"⚠️ 行业目录中没有找到CSV文件: {board_industry_dir}")
    files.sort(key=lambda f: os.path.getmtime(os.path.join(board_industry_dir, f)), reverse=True)
    latest_file = os.path.join(board_industry_dir, files[0])
    print(f"📄 使用行业文件: {latest_file}")

    # 2. 读取行业数据
    industry_df = pd.read_csv(latest_file, dtype={'code': str})
    industry_df = industry_df[['code', 'name', 'industry_code', 'industry_name']]

    # 获取所有时间数据， 使用000001.csv
    pdf = pd.read_csv(f'{zh_data_dir}/sh.000001/daily.csv')
    pdf['date'] = pd.to_datetime(pdf['date'])

    from_date = from_idx - timedelta(days=40)
    pdf = pdf[pdf['date'] >= from_date]
    data = pd.DataFrame(index=pdf['date'].unique())
    data = data.sort_index()

    select_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', ]
    add_cols = ['industry_name', 'amount', 'turn', 'mv', 'lt_mv', 'lt_share_rate',   'is_st', 'profit_ttm_y', 'profit_y', 'revenue_y', 'roeAvg_y', 'profit_ttm_q', 'profit_q', 'revenue_single_q', 'roeAvg_q', 'openinterest', ]
    # 加载 SZ510880 SH159300
    etf_list = ['SZ510880', 'SH159919', 'SZ510050', 'SZ588000', 'SZ511880']
    etf_path = '/Users/dabai/liepin/study/llm/Financial_QA/src/busi/etf_/data/etf_trading/daily'
    for etf_code in etf_list:
        etf_df = pd.read_csv(f'{etf_path}/{etf_code}.csv')
        # 选择需要的列
        etf_df = etf_df[select_cols]
        for col in add_cols:
            if col not in etf_df.columns:
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


    index_list =['csi932000', 'sz399101' , 'sh000905', 'sh000852', 'sh000046', 'sz399005', 'sz399008', 'sz399401',
                 'sz399649','sz399663','sz399377','sh000046','sz399408','sz399401','sh000991' ,
                 'sh000852', 'sz399004', 'sh000905', 'sz399006',
                 'sz399693']
    # 获取指数数据
    zz_path = '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/index'

    for index_code in index_list:

        zz_df = pd.read_csv(f'{zz_path}/{index_code}.csv')
        # 选择需要的列
        zz_df = zz_df[select_cols[:-1]]
        for col in add_cols:
            if col not in zz_df.columns:
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

    temp_stock_list = ['sh.000300',  'sh.000016', 'sh.000852', 'BK1158', ]
    for i, stock_file in enumerate(os.listdir(zh_data_dir)):
        # if i > 500:
        #     break

        # 测试
        # if len(datas) >100 and stock_file  not in temp_stock_list:
        #     continue

        # 使用指数成分股股票回测
        if stock_file not in zz_code_list and stock_file not in temp_stock_list:
            print(f'过滤非指数成分股股票: {stock_file}')
            continue # 0.1945 sz399101成分股,
            # 指数的成分股数据 0.10   中证1000-000852，中证2000-932000
            # 指数的成分股数据 0.158  中小综指-399101,中证1000-000852，中证2000-932000
            # 指数的成分股数据 0.1945  中小综指-399101,中证1000-000852
            # 指数的成分股数据 0.158  中小综指-399101,中证2000-932000
            # 指数的成分股数据 0.10,中证2000-932000
        # 0.2137，全部数据
        # 过滤创业板/科创板/北交所股票
        if ('.30' in stock_file
                or '.68' in stock_file
                or '.8' in stock_file
                or '.4' in stock_file):
            print(f'过滤创业板/科创板/北交所股票: {stock_file}')
            continue

        print(f'{i}/{stock_file}')
        # file_path = f'{zh_data_dir}/{stock_file}/daily.csv'
        file_path_a = f'{zh_data_dir}/{stock_file}/daily_a.csv'

        # 获取财务盈利信息
        financial_path = f'{financial_data_dir}/{stock_file}/income.csv'
        income_gbjg_path = f'{financial_data_dir}/{stock_file}/income_gbjg.csv'
        if os.path.exists(file_path_a):
            df = pd.read_csv(file_path_a)
            if 'code' not in df.columns:
                print(f'{stock_file} 缺少列: {col}')
                df['code'] = stock_file

            # df = merge_stock_with_industry(df, industry_df)

            # 过滤上市时间太短的股票 （A 股一年交易时间243天），取上市一年多的股票
            if len(df) < 275:
                print(f'{stock_file} 上市交易时间太短，交易的天数: {len(df)}，忽略该股票')
                continue

            # df_a = pd.read_csv(file_path_a)[['date','close']]
            # df_a.rename(columns={'close': 'close_1'}, inplace=True)

            # df = pd.merge(df, df_a, on='date', how='inner')
            df['close_1'] = df['close']

            # 使用后复权价格，factor均设置为1， 回测使用该因子
            df['factor'] = 1.0
            # 确保 date 列为 datetime 类型并排序
            df['date'] = pd.to_datetime(df['date'])
            df_sorted = df.sort_values('date')
            df_sorted.rename(columns={'isST': 'is_st', }, inplace=True)

            if os.path.exists(financial_path):

                financial_df = pd.read_csv(financial_path)
                if os.path.exists(income_gbjg_path):
                    income_gbjg_df = pd.read_csv(income_gbjg_path)[['变更日期','总股本', '已上市流通A股']]
                    income_gbjg_df.rename(columns={'变更日期': 'date', '总股本': 'totalShare_new', '已上市流通A股': 'liqaShare_a'}, inplace=True)
                else:
                    income_gbjg_df = None

                quarterly_df, annual_df = process_financial_data(financial_df)

                df_temp = merge_with_stock(df_sorted, quarterly_df, annual_df, income_gbjg_df)
                if 'totalShare_new' not in df_temp.columns:
                    df_temp['totalShare_new'] = df_temp['totalShare_q']

                if 'liqaShare_a' not in df_temp.columns:
                    df_temp['liqaShare_a'] = df_temp['liqaShare_q']

                df2_sorted = df_temp.sort_values('date').ffill().dropna()

                df = df2_sorted

                df['mv'] = df['totalShare_new'] * df['close_1'] # 市值 = 总股本 * 收盘价（不复权）
                df['lt_mv'] = df['liqaShare_a'] * df['close_1'] # 市值 = 已上市流通A股 * 收盘价（不复权）
                df['lt_share_rate'] = df['liqaShare_a'] / df['totalShare_new'] #  流通A股占比

                df['openinterest'] = 0
                df['date'] = pd.to_datetime(df['date'])
                # 价格涨跌幅度
                df['price_limit'] = (df['close'] - df['open']) / df['open'] * 100

                # 选择需要的列
                df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'mv', 'lt_mv', 'lt_share_rate',   'is_st', 'profit_ttm_y', 'profit_y', 'revenue_y', 'roeAvg_y', 'profit_ttm_q', 'profit_q', 'revenue_single_q', 'roeAvg_q', 'price_limit', 'openinterest', ]]

                df.set_index('date', inplace=True)  # 设置 datetime 为索引
                df = df.sort_index()

                data_ = pd.merge(data, df, left_index=True, right_index=True, how='left')
                data_ = data_.sort_index()  # ✅ 强制升序
                # 检查并填充关键列
                # required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'mv', 'lt_mv', 'lt_share_rate',  'profit', 'revenue', 'is_st']
                # for col in required_cols:
                #     if col not in df.columns:
                #         raise ValueError(f"缺失字段：{col} in {stock_file}")
                # df = df[required_cols]

                # data_ = df.sort_index()
                data_.loc[:, ['volume', 'openinterest']] = data_.loc[:, ['volume', 'openinterest']].fillna(0)
                data_.loc[:, ['open', 'high', 'low', 'close']] = data_.loc[:, ['open', 'high', 'low', 'close', ]].bfill()
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
                    if col not in df_sorted.columns:
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



def load_stock_data_df(from_idx, to_idx):
    """
    批量加载 data_dir 下的所有 CSV 文件，返回数据列表
    文件名将作为数据名称注入，如 '600000.csv' -> data._name = '600000'
    :param data_dir: 包含CSV的路径
    :return: list of data feeds
    """
    zz_code_data_paths = [
        # '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/raw/index/中小板指数-中小100-399005.csv',
        '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/raw/index/中小综指-399101.csv',
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

    datas = []

    base_data_path = '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data'
    zh_data_dir = Path(base_data_path) / 'market'
    financial_data_dir = Path(base_data_path).parent / 'zh_data/financial'
    board_industry_dir = Path(base_data_path).parent / 'zh_data/industry/board_industry'


    # 1. 找到行业目录中最新的CSV文件
    files = [f for f in os.listdir(board_industry_dir) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"⚠️ 行业目录中没有找到CSV文件: {board_industry_dir}")
    files.sort(key=lambda f: os.path.getmtime(os.path.join(board_industry_dir, f)), reverse=True)
    latest_file = os.path.join(board_industry_dir, files[0])
    print(f"📄 使用行业文件: {latest_file}")

    # 2. 读取行业数据
    industry_df = pd.read_csv(latest_file, dtype={'code': str})
    industry_df = industry_df[['code', 'name', 'industry_code', 'industry_name']]

    # 获取所有时间数据， 使用000001.csv
    pdf = pd.read_csv(f'{zh_data_dir}/sh.000001/daily.csv')
    pdf['date'] = pd.to_datetime(pdf['date'])

    from_date = from_idx - timedelta(days=40)
    pdf = pdf[pdf['date'] >= from_date]
    data = pd.DataFrame(index=pdf['date'].unique())
    data = data.sort_index()

    select_cols = ['date', 'open', 'high', 'low', 'close', 'volume', 'amount', ]
    add_cols = ['industry_name', 'amount', 'turn', 'mv', 'lt_mv', 'lt_share_rate',   'is_st', 'profit_ttm_y', 'profit_y', 'revenue_y', 'roeAvg_y', 'profit_ttm_q', 'profit_q', 'revenue_single_q', 'roeAvg_q', 'openinterest', ]
    # 加载 SZ510880 SH159300
    etf_list = ['SZ510880', 'SH159919', 'SZ510050', 'SZ588000', 'SZ511880']
    etf_path = '/Users/dabai/liepin/study/llm/Financial_QA/src/busi/etf_/data/etf_trading/daily'
    for etf_code in etf_list:
        etf_df = pd.read_csv(f'{etf_path}/{etf_code}.csv')
        # 选择需要的列
        etf_df = etf_df[select_cols]
        for col in add_cols:
            if col not in etf_df.columns:
                etf_df[col] = 0
        etf_df['date'] = pd.to_datetime(etf_df['date'])
        etf_df['date1'] = etf_df['date']
        etf_df.set_index('date', inplace=True)  # 设置 datetime 为索引
        etf_df = etf_df.sort_index()
        data_ = pd.merge(data, etf_df, left_index=True, right_index=True, how='left')
        data_.fillna(0, inplace=True)
        data_ = data_.sort_index()  # ✅ 强制升序
        datas.append({'code': etf_code, 'data': data_})


    index_list =['csi932000', 'sz399101' , 'sh000905', 'sh000852', 'sh000046', 'sz399005', 'sz399008', 'sz399401',
                 'sz399649','sz399663','sz399377','sh000046','sz399408','sz399401','sh000991' ,
                 'sh000852', 'sz399004', 'sh000905', 'sz399006',
                 'sz399693']
    # 获取指数数据
    zz_path = '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/index'

    for index_code in index_list:

        zz_df = pd.read_csv(f'{zz_path}/{index_code}.csv')
        # 选择需要的列
        zz_df = zz_df[select_cols[:-1]]
        for col in add_cols:
            if col not in zz_df.columns:
                zz_df[col] = 0
        zz_df['date'] = pd.to_datetime(zz_df['date'])
        zz_df['date1'] = zz_df['date']
        zz_df.set_index('date', inplace=True)  # 设置 datetime 为索引
        zz_df = zz_df.sort_index()
        data_ = pd.merge(data, zz_df, left_index=True, right_index=True, how='left')
        data_.fillna(0, inplace=True)
        data_ = data_.sort_index()  # ✅ 强制升序
        datas.append({'code': index_code, 'data': data_})

    temp_stock_list = ['sh.000300',  'sh.000016', 'sh.000852', 'BK1158', ]
    for i, stock_file in enumerate(os.listdir(zh_data_dir)):
        # if i > 500:
        #     break

        # 测试
        # if len(datas) >100 and stock_file  not in temp_stock_list:
        #     continue

        # 使用指数成分股股票回测
        if stock_file not in zz_code_list and stock_file not in temp_stock_list:
            print(f'过滤非指数成分股股票: {stock_file}')
            continue # 0.1945 sz399101成分股,
            # 指数的成分股数据 0.10   中证1000-000852，中证2000-932000
            # 指数的成分股数据 0.158  中小综指-399101,中证1000-000852，中证2000-932000
            # 指数的成分股数据 0.1945  中小综指-399101,中证1000-000852
            # 指数的成分股数据 0.158  中小综指-399101,中证2000-932000
            # 指数的成分股数据 0.10,中证2000-932000
        # 0.2137，全部数据
        # 过滤创业板/科创板/北交所股票
        if ('.30' in stock_file
                or '.68' in stock_file
                or '.8' in stock_file
                or '.4' in stock_file):
            print(f'过滤创业板/科创板/北交所股票: {stock_file}')
            continue

        print(f'{i}/{stock_file}')
        # file_path = f'{zh_data_dir}/{stock_file}/daily.csv'
        file_path_a = f'{zh_data_dir}/{stock_file}/daily_a.csv'

        # 获取财务盈利信息
        financial_path = f'{financial_data_dir}/{stock_file}/income.csv'
        income_gbjg_path = f'{financial_data_dir}/{stock_file}/income_gbjg.csv'
        if os.path.exists(file_path_a):
            df = pd.read_csv(file_path_a)
            if 'code' not in df.columns:
                print(f'{stock_file} 缺少列: {col}')
                df['code'] = stock_file

            # df = merge_stock_with_industry(df, industry_df)

            # 过滤上市时间太短的股票 （A 股一年交易时间243天），取上市一年多的股票
            if len(df) < 275:
                print(f'{stock_file} 上市交易时间太短，交易的天数: {len(df)}，忽略该股票')
                continue

            # df_a = pd.read_csv(file_path_a)[['date','close']]
            # df_a.rename(columns={'close': 'close_1'}, inplace=True)

            # df = pd.merge(df, df_a, on='date', how='inner')
            df['close_1'] = df['close']

            # 使用后复权价格，factor均设置为1， 回测使用该因子
            df['factor'] = 1.0
            # 确保 date 列为 datetime 类型并排序
            df['date'] = pd.to_datetime(df['date'])
            df_sorted = df.sort_values('date')
            df_sorted.rename(columns={'isST': 'is_st', }, inplace=True)

            if os.path.exists(financial_path):

                financial_df = pd.read_csv(financial_path)
                if os.path.exists(income_gbjg_path):
                    income_gbjg_df = pd.read_csv(income_gbjg_path)[['变更日期','总股本', '已上市流通A股']]
                    income_gbjg_df.rename(columns={'变更日期': 'date', '总股本': 'totalShare_new', '已上市流通A股': 'liqaShare_a'}, inplace=True)
                else:
                    income_gbjg_df = None

                quarterly_df, annual_df = process_financial_data(financial_df)

                df_temp = merge_with_stock(df_sorted, quarterly_df, annual_df, income_gbjg_df)
                if 'totalShare_new' not in df_temp.columns:
                    df_temp['totalShare_new'] = df_temp['totalShare_q']

                if 'liqaShare_a' not in df_temp.columns:
                    df_temp['liqaShare_a'] = df_temp['liqaShare_q']

                df2_sorted = df_temp.sort_values('date').ffill().dropna()

                df = df2_sorted

                df['mv'] = df['totalShare_new'] * df['close_1'] # 市值 = 总股本 * 收盘价（不复权）
                df['lt_mv'] = df['liqaShare_a'] * df['close_1'] # 市值 = 已上市流通A股 * 收盘价（不复权）
                df['lt_share_rate'] = df['liqaShare_a'] / df['totalShare_new'] #  流通A股占比

                df['openinterest'] = 0
                df['date'] = pd.to_datetime(df['date'])
                # 价格涨跌幅度
                df['price_limit'] = (df['close'] - df['open']) / df['open'] * 100

                # 选择需要的列
                df = df[['date', 'open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'mv', 'lt_mv', 'lt_share_rate',   'is_st', 'profit_ttm_y', 'profit_y', 'revenue_y', 'roeAvg_y', 'profit_ttm_q', 'profit_q', 'revenue_single_q', 'roeAvg_q', 'price_limit', 'openinterest', ]]
                df['date1'] = df['date']
                df.set_index('date', inplace=True)  # 设置 datetime 为索引
                df = df.sort_index()

                data_ = pd.merge(data, df, left_index=True, right_index=True, how='left')
                data_ = data_.sort_index()  # ✅ 强制升序
                # 检查并填充关键列
                # required_cols = ['datetime', 'open', 'high', 'low', 'close', 'volume', 'mv', 'lt_mv', 'lt_share_rate',  'profit', 'revenue', 'is_st']
                # for col in required_cols:
                #     if col not in df.columns:
                #         raise ValueError(f"缺失字段：{col} in {stock_file}")
                # df = df[required_cols]

                # data_ = df.sort_index()
                data_.loc[:, ['volume', 'openinterest']] = data_.loc[:, ['volume', 'openinterest']].fillna(0)
                data_.loc[:, ['open', 'high', 'low', 'close']] = data_.loc[:, ['open', 'high', 'low', 'close', ]].bfill()
                data_.bfill(inplace=True)
                data_.fillna(0, inplace=True)
                rsub_cols = [ 'open', 'high', 'low', 'close', ]

                data_.dropna(subset=rsub_cols, inplace=True)

                # print("最终合并后的 data_ 形状:", data_.shape)
                # print("缺失字段统计:\n", data_.isnull().sum())
                # print("close 列前5行:\n", data_['close'].head())

                # if df.empty or len(df) < 100:
                #     continue

                # data._name = stock_file.replace('.csv', '')  # 设置数据名称（用于后续匹配指数名等）
                # print(f'添加数据源：{data._name}，数据日期范围：{df["datetime"].min()} ~ {df["datetime"].max()}，共 {len(df)} 条记录')
                datas.append({'code': stock_file, 'data': data_})
            else:
                print(f'{stock_file} 缺少财务信息')
                # 选择需要的列
                df_sorted = df_sorted[select_cols]
                for col in add_cols:
                    if col not in df_sorted.columns:
                        df_sorted[col] = 0

                df_sorted['date1'] = df_sorted['date']
                df_sorted.set_index('date', inplace=True)  # 设置 datetime 为索引
                df_sorted = df_sorted.sort_index()
                data_ = pd.merge(data, df_sorted, left_index=True, right_index=True, how='left')
                data_.fillna(0, inplace=True)
                data_ = data_.sort_index()  # ✅ 强制升序


                datas.append({'code': stock_file, 'data': data_})

    return datas

