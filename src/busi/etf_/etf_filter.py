import akshare as ak
import pandas as pd
from datetime import datetime, timedelta
from scipy.stats import linregress
import numpy as np
import os

from busi.etf_.etf_data import EtfDataHandle


def get_etf_list():
    df = EtfDataHandle.get_and_download_etf_info()
    # df = df[df['类型'] == '场内ETF']
    # print(df.columns)
    # 代码, 名称, 最新价, IOPV实时估值, 基金折价率, 涨跌额, 涨跌幅, 成交量, 成交额, 开盘价, 最高价, 最低价, 昨收, 振幅, 换手率, 量比, 委比, 外盘, 内盘, 主力净流入 - 净额, 主力净流入 - 净占比, 超大单净流入 - 净额, 超大单净流入 - 净占比, 大单净流入 - 净额, 大单净流入 - 净占比, 中单净流入 - 净额, 中单净流入 - 净占比, 小单净流入 - 净额, 小单净流入 - 净占比, 现手, 买一, 卖一, 最新份额, 流通市值, 总市值, 数据日期, 更新时间

    df['基金代码'] = df['代码']
    df['基金简称'] = df['名称']
    df['基金规模'] = df['总市值']
    return df


def get_price_series(code, days=180):
    end = datetime.today()
    start = end - timedelta(days=days)
    try:
        df = ak.fund_etf_hist_em(code, start_date=start.strftime('%Y%m%d'), end_date=end.strftime('%Y%m%d'))
        df = df[['日期', '收盘']]
        df.columns = ['date', 'close']
        df['date'] = pd.to_datetime(df['date'])
        df.sort_values('date', inplace=True)
        return df
    except:
        return None


def calc_simple_returns(df):
    try:
        price_start = df.iloc[0]['close']
        price_end = df.iloc[-1]['close']
        return (price_end - price_start) / price_start
    except:
        return None


def calc_r2_slope(df):
    try:
        y = df['close'].values
        x = np.arange(len(y))
        slope, intercept, r_value, p_value, std_err = linregress(x, y)
        return slope * (r_value ** 2)
    except:
        return None


def filter_etf(top_k=50, method='simple'):
    df = get_etf_list()
    df = df.dropna()
    '''
    成立时间 大于 1 年
    日均成交额500w
    日均成交量10w
    资产规模10亿
    '''
    # df = df[(df['基金规模'] > 1) & (df['成交额'] > 500)]
    df = df[
        (df["成交额"] > 1e8) &  # 成交额 > 1亿元
        (df["换手率"] > 0.5) &  # 换手率 > 0.5%
        (df["基金折价率"].between(-2, 2)) &  # 折价率在±2%以内
        (df["总市值"] > 1e9) &  # 总市值 > 10亿
        (df["流通市值"] > 5e8) &  # 流通市值 > 5亿
        (df["主力净流入-净额"] > 0)  # 主力净流入为正
        ]

    df = df.reset_index(drop=True)

    print( len(df) )

    scores = []
    for _, row in df.iterrows():
        code = row['基金代码']
        name = row['基金简称']
        price_df = get_price_series(code)

        if price_df is None or len(price_df) < 60:
            continue

        if method == 'simple':
            r3m = calc_simple_returns(get_price_series(code, 90))
            r6m = calc_simple_returns(get_price_series(code, 180))
            r12m = calc_simple_returns(get_price_series(code, 365))
            if None in (r3m, r6m, r12m):
                continue
            score = 0.3 * r3m + 0.3 * r6m + 0.4 * r12m
        elif method == 'r2':
            r6m = calc_simple_returns(price_df)
            r2score = calc_r2_slope(price_df)
            if r6m is None or r2score is None:
                continue
            score = r6m * r2score
        else:
            raise ValueError("Unknown method")

        scores.append({
            '基金代码': code,
            '基金简称': name,
            '基金规模': row['基金规模'],
            '动量得分': score
        })

    df_result = pd.DataFrame(scores)
    df_result = df_result.sort_values('动量得分', ascending=False).reset_index(drop=True)
    return df_result.head(top_k)

if __name__ == '__main__':
    print("方式一：简单收益率动量")
    simple_df = filter_etf(method='simple')
    simple_df.to_csv("data/filter/top_etf_simple.csv", index=False)

    print("方式二：收益率 × R² 动量")
    r2_df = filter_etf(method='r2')
    r2_df.to_csv("data/filter/top_etf_r2.csv", index=False)