import os
import random
import time

import pandas as pd

import akshare as ak
import requests


def stock_board_industry_hist_em(
    symbol: str = "小金属",
    start_date: str = "20211201",
    end_date: str = "20500101",
    period: str = "日k",
    adjust: str = "",
) -> pd.DataFrame:
    """
    东方财富网-沪深板块-行业板块-历史行情
    https://quote.eastmoney.com/bk/90.BK1027.html
    :param symbol: 板块名称
    :type symbol: str
    :param start_date: 开始时间
    :type start_date: str
    :param end_date: 结束时间
    :type end_date: str
    :param period: 周期; choice of {"日k", "周k", "月k"}
    :type period: str
    :param adjust: choice of {'': 不复权, "qfq": 前复权, "hfq": 后复权}
    :type adjust: str
    :return: 历史行情
    :rtype: pandas.DataFrame
    """
    em_code = symbol
    period_map = {
        "日k": "101",
        "周k": "102",
        "月k": "103",
    }
    adjust_map = {"": "0", "qfq": "1", "hfq": "2"}

    url = "https://push2his.eastmoney.com/api/qt/stock/kline/get"
    params = {
        "secid": f"90.{em_code}",
        "fields1": "f1,f2,f3,f4,f5,f6",
        "fields2": "f51,f52,f53,f54,f55,f56,f57,f58,f59,f60,f61",
        "klt": period_map[period],
        "fqt": adjust_map[adjust],
        "beg": start_date,
        "end": end_date,
        "smplmt": "10000",
        "lmt": "1000000",
    }
    r = requests.get(url, params=params)
    data_json = r.json()
    temp_df = pd.DataFrame([item.split(",") for item in data_json["data"]["klines"]])
    temp_df.columns = [
        "日期",
        "开盘",
        "收盘",
        "最高",
        "最低",
        "成交量",
        "成交额",
        "振幅",
        "涨跌幅",
        "涨跌额",
        "换手率",
    ]
    temp_df = temp_df[
        [
            "日期",
            "开盘",
            "收盘",
            "最高",
            "最低",
            "涨跌幅",
            "涨跌额",
            "成交量",
            "成交额",
            "振幅",
            "换手率",
        ]
    ]
    temp_df["开盘"] = pd.to_numeric(temp_df["开盘"], errors="coerce")
    temp_df["收盘"] = pd.to_numeric(temp_df["收盘"], errors="coerce")
    temp_df["最高"] = pd.to_numeric(temp_df["最高"], errors="coerce")
    temp_df["最低"] = pd.to_numeric(temp_df["最低"], errors="coerce")
    temp_df["涨跌幅"] = pd.to_numeric(temp_df["涨跌幅"], errors="coerce")
    temp_df["涨跌额"] = pd.to_numeric(temp_df["涨跌额"], errors="coerce")
    temp_df["成交量"] = pd.to_numeric(temp_df["成交量"], errors="coerce")
    temp_df["成交额"] = pd.to_numeric(temp_df["成交额"], errors="coerce")
    temp_df["振幅"] = pd.to_numeric(temp_df["振幅"], errors="coerce")
    temp_df["换手率"] = pd.to_numeric(temp_df["换手率"], errors="coerce")
    return temp_df


def dfcf_index_data_BK1158():
    # 获取微盘股指数 东财
    stock_board_industry_spot_em_df = stock_board_industry_hist_em(symbol="BK1158")

    stock_board_industry_spot_em_df.rename(columns={"日期": "date",
                                                    "开盘": "open",
                                                    "收盘": "close",
                                                    "最高": "high",
                                                    "最低": "low",
                                                    "成交量": "volume",
                                                    "成交额": "amount",
                                                    "换手率": "turn",
                                                    "涨跌幅": "pctChg",
                                                    }, inplace=True )
    stock_board_industry_spot_em_df["factor"] = 1
    stock_board_industry_spot_em_df["is_ST"] = 0
    return stock_board_industry_spot_em_df[['date','open','close','high','low','volume','amount','turn','pctChg',"is_ST"]]

# open,high,low,close,volume,amount,turn,pctChg,factor
if __name__ == '__main__':

    '''
    # symbol="上证系列指数"；choice of {"沪深重要指数", "上证系列指数", "深证系列指数", "指数成份", "中证系列指数"}
    for name_zs in ["沪深重要指数", "上证系列指数", "深证系列指数", "指数成份", "中证系列指数"]:
        print(name_zs)
        stock_zh_index_spot_em_df = ak.stock_zh_index_spot_em(symbol=name_zs)
        print(stock_zh_index_spot_em_df)
        print('---'*20)
        print('\n')
        stock_zh_index_spot_em_df.to_csv(f"{name_zs}.csv", index=False)
    # stock_zh_index_spot_em_df = ak.stock_zh_index_spot_em(symbol="上证系列指数")
    # print(stock_zh_index_spot_em_df)
    '''
    bast_path = '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/index'
    BK1158_df = dfcf_index_data_BK1158()
    BK1158_df.to_csv(f"{bast_path}/BK1158.csv", index=False)

    code_path = f"/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/market/BK1158"
    if not os.path.exists(code_path):
        os.mkdir(code_path)
    BK1158_df.to_csv(f"{code_path}/daily.csv", index=False)
    BK1158_df.to_csv(f"{code_path}/daily_a.csv", index=False)

    # 000开头是上证、

    # 000开头是上证、
    zs_list = ["沪深重要指数",  "指数成份", ]

    if not os.path.exists(bast_path):
        os.makedirs(bast_path)

    for i, name_zs in enumerate(zs_list):
        stock_zh_index_spot_em_df = ak.stock_zh_index_spot_em(symbol=name_zs)
        print(stock_zh_index_spot_em_df['代码'].tolist())
        code_list = stock_zh_index_spot_em_df['代码'].tolist()

        for j, code in enumerate(code_list):
            query_code =  code
            query_code1 =  code
            if code.startswith("000"):
                query_code = 'sh' + query_code
                query_code1 = 'sh.' + query_code1
            elif code.startswith("399"):
                query_code = 'sz' + query_code
                query_code1 = 'sz.' + query_code1
            elif code.startswith("93"):
                query_code = 'csi' + query_code
                query_code1 = 'csi' + query_code1
            else:
                continue

            time.sleep(random.randint(1,3))

            print(f'important_indices_sh_sz execute {i}-{j}:', query_code)
            stock_zh_index_daily_em_df = ak.stock_zh_index_daily_em(symbol=query_code)
            # print(stock_zh_index_daily_em_df)
            stock_zh_index_daily_em_df.to_csv(f"{bast_path}/{query_code}.csv", index=False)

            code_path = f"/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/market/{query_code1}"
            if not os.path.exists(code_path):
                os.mkdir(code_path)

            if not os.path.exists(f"{code_path}/daily.csv"):
                stock_zh_index_daily_em_df.to_csv(f"{code_path}/daily.csv", index=False)
                stock_zh_index_daily_em_df.to_csv(f"{code_path}/daily_a.csv", index=False)


    # 中证 2000 指数
    # symbol: 带市场标识的指数代码;
    # sz: 深交所, 399
    # sh: 上交所, 000开头
    # csi: 中信指数 + id(000905)  93/
    # stock_zh_index_daily_em_df = ak.stock_zh_index_daily_em(symbol="csi932000")
    # print(stock_zh_index_daily_em_df)
    # stock_zh_index_daily_em_df.to_csv("中证2000-csi932000.csv", index=False)