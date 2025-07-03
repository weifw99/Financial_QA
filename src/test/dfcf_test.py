import akshare as ak
import pandas as pd
import requests


def stock_board_industry_hist_em(
    symbol: str = "小金属",
    start_date: str = "20211201",
    end_date: str = "20220401",
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



if __name__ == '__main__':
    # stock_board_industry_name_em_df = ak.stock_board_industry_name_em()
    # print(stock_board_industry_name_em_df)
    #
    # stock_board_industry_spot_em_df = ak.stock_board_industry_spot_em(symbol="小金属")
    # print(stock_board_industry_spot_em_df)
    #
    #
    # # 获取微盘股指数 东财
    # stock_board_industry_spot_em_df = stock_board_industry_hist_em(symbol="BK1158")
    # print(stock_board_industry_spot_em_df)
    #
    # stock_board_industry_cons_em_df = ak.stock_board_industry_cons_em(symbol="小金属")
    # print(stock_board_industry_cons_em_df)

    # 获取 BK1158 的成分股 东财
    stock_board_industry_cons_em_df = ak.stock_board_industry_cons_em(symbol="BK1158")
    print(stock_board_industry_cons_em_df)