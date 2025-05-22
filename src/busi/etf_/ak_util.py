import requests

import pandas as pd
from io import StringIO


def fund_etf_fund_daily_em() -> pd.DataFrame:
    """
    东方财富网-天天基金网-基金数据-场内交易基金
    https://fund.eastmoney.com/cnjy_dwjz.html
    :return: 当前交易日的所有场内交易基金数据
    :rtype: pandas.DataFrame
    """
    url = "https://fund.eastmoney.com/cnjy_dwjz.html"
    # r = requests.get(url, headers=headers)
    # r.encoding = "gb2312"

    path_ = 'data/fund_etf_fund_daily_em/cnjy_dwjz.html'

    with open(path_, "r", encoding="utf-8") as f:
        r_text = f.read()

    show_day = pd.read_html(StringIO(r_text))[1].iloc[0, 6:10].tolist()
    temp_df = pd.read_html(StringIO(r_text))[1].iloc[1:, 2:]
    temp_df_columns = temp_df.iloc[0, :].tolist()[1:]
    temp_df = temp_df.iloc[1:, 1:]
    temp_df.columns = temp_df_columns
    temp_df["基金简称"] = temp_df["基金简称"].str.strip("基金吧档案")
    temp_df.reset_index(inplace=True, drop=True)
    temp_df.columns = [
        "基金代码",
        "基金简称",
        "类型",
        f"{show_day[0]}-单位净值",
        f"{show_day[0]}-累计净值",
        f"{show_day[2]}-单位净值",
        f"{show_day[2]}-累计净值",
        "增长值",
        "增长率",
        "市价",
        "折价率",
    ]
    return temp_df


if __name__ == "__main__":
    result = fund_etf_fund_daily_em()

    print( len( result ) )
    print( result.head( 10 ) )