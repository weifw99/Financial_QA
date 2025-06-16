import baostock as bs
import pandas as pd


def get_financial_data(stock_code, year, quarter):
    """
    获取指定股票某季度的财务数据：净利润、营业收入、是否ST
    :param stock_code: 股票代码，如'sh.600000'
    :param year: 财报年份，如2023
    :param quarter: 季度 1-4
    :return: dict，包含净利润(profit), 营业收入(revenue), 是否ST(is_st)
    """
    lg = bs.login()
    if not lg.error_code == '0':
        print("登录失败:", lg.error_msg)
        return None

    # 查询财务指标（逐季度）
    rs = bs.query_profit_data(code=stock_code, year=year, quarter=quarter)
    profit = None
    MBRevenue = None
    totalShare = None
    if rs.error_code == '0':
        data_list = []
        while rs.next():
            row = rs.get_row_data()
            data_list.append(row)
        df_profit = pd.DataFrame(data_list, columns=rs.fields)
        if not df_profit.empty:
            profit = float(df_profit.loc[0, 'netProfit'])  # 净利润(元)
            MBRevenue = float(df_profit.loc[0, 'MBRevenue'])  # 营业收入(元)
            totalShare = float(df_profit.loc[0, 'totalShare'])  # 总股本
            # 总市值 = 股票价格 × 总股票数。例如，对于A股上市公司，市值可以通过A股每股股价乘以总股本来计算。
    else:
        print("查询利润数据失败", rs.error_msg)


    bs.logout()

    return {
        'profit': profit,
        'MBRevenue': MBRevenue,
        'totalShare': totalShare,
    }


def get_market_value(stock_code, date):
    """
    获取股票指定日期的总市值（单位：万元）
    :param stock_code: 股票代码
    :param date: 日期，格式'YYYY-MM-DD'
    :return: float 市值
    """
    lg = bs.login()
    if not lg.error_code == '0':
        print("登录失败:", lg.error_msg)
        return None

    bs.query_history_k_data_plus()

    rs = bs.query_history_k_data(stock_code,
                                 "date,code,close,total_mv",
                                 start_date=date, end_date=date,
                                 frequency="d", adjustflag='3')  # 前复权
    mv = None
    if rs.error_code == '0':
        data_list = []
        while rs.next():
            data_list.append(rs.get_row_data())
        df = pd.DataFrame(data_list, columns=rs.fields)
        if not df.empty:
            mv = float(df.loc[0, 'total_mv'])  # 总市值，单位万元
    else:
        print("查询市值失败", rs.error_msg)

    bs.logout()
    return mv


if __name__ == "__main__":
    stock = 'sh.600000'  # 示例：浦发银行
    year = 2023
    quarter = 1
    date = '2023-06-30'

    fin_data = get_financial_data(stock, year, quarter)
    print(f"财务数据：{fin_data}")

    mv = get_market_value(stock, date)
    print(f"{date} 总市值（万元）：{mv}")