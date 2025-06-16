import baostock as bs
import pandas as pd


def get_financial_data(stock_code):
    """
    获取指定股票某季度的财务数据：净利润、营业收入、是否ST
    :param stock_code: 股票代码，如'sh.600000'
    """
    lg = bs.login()
    if not lg.error_code == '0':
        print("登录失败:", lg.error_msg)
        return None
    import  datetime
    y = datetime.datetime.now().year

    quarter = [1, 2, 3, 4]

    data_list = []
    for year in [i for i in range(2000, y+1) ]:
        for q in quarter:
            rs = bs.query_profit_data(code=stock_code, year=year, quarter=q)
            if rs.error_code == '0':
                while rs.next():
                    row = rs.get_row_data()
                    data_list.append(row)
    df_profit = pd.DataFrame(data_list, columns=rs.fields).sort_values(['pubDate'])
    print(len(df_profit))
    print(df_profit.head())

    if not df_profit.empty:
        pass

    bs.logout()

    return df_profit

def test():
# if __name__ == "__main__":
    stock = 'sh.600000'  # 示例：浦发银行
    year = 2023
    quarter = 1
    date = '2023-06-30'

    fin_data = get_financial_data(stock)
    print(f"财务数据：{fin_data}")