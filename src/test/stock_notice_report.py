

import akshare as ak



if __name__ == '__main__':
    # stock_notice_report_df = ak.stock_notice_report(symbol='财务报告', date="20240613")
    # print(stock_notice_report_df)


    stock_financial_abstract_ths_df = ak.stock_financial_abstract_ths(symbol="600004", indicator="按年度")
    print(stock_financial_abstract_ths_df)

    import akshare as ak

    stock_financial_abstract_df = ak.stock_financial_abstract(symbol="600004")
    print(stock_financial_abstract_df)




