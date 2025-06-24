import pandas as pd

import akshare as ak



if __name__ == '__main__':
    import akshare as ak

    # symbol="上证系列指数"；choice of {"沪深重要指数", "上证系列指数", "深证系列指数", "指数成份", "中证系列指数"}
    for name_zs in ["沪深重要指数", "上证系列指数", "深证系列指数", "指数成份", "中证系列指数"]:
        print(name_zs)
        stock_zh_index_spot_em_df = ak.stock_zh_index_spot_em(symbol=name_zs)
        print(stock_zh_index_spot_em_df)
        print('---'*20)
        print('\n')
    # stock_zh_index_spot_em_df = ak.stock_zh_index_spot_em(symbol="上证系列指数")
    # print(stock_zh_index_spot_em_df)

    # 000开头是上证、

    # 中证 2000 指数
    # symbol: 带市场标识的指数代码; sz: 深交所, sh: 上交所, csi: 中信指数 + id(000905)
    stock_zh_index_daily_em_df = ak.stock_zh_index_daily_em(symbol="csi932000")
    print(stock_zh_index_daily_em_df)