import os
import random
import time

import pandas as pd

import akshare as ak



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

    # 000开头是上证、

    # 000开头是上证、
    zs_list = ["沪深重要指数",  "指数成份", ]

    bast_path = '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/index'
    if not os.path.exists(bast_path):
        os.makedirs(bast_path)

    for i, name_zs in enumerate(zs_list):
        stock_zh_index_spot_em_df = ak.stock_zh_index_spot_em(symbol=name_zs)
        print(stock_zh_index_spot_em_df['代码'].tolist())
        code_list = stock_zh_index_spot_em_df['代码'].tolist()

        for j, code in enumerate(code_list):
            query_code =  code
            if code.startswith("000"):
                query_code = 'sh' + query_code
            elif code.startswith("399"):
                query_code = 'sz' + query_code
            elif code.startswith("93"):
                query_code = 'csi' + query_code
            else:
                continue

            time.sleep(random.randint(1,3))

            print(f'important_indices_sh_sz execute {i}-{j}:', query_code)
            stock_zh_index_daily_em_df = ak.stock_zh_index_daily_em(symbol=query_code)
            # print(stock_zh_index_daily_em_df)
            stock_zh_index_daily_em_df.to_csv(f"{bast_path}/{query_code}.csv", index=False)


    # 中证 2000 指数
    # symbol: 带市场标识的指数代码;
    # sz: 深交所, 399
    # sh: 上交所, 000开头
    # csi: 中信指数 + id(000905)  93/
    # stock_zh_index_daily_em_df = ak.stock_zh_index_daily_em(symbol="csi932000")
    # print(stock_zh_index_daily_em_df)
    # stock_zh_index_daily_em_df.to_csv("中证2000-csi932000.csv", index=False)