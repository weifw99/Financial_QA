import os
import random
import time

import pandas as pd

import akshare as ak
import requests

from src.data.zh_data.index.important_indices_sh_sz import dfcf_index_data_BK1158

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
    if not os.path.exists(bast_path):
        os.makedirs(bast_path)

    bast_path = '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/index'

    code_list: list[str] = ['932000', '399101', '399005', '399401', '399663', '399008', '000300', '000016','000046',
                            '000905', '000906', '000132','000133', '000010','000009', ]
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

        time.sleep(random.randint(2,5))

        print(f'important_indices_sh_sz execute - {j}:', query_code)
        stock_zh_index_daily_em_df = ak.stock_zh_index_daily_em(symbol=query_code)
        # print(stock_zh_index_daily_em_df)
        stock_zh_index_daily_em_df.to_csv(f"{bast_path}/{query_code}.csv", index=False)

        code_path = f"/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/market/{query_code1}"
        if not os.path.exists(code_path):
            os.mkdir(code_path)

        if len(stock_zh_index_daily_em_df)>0:
            # if not os.path.exists(f"{code_path}/daily.csv"):
            #     stock_zh_index_daily_em_df.to_csv(f"{code_path}/daily.csv", index=False)
            #     stock_zh_index_daily_em_df.to_csv(f"{code_path}/daily_a.csv", index=False)
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