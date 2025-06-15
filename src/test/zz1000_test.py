import pandas as pd

import akshare as ak



if __name__ == '__main__':

    # 获取中证1000（000852）成分股
    df = ak.index_stock_cons(symbol="000852")
    print(df.head())

    print( len(df) )

    index_stock_cons_csindex_df = ak.index_stock_cons_csindex(symbol="000852")
    print(index_stock_cons_csindex_df.head())
    print(len(index_stock_cons_csindex_df))

    index_detail_cni_df = ak.index_detail_cni(symbol='399001', date='202404')
    print(index_detail_cni_df.head())
    print(len(index_detail_cni_df))

    # index_detail_cni_df = ak.index_detail_cni(symbol='000852', date='202404')
    # print(index_detail_cni_df)
