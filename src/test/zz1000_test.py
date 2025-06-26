import pandas as pd

import akshare as ak



if __name__ == '__main__':

    # 获取中证1000（000852）成分股
    # df = ak.index_stock_cons(symbol="000852")
    # print(df.head())
    #
    # print( len(df) )

    index_stock_cons_csindex_df = ak.index_stock_cons_csindex(symbol="000852")
    print(index_stock_cons_csindex_df.head())
    print(len(index_stock_cons_csindex_df))

    index_stock_cons_csindex_df['type'] = index_stock_cons_csindex_df['交易所'].apply(
        lambda x: 'sz.' if x == '深圳证券交易所' else 'sh.') + index_stock_cons_csindex_df['成分券代码']

    index_stock_cons_csindex_df.to_csv("中证1000-000852.csv", index=False)

    index_stock_cons_csindex_df = ak.index_stock_cons_csindex(symbol="932000")
    print(index_stock_cons_csindex_df.head())
    print(len(index_stock_cons_csindex_df))

    index_stock_cons_csindex_df['type'] = index_stock_cons_csindex_df['交易所'].apply(
        lambda x: 'sz.' if x == '深圳证券交易所' else 'sh.') + index_stock_cons_csindex_df['成分券代码']

    index_stock_cons_csindex_df.to_csv("中证2000-932000.csv", index=False)

    import akshare as ak

    # 获取中证1000（000852）成分股
    df = ak.index_stock_cons(symbol="000852")
    index_stock_cons_csindex_df = ak.index_stock_cons_csindex(symbol="000852")
    print(len(index_stock_cons_csindex_df))

    index_stock_cons_csindex_df['type'] = index_stock_cons_csindex_df['交易所'].apply(
        lambda x: 'SZ' if x == '深圳证券交易所' else 'SH') + index_stock_cons_csindex_df['成分券代码']

    index_stock_cons_csindex_df = index_stock_cons_csindex_df[['type', '成分券代码']]
    index_stock_cons_csindex_df.columns = ['type', 'code']
    '''
    品种代码  品种名称        纳入日期
    002093  国脉科技  2025-06-16
    '''
    df.columns = ['code', 'code_name', 'start_time']
    df['end_time'] = '2025-06-23'

    df = pd.merge(df, index_stock_cons_csindex_df, left_on='code', right_on='code', how='left')
    df1 = df[['type', 'start_time', 'end_time']]
    df1.columns = ['code', 'start_time', 'end_time']


    # index_detail_cni_df = ak.index_detail_cni(symbol='000852', date='202404')
    # print(index_detail_cni_df)
