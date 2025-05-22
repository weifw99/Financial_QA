import mplfinance as mpf

import matplotlib.pyplot as plt

def show_k(df, save_path= 'apple_candlestick.png'):

    # 将数据转换为mplfinance所需的格式 open,high,low,close,volume
    data = df[['open', 'high', 'low', 'close', 'volume']]
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    print( data.index)

    start_date = pd.Timestamp('2024-01-01')  # 假设起始日期
    date_range = pd.date_range(start=start_date, periods=len(data), freq='D')
    data.index = date_range

    # data.index = pd.to_datetime(data.index)  # 转换索引为DatetimeIndex
    # data.index.name = 'Date'
    data.index.name = 'Date'
    data = data.astype(float)

    kwargs = dict(type='candle', volume=True, show_nontrading=True,
                  style='yahoo', title='K-day', ylabel='price')
    # mpf.plot(data, **kwargs, mav=(5, 10, 20))
    mpf.plot(data, **kwargs, mav=(10, 20, 40), savefig=save_path)

    # mpf.plot(data, type='candle', volume=True, show_nontrading=True)
    # plt.savefig('k_line_chart.png')


def show_k_2(df, result_df, save_path= 'apple_candlestick.png'):
    import pandas as pd

    merge_df = pd.concat([df, result_df], axis=0)

    # print(df.shape, result_df.shape, merge_df.shape)

    # 将数据转换为mplfinance所需的格式 open,high,low,close,volume
    data = merge_df[['open', 'high', 'low', 'close', 'volume']]
    data.columns = ['Open', 'High', 'Low', 'Close', 'Volume']
    # print( data.index)

    start_date = pd.Timestamp('2024-01-01')  # 假设起始日期
    date_range = pd.date_range(start=start_date, periods=len(data), freq='D')
    data.index = date_range

    # data.index = pd.to_datetime(data.index)  # 转换索引为DatetimeIndex
    # data.index.name = 'Date'
    data.index.name = 'Date'
    data = data.astype(float)

    kwargs = dict(type='candle', volume=True, show_nontrading=True,
                  style='yahoo', title='K-day', ylabel='price')
    # mpf.plot(data, **kwargs, mav=(5, 10, 20))
    mpf.plot(data, **kwargs, mav=(10, 20, 40), savefig=save_path)

    # mpf.plot(data, type='candle', volume=True, show_nontrading=True)
    # plt.savefig('k_line_chart.png')

    # import finplot as fplt
    # fplt.candlestick_ochl(data[['Open', 'Close', 'High', 'Low']])
    # fplt.show()

if __name__ == '__main__':
    import pandas as pd

    df = pd.read_csv('/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data_copy_test_predict1/sh.600642_c2guNjAwNjQy/2024-11-01_0_input.csv', sep=',')
    result_df = pd.read_csv('/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data_copy_test_predict1/sh.600642_c2guNjAwNjQy/2024-11-01_0_result.csv', sep=',')
    predict_df = pd.read_csv('/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data_copy_test_predict1/sh.600642_c2guNjAwNjQy/2024-11-01_0_predict.csv', sep=',')

    show_k(df)
    show_k_2(df, result_df)
    show_k_2(df, predict_df)