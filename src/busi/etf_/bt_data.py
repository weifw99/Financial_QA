import backtrader as bt
import pandas as pd
from busi.etf_.etf_data import EtfDataHandle
from busi.etf_.util_moment import (
    momentum_linear,
    momentum_simple,
    log_momentum_simple,
    log_momentum_r2,
    line_log_momentum_r2,
    momentum_dual,
    momentum_dual_v2
)

import pandas as pd
import numpy as np
from scipy import stats


def calc_slope(y):
    """计算线性回归斜率"""
    x = np.arange(len(y))
    if len(y) < 5:
        return np.nan
    slope, _, _, _, _ = stats.linregress(x, y)
    return slope


def calc_r2(y):
    """计算R²"""
    x = np.arange(len(y))
    if len(y) < 5:
        return np.nan
    _, _, r_value, _, _ = stats.linregress(x, y)
    return r_value ** 2


def calc_log_return(y):
    """计算对数收益率"""
    if len(y) < 2:
        return np.nan
    return np.log(y.iloc[-1] / y.iloc[0])


def add_rolling_momentum_slope(df: pd.DataFrame, window: int = 365, price_col: str = "close") -> pd.DataFrame:
    """
    给定 DataFrame，为每一支ETF计算 rolling 动量 (斜率  )

    :param df: 输入数据，要求至少有 ['ts_code', 'trade_date', price_col] 三列
    :param window: 滚动窗口大小，默认365天
    :param price_col: 使用哪个价格列来计算，默认是 'close'
    :return: 添加新列 的DataFrame
    """
    df = df.copy()
    df = df.sort_values(['symbol', 'date'])  # 按股票代码和时间排序

    slope_col = f"slope_{window}d"
    df[slope_col] = (
        df.groupby('symbol')[price_col]
        .rolling(window=window, min_periods=window)
        .apply(calc_slope, raw=True)
        .reset_index(level=0, drop=True)
    )

    r2_col = f"r2_{window}d"
    df[r2_col] = (
        df.groupby('symbol')[price_col]
        .rolling(window=window, min_periods=window)
        .apply(calc_r2, raw=True)
        .reset_index(level=0, drop=True)
    )

    # # 对数收益率
    #     log_return = np.log(closes[-1] / closes[-window])
    # 计算N 天的涨跌幅
    pivot_col = f"pivot_{window}d"
    df[pivot_col] = (
        df.groupby('symbol')[price_col]
        .pct_change(periods=window)
        .reset_index(level=0, drop=True)
    )

    pivot_log_col = f"pivot_log_{window}d"
    df[pivot_log_col] = (
        df.groupby('symbol')[price_col]
        .rolling(window=window, min_periods=window)
        .apply(calc_log_return, raw=True)
        .reset_index(level=0, drop=True)
    )

    return df


def momentum_linear_pandas(df, window=20):
    """线性动量"""
    result = df.groupby('symbol')['close'].rolling(window=window).apply(calc_slope)
    return result.reset_index(level=0, drop=True)

def momentum_simple_pandas(df, window=20):
    """简单动量"""
    result = df.groupby('symbol')['close'].pct_change(periods=window)
    return result

def log_momentum_simple_pandas(df, window=20):
    """对数简单动量"""
    result = df.groupby('symbol')['close'].rolling(window=window).apply(calc_log_return)
    return result.reset_index(level=0, drop=True)

def log_momentum_r2_pandas(df, window=20):
    """对数R2动量"""
    log_returns = df.groupby('symbol')['close'].rolling(window=window).apply(calc_log_return)
    r2 = df.groupby('symbol')['close'].rolling(window=window).apply(calc_r2)
    result = log_returns * r2
    return result.reset_index(level=0, drop=True)

def line_log_momentum_r2_pandas(df, window=20):
    """线性对数R2动量"""
    slope = df.groupby('symbol')['close'].rolling(window=window).apply(calc_slope)
    log_returns = df.groupby('symbol')['close'].rolling(window=window).apply(calc_log_return)
    r2 = df.groupby('symbol')['close'].rolling(window=window).apply(calc_r2)
    result = slope * log_returns * r2
    return result.reset_index(level=0, drop=True)

def momentum_dual_pandas(df, long_window=90, short_window=20, smooth_long=20, smooth_short=5,
                        min_long_return=0.02, min_short_return=0.01, long_weight=0.7, short_weight=0.3):
    """双动量"""
    # 计算平滑价格
    smooth_price = df.groupby('symbol')['close'].rolling(window=smooth_long).mean()
    smooth_price = smooth_price.reset_index(level=0, drop=True)
    
    # 计算长期动量
    long_slope = df.groupby('symbol')['close'].rolling(window=long_window).apply(calc_slope)
    long_slope = long_slope.reset_index(level=0, drop=True)
    long_r2 = df.groupby('symbol')['close'].rolling(window=long_window).apply(calc_r2)
    long_r2 = long_r2.reset_index(level=0, drop=True)
    long_return = df.groupby('symbol')['close'].rolling(window=long_window).apply(calc_log_return)
    long_return = long_return.reset_index(level=0, drop=True)
    long_score = long_slope * long_r2
    long_score[long_return < min_long_return] = np.nan
    
    # 计算短期动量
    short_slope = df.groupby('symbol')['close'].rolling(window=short_window).apply(calc_slope)
    short_slope = short_slope.reset_index(level=0, drop=True)
    short_r2 = df.groupby('symbol')['close'].rolling(window=short_window).apply(calc_r2)
    short_r2 = short_r2.reset_index(level=0, drop=True)
    short_return = df.groupby('symbol')['close'].rolling(window=short_window).apply(calc_log_return)
    short_return = short_return.reset_index(level=0, drop=True)
    short_score = short_slope * short_r2
    short_score[short_return < min_short_return] = np.nan
    
    # 组合动量
    result = long_score * long_weight + short_score * short_weight
    return result

def momentum_dual_v2_pandas(df, long_window=90, short_window=20, smooth_long=20, smooth_short=5,
                          min_long_return=0.02, min_short_return=0.01, slope_positive_filter=True,
                          weight_long=0.7, weight_short=0.3):
    """双动量V2"""
    # 计算平滑价格
    smooth_price = df.groupby('symbol')['close'].rolling(window=smooth_long).mean()
    smooth_price = smooth_price.reset_index(level=0, drop=True)
    
    # 计算长期动量
    long_slope = df.groupby('symbol')['close'].rolling(window=long_window).apply(calc_slope)
    long_slope = long_slope.reset_index(level=0, drop=True)
    long_r2 = df.groupby('symbol')['close'].rolling(window=long_window).apply(calc_r2)
    long_r2 = long_r2.reset_index(level=0, drop=True)
    long_return = df.groupby('symbol')['close'].rolling(window=long_window).apply(calc_log_return)
    long_return = long_return.reset_index(level=0, drop=True)
    long_score = long_slope * long_r2
    long_score[long_return < min_long_return] = np.nan
    
    # 计算短期动量
    short_slope = df.groupby('symbol')['close'].rolling(window=short_window).apply(calc_slope)
    short_slope = short_slope.reset_index(level=0, drop=True)
    short_r2 = df.groupby('symbol')['close'].rolling(window=short_window).apply(calc_r2)
    short_r2 = short_r2.reset_index(level=0, drop=True)
    short_return = df.groupby('symbol')['close'].rolling(window=short_window).apply(calc_log_return)
    short_return = short_return.reset_index(level=0, drop=True)
    short_score = short_slope * short_r2
    short_score[short_return < min_short_return] = np.nan
    
    # 斜率过滤
    if slope_positive_filter:
        long_score[long_slope < 0] = np.nan
        short_score[short_slope < 0] = np.nan
    
    # 组合动量
    result = long_score * weight_long + short_score * weight_short
    return result


#获取数据
class Getdata():

    def __init__(self, symbols: list[str]):
        self.data_fetcher = EtfDataHandle()
        self.symbols = symbols

    def dailydata(self):
        """获取日线数据"""
        data = self.dailydata_no_index()
        # 设置多级索引
        data.set_index(['date', ], inplace=True)
        # 选择需要的列
        data = data[['symbol', 'open', 'high', 'low', 'close', 'volume', 'openinterest']]
        return data

    def dailydata_no_index(self):
        """获取日线数据"""
        # 获取所有数据
        if self.symbols is None:
            data = EtfDataHandle().get_down_all_data()
        else:
            data = EtfDataHandle().get_local_symbols_data(self.symbols)

        if data is None or data.empty:
            print("警告：从EtfDataHandle获取数据为空")
            return None

        print(f"原始数据基本信息：\n{data.info()}")
        print(f"原始数据前5行：\n{data.head()}")

        # 数据选择
        rename_cols = {
            '代码': 'symbol',
            # '日期': 'date',
            '开盘': 'open',
            '收盘': 'close',
            '最高': 'high',
            '最低': 'low',
            '成交量': 'volume'
        }
        data.rename(columns=rename_cols, inplace=True)
        data['openinterest'] = 0

        # 检查数据完整性
        print(f"数据时间范围：{data['date'].min()} 到 {data['date'].max()}")
        print(f"ETF数量：{data['symbol'].nunique()}")
        print(f"价格数据统计：\n{data[['open', 'high', 'low', 'close']].describe()}")

        # 检查是否有零值或空值
        zero_prices = data[data['close'] == 0]
        if not zero_prices.empty:
            print(f"警告：发现{len(zero_prices)}条收盘价为0的记录")
            print(f"零价格记录：\n{zero_prices[['symbol', 'date', 'close']]}")

        data['symbol'] = data['symbol'].apply(lambda x: x.replace('SH', '').replace('SZ', '').replace('WZ', ''))
        # 按日期和股票代码排序
        data = data.sort_values(['date', 'symbol'])

        # 将日期转换为datetime类型
        data['date'] = pd.to_datetime(data['date'])

        # 选择需要的列
        data = data[['date', 'symbol', 'open', 'high', 'low', 'close', 'volume', 'openinterest']]

        return data

    def dailydata1(self, momentum_params=None):
        """获取日线数据并计算动量指标"""
        if momentum_params is None:
            momentum_params = {}
            
        # 获取基础数据
        df = self.dailydata()
        if df is None or df.empty:
            print("警告：获取数据为空")
            return None
            
        print(f"数据基本信息：\n{df.info()}")
        print(f"数据前5行：\n{df.head()}")
        
        # 检查数据完整性
        print(f"数据时间范围：{df.index.get_level_values('date').min()} 到 {df.index.get_level_values('date').max()}")
        print(f"ETF数量：{df.index.get_level_values('symbol').nunique()}")
        
        # 检查并处理重复索引
        if df.index.duplicated().any():
            print(f"警告：发现{df.index.duplicated().sum()}个重复索引")
            # 按symbol分组，对每个组内的重复索引进行处理
            df = df.groupby(level='symbol').apply(lambda x: x[~x.index.duplicated(keep='first')])
            df = df.reset_index(level=0, drop=True)
            print(f"处理后的数据基本信息：\n{df.info()}")
        
        # 计算动量指标
        if 'linear_window' in momentum_params:
            window = momentum_params['linear_window']
            print(f"计算线性动量，窗口大小：{window}")
            # 按symbol分组计算动量
            df['momentum_linear'] = df.groupby(level='symbol')['close'].apply(
                lambda x: (x - x.shift(window)) / x.shift(window)
            ).reset_index(level=0, drop=True)
            print(f"线性动量计算结果：\n{df['momentum_linear'].describe()}")
            
        if 'simple_window' in momentum_params:
            window = momentum_params['simple_window']
            print(f"计算简单动量，窗口大小：{window}")
            df['momentum_simple'] = df.groupby(level='symbol')['close'].pct_change(periods=window)
            print(f"简单动量计算结果：\n{df['momentum_simple'].describe()}")
            
        if 'log_simple_window' in momentum_params:
            window = momentum_params['log_simple_window']
            print(f"计算对数简单动量，窗口大小：{window}")
            df['momentum_log_simple'] = df.groupby(level='symbol')['close'].apply(
                lambda x: np.log(x / x.shift(window))
            ).reset_index(level=0, drop=True)
            print(f"对数简单动量计算结果：\n{df['momentum_log_simple'].describe()}")
            
        if 'log_r2_window' in momentum_params:
            window = momentum_params['log_r2_window']
            print(f"计算对数R2动量，窗口大小：{window}")
            log_returns = df.groupby(level='symbol')['close'].apply(
                lambda x: np.log(x / x.shift(window))
            ).reset_index(level=0, drop=True)
            r2 = df.groupby(level='symbol')['close'].rolling(window=window).apply(
                lambda x: calc_r2(x) if len(x) >= window else np.nan
            ).reset_index(level=0, drop=True)
            df['momentum_log_r2'] = log_returns * r2
            print(f"对数R2动量计算结果：\n{df['momentum_log_r2'].describe()}")
            
        if 'line_log_r2_window' in momentum_params:
            window = momentum_params['line_log_r2_window']
            print(f"计算线性对数R2动量，窗口大小：{window}")
            slope = df.groupby(level='symbol')['close'].rolling(window=window).apply(
                lambda x: calc_slope(x) if len(x) >= window else np.nan
            ).reset_index(level=0, drop=True)
            log_returns = df.groupby(level='symbol')['close'].apply(
                lambda x: np.log(x / x.shift(window))
            ).reset_index(level=0, drop=True)
            r2 = df.groupby(level='symbol')['close'].rolling(window=window).apply(
                lambda x: calc_r2(x) if len(x) >= window else np.nan
            ).reset_index(level=0, drop=True)
            df['momentum_line_log_r2'] = slope * log_returns * r2
            print(f"线性对数R2动量计算结果：\n{df['momentum_line_log_r2'].describe()}")
            
        if all(k in momentum_params for k in ['long_window', 'short_window']):
            print("计算双动量")
            long_window = momentum_params['long_window']
            short_window = momentum_params['short_window']
            smooth_long = momentum_params.get('smooth_long', 20)
            smooth_short = momentum_params.get('smooth_short', 5)
            min_long_return = momentum_params.get('min_long_return', 0.02)
            min_short_return = momentum_params.get('min_short_return', 0.01)
            long_weight = momentum_params.get('long_weight', 0.7)
            short_weight = momentum_params.get('short_weight', 0.3)
            
            # 计算平滑价格
            smooth_price = df.groupby(level='symbol')['close'].rolling(window=smooth_long).mean()
            smooth_price = smooth_price.reset_index(level=0, drop=True)
            
            # 计算长期动量
            long_slope = df.groupby(level='symbol')['close'].rolling(window=long_window).apply(
                lambda x: calc_slope(x) if len(x) >= long_window else np.nan
            ).reset_index(level=0, drop=True)
            long_r2 = df.groupby(level='symbol')['close'].rolling(window=long_window).apply(
                lambda x: calc_r2(x) if len(x) >= long_window else np.nan
            ).reset_index(level=0, drop=True)
            long_return = df.groupby(level='symbol')['close'].apply(
                lambda x: np.log(x / x.shift(long_window))
            ).reset_index(level=0, drop=True)
            long_score = long_slope * long_r2
            long_score[long_return < min_long_return] = np.nan
            
            # 计算短期动量
            short_slope = df.groupby(level='symbol')['close'].rolling(window=short_window).apply(
                lambda x: calc_slope(x) if len(x) >= short_window else np.nan
            ).reset_index(level=0, drop=True)
            short_r2 = df.groupby(level='symbol')['close'].rolling(window=short_window).apply(
                lambda x: calc_r2(x) if len(x) >= short_window else np.nan
            ).reset_index(level=0, drop=True)
            short_return = df.groupby(level='symbol')['close'].apply(
                lambda x: np.log(x / x.shift(short_window))
            ).reset_index(level=0, drop=True)
            short_score = short_slope * short_r2
            short_score[short_return < min_short_return] = np.nan
            
            # 组合动量
            df['momentum_dual'] = long_score * long_weight + short_score * short_weight
            print(f"双动量计算结果：\n{df['momentum_dual'].describe()}")
            
            # 计算双动量V2
            if 'slope_positive_filter' in momentum_params:
                slope_positive_filter = momentum_params['slope_positive_filter']
                weight_long = momentum_params.get('weight_long', 0.7)
                weight_short = momentum_params.get('weight_short', 0.3)
                
                if slope_positive_filter:
                    long_score[long_slope < 0] = np.nan
                    short_score[short_slope < 0] = np.nan
                
                df['momentum_dual_v2'] = long_score * weight_long + short_score * weight_short
                print(f"双动量V2计算结果：\n{df['momentum_dual_v2'].describe()}")
        
        # 选择需要的列
        cols = ['open', 'high', 'low', 'close', 'volume', 'openinterest']
        momentum_cols = [col for col in df.columns if col.startswith('momentum_')]
        df = df[cols + momentum_cols]
        
        return df


#拓展数据
class Dailydataextend(bt.feeds.PandasData):
    # 增加动量指标线
    lines = (
        'momentum_linear', 'momentum_simple', 'momentum_log_simple',
        'momentum_log_r2', 'momentum_line_log_r2', 'momentum_dual', 'momentum_dual_v2'
    )
    
    # 定义参数
    params = (
        ('momentum_linear', -1),
        ('momentum_simple', -1),
        ('momentum_log_simple', -1),
        ('momentum_log_r2', -1),
        ('momentum_line_log_r2', -1),
        ('momentum_dual', -1),
        ('momentum_dual_v2', -1),
        ('dtformat', '%Y-%m-%d'),
    )
    
    def __init__(self, **kwargs):
        # 确保数据格式正确
        if 'dataname' in kwargs:
            df = kwargs['dataname']
            if isinstance(df.index, pd.MultiIndex):
                # 重置索引，将日期和symbol作为列
                df = df.reset_index()
                # 设置日期为索引
                df.set_index('date', inplace=True)
                kwargs['dataname'] = df
        
        super().__init__(**kwargs)
        
        # 验证数据
        if self.p.dataname is None or self.p.dataname.empty:
            print("警告：输入数据为空")
            return
            
        print(f"数据源基本信息：\n{self.p.dataname.info()}")
        print(f"数据源前5行：\n{self.p.dataname.head()}")
        
        # 检查价格数据
        price_cols = ['open', 'high', 'low', 'close']
        for col in price_cols:
            if col not in self.p.dataname.columns:
                print(f"警告：缺少价格列 {col}")
            else:
                zero_prices = self.p.dataname[self.p.dataname[col] == 0]
                if not zero_prices.empty:
                    print(f"警告：{col}列中有{len(zero_prices)}个零值")
                    print(f"零值记录：\n{zero_prices[['symbol', col]].head()}")

