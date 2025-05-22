"""因子计算模块

主要功能：
1. 技术指标计算
2. 量价因子分析
3. 基本面因子
4. 宏观因子
"""

from typing import List, Dict, Optional, Union
from datetime import datetime

import numpy as np
import pandas as pd
from scipy import stats
from sklearn.preprocessing import StandardScaler

class TechnicalFactors:
    @staticmethod
    def calculate_ma(prices: pd.Series, windows: List[int]) -> pd.DataFrame:
        """计算移动平均线
        
        Args:
            prices: 价格序列
            windows: 移动窗口列表，如[5,10,20,60]
            
        Returns:
            DataFrame包含不同周期的移动平均线
        """
        ma_dict = {}
        for window in windows:
            ma_dict[f'MA{window}'] = prices.rolling(window).mean()
        return pd.DataFrame(ma_dict)
    
    @staticmethod
    def calculate_macd(
        prices: pd.Series,
        fast_period: int = 12,
        slow_period: int = 26,
        signal_period: int = 9
    ) -> pd.DataFrame:
        """计算MACD指标
        
        Args:
            prices: 价格序列
            fast_period: 快线周期
            slow_period: 慢线周期
            signal_period: 信号线周期
            
        Returns:
            DataFrame包含MACD指标
        """
        exp1 = prices.ewm(span=fast_period, adjust=False).mean()
        exp2 = prices.ewm(span=slow_period, adjust=False).mean()
        macd = exp1 - exp2
        signal = macd.ewm(span=signal_period, adjust=False).mean()
        hist = macd - signal
        
        return pd.DataFrame({
            'MACD': macd,
            'Signal': signal,
            'Histogram': hist
        })
        
    @staticmethod
    def calculate_rsi(
        prices: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """计算RSI指标
        
        Args:
            prices: 价格序列
            period: RSI周期
            
        Returns:
            RSI指标序列
        """
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

class PriceVolumeFactors:
    @staticmethod
    def calculate_vwap(
        prices: pd.Series,
        volumes: pd.Series,
        window: int = None
    ) -> pd.Series:
        """计算成交量加权平均价格(VWAP)
        
        Args:
            prices: 价格序列
            volumes: 成交量序列
            window: 计算窗口，None表示全周期
            
        Returns:
            VWAP序列
        """
        if window is None:
            return (prices * volumes).sum() / volumes.sum()
        return (prices * volumes).rolling(window).sum() / volumes.rolling(window).sum()
    
    @staticmethod
    def calculate_money_flow(
        high: pd.Series,
        low: pd.Series,
        close: pd.Series,
        volume: pd.Series,
        period: int = 14
    ) -> pd.Series:
        """计算资金流向指标(MFI)
        
        Args:
            high: 最高价序列
            low: 最低价序列
            close: 收盘价序列
            volume: 成交量序列
            period: 计算周期
            
        Returns:
            MFI指标序列
        """
        typical_price = (high + low + close) / 3
        money_flow = typical_price * volume
        
        delta = typical_price.diff()
        positive_flow = (money_flow.where(delta > 0, 0)).rolling(period).sum()
        negative_flow = (money_flow.where(delta < 0, 0)).rolling(period).sum()
        
        mfi = 100 - (100 / (1 + positive_flow / negative_flow))
        return mfi

class FundamentalFactors:
    @staticmethod
    def calculate_pe_percentile(
        pe_ratio: pd.Series,
        window: int = 252
    ) -> pd.Series:
        """计算PE百分位数
        
        Args:
            pe_ratio: 市盈率序列
            window: 回溯窗口
            
        Returns:
            PE百分位数序列
        """
        return pe_ratio.rolling(window).apply(
            lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1])
        )
    
    @staticmethod
    def calculate_pb_percentile(
        pb_ratio: pd.Series,
        window: int = 252
    ) -> pd.Series:
        """计算PB百分位数
        
        Args:
            pb_ratio: 市净率序列
            window: 回溯窗口
            
        Returns:
            PB百分位数序列
        """
        return pb_ratio.rolling(window).apply(
            lambda x: stats.percentileofscore(x.dropna(), x.iloc[-1])
        )

class MacroFactors:
    @staticmethod
    def calculate_industry_momentum(
        industry_returns: pd.DataFrame,
        window: int = 20
    ) -> pd.DataFrame:
        """计算行业动量因子
        
        Args:
            industry_returns: 行业收益率DataFrame
            window: 动量窗口
            
        Returns:
            行业动量因子DataFrame
        """
        return industry_returns.rolling(window).mean()
    
    @staticmethod
    def calculate_market_sentiment(
        index_close: pd.Series,
        index_volume: pd.Series,
        window: int = 20
    ) -> pd.Series:
        """计算市场情绪指标
        
        Args:
            index_close: 指数收盘价序列
            index_volume: 指数成交量序列
            window: 计算窗口
            
        Returns:
            市场情绪指标序列
        """
        price_momentum = index_close.pct_change(window)
        volume_momentum = index_volume.pct_change(window)
        return (price_momentum + volume_momentum) / 2