"""港股市场数据获取模块

主要功能：
1. 港股K线数据（日线、分钟线）
2. 实时行情
3. 基本面数据
"""

import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union

import pandas as pd
import yfinance as yf

class MarketDataAPI:
    def __init__(self):
        """初始化市场数据API"""
        self._stock_list_cache = None
        self._cache_time = None
        self._cache_duration = timedelta(days=1)  # 缓存更新周期为1天
        self._max_retries = 3  # 最大重试次数
        self._retry_delay = 0.1  # 重试间隔（秒）

    def _get_api_delay(self):
        """获取API调用间隔时间
        
        Returns:
            float: 随机延迟时间（秒）
        """
        return random.uniform(0.1, 0.2)
        
    def get_stock_list(self) -> pd.DataFrame:
        """获取港股股票列表
        
        Returns:
            DataFrame包含以下字段：
            - code: 股票代码
            - code_name: 股票名称
            - industry: 所属行业
            - type: 证券类型
            - status: 上市状态
        """
        # 检查缓存是否有效
        now = datetime.now()
        if (self._stock_list_cache is not None and 
            self._cache_time is not None and
            now - self._cache_time < self._cache_duration):
            return self._stock_list_cache
            
        try:
            print("[INFO] 开始获取港股列表")
            
            # 从Excel文件读取港股列表
            from pathlib import Path
            from . import RAW_DATA_DIR
            excel_file = RAW_DATA_DIR / 'ListOfSecurities.xlsx'
            
            if not excel_file.exists():
                print(f"[ERROR] 文件不存在: {excel_file}")
                return pd.DataFrame()
            
            # 读取Excel文件，跳过前两行
            df = pd.read_excel(excel_file, skiprows=2, dtype={
                'Stock Code': str,
                'Name of Securities': str,
                'Category': str,
                'Sub-Category': str,
                'Board Lot': str
            })
            
            # 确保必要的列存在
            required_columns = ['Stock Code', 'Name of Securities', 'Category', 'Sub-Category', 'Board Lot']
            missing_columns = [col for col in required_columns if col not in df.columns]
            if missing_columns:
                print(f"[ERROR] Excel文件缺少必要的列: {missing_columns}")
                return pd.DataFrame()
            
            # 重命名列
            df = df.rename(columns={
                'Stock Code': 'code',
                'Name of Securities': 'code_name',
                'Category': 'industry',
                'Sub-Category': 'type',
                'Board Lot': 'status'
            })
            
            # 处理股票代码
            df['code'] = df['code'].astype(str)
            # 移除可能存在的$符号
            df['code'] = df['code'].str.replace('$', '')
            # 添加.HK后缀
            df['code'] = df['code'] + '.HK'
            
            # 验证股票是否有效
            valid_stocks = []
            for _, row in df.iterrows():
                try:
                    ticker = yf.Ticker(row['code'])
                    info = ticker.info
                    print(f"[INFO] 股票{row['code']}, {info}")
                    if info and 'regularMarketPrice' in info:
                        valid_stocks.append(row)
                    else:
                        print(f"[WARN] 股票{row['code']}可能已退市或无效")
                except Exception as e:
                    print(f"[WARN] 验证股票{row['code']}时出错: {str(e)}")
                time.sleep(self._get_api_delay())
            
            df = pd.DataFrame(valid_stocks)
            
            # 只保留需要的列
            df = df[['code', 'code_name', 'industry', 'type', 'status']]
            
            print(f"[INFO] 港股列表数据数目：{len(df)}")
            print(df.head())
            
            # 保存到CSV文件
            stock_list_file = RAW_DATA_DIR / 'hk_stock_list.csv'
            df.to_csv(stock_list_file, index=False, encoding='utf-8')
            print(f"[INFO] 港股列表数据已保存到: {stock_list_file}")
            
            # 更新缓存
            self._stock_list_cache = df
            self._cache_time = now
                
            return df
            
        except Exception as e:
            print(f"[ERROR] 获取港股列表失败: {str(e)}")
            return pd.DataFrame()
        
    async def get_stock_daily(
        self,
        code: str,
        start_date: str,
        end_date: Optional[str] = None,
        frequency: str = 'd'
    ) -> pd.DataFrame:
        """获取港股K线数据
        
        Args:
            code: 股票代码（如：0700.HK）
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD），默认为当前日期
            frequency: 数据周期，默认为'd'（日K线）
                      - 'd': 日K线
                      - 'w': 周K线
                      - 'm': 月K线
                      - '1h': 1小时线
                      - '30m': 30分钟线
                      - '15m': 15分钟线
                      - '5m': 5分钟线
            
        Returns:
            DataFrame包含以下字段：
            - date: 交易日期/时间
            - open: 开盘价
            - high: 最高价
            - low: 最低价
            - close: 收盘价
            - volume: 成交量
            - amount: 成交额
        """
        # 验证frequency参数
        valid_frequencies = {'d', 'w', 'm', '1h', '30m', '15m', '5m'}
        if frequency.lower() not in valid_frequencies:
            print(f"无效的数据周期: {frequency}，使用默认值'd'")
            frequency = 'd'
        
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        # 处理股票代码中的$符号
        code = code.replace('$', '')
            
        for retry in range(self._max_retries):
            try:
                # 使用yfinance获取K线数据
                ticker = yf.Ticker(code)
                
                # 验证股票是否有效
                info = ticker.info
                if not info or 'regularMarketPrice' not in info:
                    print(f"[WARN] 股票{code}可能已退市或无效")
                    return pd.DataFrame()
                
                # 根据frequency选择interval参数
                interval_map = {
                    'd': '1d',
                    'w': '1wk',
                    'm': '1mo',
                    '1h': '1h',
                    '30m': '30m',
                    '15m': '15m',
                    '5m': '5m'
                }
                
                # 对于分钟级数据，限制时间范围
                if frequency in ['5m', '15m', '30m', '1h']:
                    # 计算最近60天的日期
                    end = datetime.now()
                    start = end - timedelta(days=60)
                    start_date = start.strftime('%Y-%m-%d')
                    end_date = end.strftime('%Y-%m-%d')
                    print(f"[INFO] 分钟级数据限制在最近60天内: {start_date} -> {end_date}")
                
                try:
                    df = ticker.history(
                        start=start_date,
                        end=end_date,
                        interval=interval_map[frequency.lower()]
                    )
                except Exception as e:
                    if "possibly delisted" in str(e):
                        print(f"[WARN] 股票{code}可能已退市")
                        return pd.DataFrame()
                    raise e
                
                if df.empty:
                    print(f"[WARN] 股票{code}没有数据")
                    return pd.DataFrame()
                
                # 重置索引，将日期变为列
                df = df.reset_index()
                
                # 重命名列
                column_map = {
                    'Date': 'date',
                    'Open': 'open',
                    'High': 'high',
                    'Low': 'low',
                    'Close': 'close',
                    'Volume': 'volume',
                    'Dividends': 'dividends',
                    'Stock Splits': 'stock_splits'
                }
                df = df.rename(columns=column_map)
                
                # 只保留需要的列
                required_columns = ['date', 'open', 'high', 'low', 'close', 'volume']
                df = df[required_columns]
                
                # 格式化日期列
                df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                
                time.sleep(self._get_api_delay())  # API调用后休眠
                
                return df
                
            except Exception as e:
                if retry < self._max_retries - 1:
                    print(f"[WARN] 第{retry + 1}次尝试失败: {str(e)}")
                    print(f"[INFO] 将在{self._retry_delay}秒后进行第{retry + 2}次重试")
                    time.sleep(self._retry_delay)
                    continue
                print(f"[ERROR] 所有重试均失败，最后一次错误: {str(e)}")
                return pd.DataFrame() 