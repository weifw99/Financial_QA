"""市场数据获取模块

主要功能：
1. A股K线数据（日线、分钟线）
2. 实时行情（Level-1/Level-2）
3. 逐笔成交
4. 盘口数据
"""

import asyncio
import time
import random
from datetime import datetime, timedelta
from typing import List, Dict, Optional, Union

import pandas as pd
import baostock as bs
import aiohttp

from .connection import ConnectionManager

class MarketDataAPI:
    def __init__(self, token: str = None, connection_manager: ConnectionManager = None):
        """初始化市场数据API
        
        Args:
            token: API token (Baostock不需要token)
        """
        self._stock_list_cache = None
        self._cache_time = None
        self._cache_duration = timedelta(days=1)  # 缓存更新周期为1天
        self._max_retries = 3  # 最大重试次数
        self._retry_delay = 0.1  # 重试间隔（秒）
        self._conn_manager = ConnectionManager()
    def _get_api_delay(self):
        """获取API调用间隔时间
        
        Returns:
            float: 随机延迟时间（秒）
        """
        return random.uniform(0.1, 0.2)
        
    def get_stock_list(self) -> pd.DataFrame:
        """获取A股股票列表
        
        Returns:
            DataFrame包含以下字段：
            - code: 股票代码
            - code_name: 股票名称
            - industry: 所属行业
            - ipoDate: 上市日期
            - outDate: 退市日期
            - type: 证券类型
            - status: 上市状态
        """
        # 检查缓存是否有效
        now = datetime.now()
        if (self._stock_list_cache is not None and 
            self._cache_time is not None and
            now - self._cache_time < self._cache_duration):
            return self._stock_list_cache
            
    
        for retry in range(self._max_retries):
            try:
                print(f"[INFO] 尝试第{retry + 1}次获取股票列表")
                # 检查登录状态
                try:
                    print("[INFO] Baostock登录成功")
                except Exception as login_err:
                    print(f"[ERROR] Baostock登录失败: {login_err}")
                    raise login_err
                
                # 获取所有A股列表
                print("[INFO] 开始查询股票列表数据")
                rs = bs.query_stock_basic()
                time.sleep(self._get_api_delay())  # API调用后休眠
                
                # 检查API返回状态
                if rs.error_code != '0':
                    error_msg = f"API错误 - 错误码: {rs.error_code}, 错误信息: {rs.error_msg}"
                    print(f"[ERROR] {error_msg}")
                    raise Exception(error_msg)
                    
                print("[INFO] 股票列表数据查询成功")
                break
                
            except Exception as e:
                if retry < self._max_retries - 1:
                    print(f"[WARN] 第{retry + 1}次尝试失败: {str(e)}")
                    print(f"[INFO] 将在{self._retry_delay}秒后进行第{retry + 2}次重试")
                    time.sleep(self._retry_delay)
                    continue
                print(f"[ERROR] 所有重试均失败，最后一次错误: {str(e)}")
                return pd.DataFrame()
                
        data_list = []
        while (rs.error_code == '0') & rs.next():
            # print( rs.get_row_data() )
            data_list.append(rs.get_row_data())
            
        df = pd.DataFrame(data_list, columns=rs.fields)
            
        # 添加市场标识前缀
        '''
        def add_prefix(code):
            # 确保股票代码格式正确
            if not code.isdigit() or len(code) != 6:
                return None
            # 添加市场标识前缀
            if code.startswith('6'):
                return f'sh.{code}'
            elif code.startswith(('0', '3')):
                return f'sz.{code}'
            return None
        df['code'] = df['code'].apply(add_prefix)
            '''
            
        # 过滤掉无效的股票代码
        df = df[df['code'].notna()]
        
        # 过滤一些不要的 type 类型
        # type	证券类型，其中1：股票，2：指数，3：其它，4：可转债，5：ETF
        # status	上市状态，其中1：上市，0：退市
        if df.empty or 'type' not in df.columns:
            print("[ERROR] 获取到的股票列表数据无效或缺少必要的列")
            return pd.DataFrame()
        # df = df[df['type'].isin(['1', '2',])]
        # df = df[df['type'].isin(['1'])]
        
        print(f"[INFO] 股票列表数据数目：{len(df)}")
        print(df.head())
        
        # 格式化日期列
        if 'ipoDate' in df.columns:
            df['ipoDate'] = pd.to_datetime(df['ipoDate']).dt.strftime('%Y-%m-%d')
        if 'outDate' in df.columns:
            df['outDate'] = pd.to_datetime(df['outDate']).dt.strftime('%Y-%m-%d')

        # 保存到CSV文件
        from pathlib import Path
        from . import RAW_DATA_DIR
        raw_data_dir = RAW_DATA_DIR
        stock_list_file = raw_data_dir / 'stock_list.csv'
        df.to_csv(stock_list_file, index=False, encoding='utf-8')
        print(f"[INFO] 股票列表数据已保存到: {stock_list_file}")
        
        # 更新缓存
        self._stock_list_cache = df
        self._cache_time = now
            
        return df
        
    async def get_stock_daily(
        self,
        code: str,
        start_date: str,
        end_date: Optional[str] = None,
        frequency: str = 'd'
    ) -> pd.DataFrame:
        """获取股票K线数据
        
        Args:
            code: 股票代码（如：sh.600000）
            start_date: 开始日期（YYYY-MM-DD）
            end_date: 结束日期（YYYY-MM-DD），默认为当前日期
            frequency: 数据周期，默认为'd'（日K线）
                      - 'd': 日K线
                      - 'w': 周K线
                      - 'm': 月K线
                      - '5': 5分钟线
                      - '15': 15分钟线
                      - '30': 30分钟线
                      - '60': 60分钟线
            
        Returns:
            DataFrame包含以下字段：
            - date: 交易日期/时间
            - open: 开盘价
            - high: 最高价
            - low: 最低价
            - close: 收盘价
            - volume: 成交量
            - amount: 成交额
            - adjustflag: 复权类型
            - turn: 换手率
            - tradestatus: 交易状态
            - pctChg: 涨跌幅
            - peTTM: 动态市盈率（仅日线数据）
            - pbMRQ: 市净率（仅日线数据）
            - psTTM: 市销率（仅日线数据）
            - pcfNcfTTM: 市现率（仅日线数据）
        """
        # 验证frequency参数
        valid_frequencies = {'d', 'w', 'm', '5', '15', '30', '60'}
        if frequency.lower() not in valid_frequencies:
            print(f"无效的数据周期: {frequency}，使用默认值'd'")
            frequency = 'd'
        
        # 根据frequency选择查询字段
        if frequency.lower() == 'd':  # 日线
            fields = "date,open,high,low,close,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM"
        elif frequency.lower() in {'w', 'm'}:  # 周线、月线
            fields = "date,open,high,low,close,volume,amount,adjustflag,turn,pctChg"
        else:  # 分钟线
            fields = "date,time,open,high,low,close,volume,amount,adjustflag"
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        for retry in range(self._max_retries):
            try:
                rs = bs.query_history_k_data_plus(
                    code,
                    fields,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency.lower(),
                    adjustflag="1"  # 复权状态(1：后复权， 2：前复权，3：不复权）
                )
                time.sleep(self._get_api_delay())  # API调用后休眠
                
                if rs.error_code != '0':
                    error_msg = f"API错误 - 错误码: {rs.error_code}, 错误信息: {rs.error_msg}"
                    print(f"[ERROR] {error_msg}")
                    raise Exception(error_msg)
                    
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                    
                df = pd.DataFrame(data_list, columns=rs.fields)
                
                # 格式化日期列
                if not df.empty:
                    if frequency.lower() in {'d', 'w', 'm'}:  # 日线、周线、月线
                        if 'date' in df.columns:
                            df['date'] = pd.to_datetime(df['date'])
                    else:  # 分钟线
                        if 'date' in df.columns and 'time' in df.columns:
                            # 将time字段从'yyyyMMddHHmmssSSS'格式转换为'HH:mm:ss'格式
                            df['time'] = df['time'].apply(lambda x: x[8:10] + ':' + x[10:12] + ':' + x[12:14])
                            df['datetime'] = pd.to_datetime(df['date'] + ' ' + df['time'])
                            df.drop(['date', 'time'], axis=1, inplace=True)
                    
                # 数值类型转换
                numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
                if frequency.lower() in {'d', 'w', 'm'}:  # 日线、周线、月线额外的指标
                    numeric_columns.extend(['turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM'])
                    
                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')
                        
                # 根据时间列排序
                sort_column = 'datetime' if 'datetime' in df.columns else 'date'
                return df.sort_values(sort_column, ascending=False)
            except Exception as e:
                if retry < self._max_retries - 1:
                    print(f"第{retry + 1}次尝试失败: {e}，将在{self._retry_delay}秒后重试")
                    time.sleep(self._retry_delay)
                    continue
                print(f"获取日线数据失败: {e}")
                return pd.DataFrame()
            
    async def get_realtime_quotes(
        self,
        codes: List[str]
    ) -> Dict[str, Dict]:
        """获取实时行情数据
        
        Args:
            codes: 股票代码列表（如：['sh.600000', 'sz.000001']）
            
        Returns:
            字典，key为股票代码，value为行情数据字典
        """
        result = {}
        for code in codes:
            for retry in range(self._max_retries):
                try:
                    rs = bs.query_rt_data(code)
                    time.sleep(self._get_api_delay())  # API调用后休眠
                    if rs.error_code != '0':
                        raise Exception(f"获取实时行情失败: {rs.error_msg}")
                        
                    data_list = []
                    while (rs.error_code == '0') & rs.next():
                        data_list.append(rs.get_row_data())
                        
                    if data_list:
                        result[code] = {
                            field: value
                            for field, value in zip(rs.fields, data_list[0])
                        }
                        break
                    else:
                        result[code] = {}
                        break
                except Exception as e:
                    if retry < self._max_retries - 1:
                        print(f"第{retry + 1}次尝试失败: {e}，将在{self._retry_delay}秒后重试")
                        time.sleep(self._retry_delay)
                        continue
                    print(f"获取实时行情失败: {e}")
                    result[code] = {}
                    break
                
        return result

    async def get_adjust_factor(
        self,
        code: str,
        start_date: str = '2015-01-01',
        end_date: Optional[str] = None
    ) -> pd.DataFrame:
        """获取股票复权因子信息
        
        Args:
            code: 股票代码（如：sh.600000）
            start_date: 开始日期（YYYY-MM-DD），默认为2015-01-01
            end_date: 结束日期（YYYY-MM-DD），默认为当前日期
            
        Returns:
            DataFrame包含以下字段：
            - code: 股票代码
            - date: 交易日期
            - adjust_factor: 复权因子
        """
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')
            
        for retry in range(self._max_retries):
            try:
                print(f"[INFO] 尝试第{retry + 1}次获取股票{code}的复权因子信息")
                rs = bs.query_adjust_factor(
                    code=code,
                    # start_date=start_date,
                    # end_date=end_date
                )
                time.sleep(self._get_api_delay())
                
                if rs.error_code != '0':
                    error_msg = f"API错误 - 错误码: {rs.error_code}, 错误信息: {rs.error_msg}"
                    print(f"[ERROR] {error_msg}")
                    raise Exception(error_msg)
                    
                print(f"[INFO] 股票{code}的复权因子信息查询成功")
                break
                
            except Exception as e:
                if retry < self._max_retries - 1:
                    print(f"[WARN] 第{retry + 1}次尝试失败: {str(e)}")
                    print(f"[INFO] 将在{self._retry_delay}秒后进行第{retry + 2}次重试")
                    time.sleep(self._retry_delay)
                    continue
                print(f"[ERROR] 所有重试均失败，最后一次错误: {str(e)}")
                return pd.DataFrame()
                
        data_list = []
        while (rs.error_code == '0') & rs.next():
            data_list.append(rs.get_row_data())
            
        df = pd.DataFrame(data_list, columns=rs.fields)
        
        if not df.empty:
            # 格式化日期列
            df['dividOperateDate'] = pd.to_datetime(df['dividOperateDate']).dt.strftime('%Y-%m-%d')
            
        return df

    import pandas as pd
    from datetime import timedelta, datetime

    def query_hs300_stocks(self, start_date: str, end_date: str) -> pd.DataFrame:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        all_data = []

        current = start
        while current <= end:
            if current.day == 15:
                date_str = current.strftime('%Y-%m-%d')
                rs = bs.query_hs300_stocks(date=date_str)
                print(f"query_hs300_stocks {date_str}")

                if rs.error_code != '0':
                    print(f"Error querying data for {date_str}: {rs.error_msg}")
                    continue

                # 将结果集逐条读取并添加到列表中
                while rs.next():
                    row = rs.get_row_data()
                    # row.append(date_str)  # 添加查询日期作为原始数据的 updateDate
                    all_data.append(row)

            current += timedelta(days=1)

        rs = bs.query_hs300_stocks()

        # 将结果集逐条读取并添加到列表中
        while (rs.error_code == '0') & rs.next():
            row = rs.get_row_data()
            # row.append(date_str)  # 添加查询日期作为原始数据的 updateDate
            all_data.append(row)

        # 构建 DataFrame
        columns = ['updateDate', 'code', 'code_name']
        df = pd.DataFrame(all_data, columns=columns)

        # 处理 code 列：转大写、去点号
        df['code'] = df['code'].str.upper().str.replace('.', '', regex=False)

        # 按 code 和 code_name 分组，获取最小和最大时间
        grouped = df.groupby(['code'])['updateDate'].agg(['min', 'max'])
        grouped.columns = ['start_time', 'end_time']
        grouped = grouped.reset_index()
        # 保存结果
        return grouped

    def query_zz500_stocks(self, start_date: str, end_date: str ) -> pd.DataFrame:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        all_data = []

        current = start
        while current <= end:
            if current.day == 15:
                date_str = current.strftime('%Y-%m-%d')
                print(f"query_zz500_stocks {date_str}")

                rs = bs.query_zz500_stocks(date=date_str)

                if rs.error_code != '0':
                    print(f"Error querying data for {date_str}: {rs.error_msg}")
                    continue

                # 将结果集逐条读取并添加到列表中
                while rs.next():
                    row = rs.get_row_data()
                    # row.append(date_str)  # 添加查询日期作为原始数据的 updateDate
                    all_data.append(row)

            current += timedelta(days=1)

        rs = bs.query_zz500_stocks()

        # 将结果集逐条读取并添加到列表中
        while (rs.error_code == '0') & rs.next():
            row = rs.get_row_data()
            # row.append(date_str)  # 添加查询日期作为原始数据的 updateDate
            all_data.append(row)
        # 构建 DataFrame
        columns = ['updateDate', 'code', 'code_name']
        df = pd.DataFrame(all_data, columns=columns)

        # 处理 code 列：转大写、去点号
        df['code'] = df['code'].str.upper().str.replace('.', '', regex=False)

        # 按 code 和 code_name 分组，获取最小和最大时间
        grouped = df.groupby(['code'])['updateDate'].agg(['min', 'max'])
        grouped.columns = ['start_time', 'end_time']
        grouped = grouped.reset_index()
        # 保存结果
        return grouped

    def query_sz50_stocks(self, start_date: str, end_date: str ) -> pd.DataFrame:
        start = datetime.strptime(start_date, '%Y-%m-%d')
        end = datetime.strptime(end_date, '%Y-%m-%d')

        all_data = []

        current = start
        while current <= end:
            if current.day == 15:
                date_str = current.strftime('%Y-%m-%d')
                rs = bs.query_sz50_stocks(date=date_str)
                print(f"query_sz50_stocks {date_str}")

                if rs.error_code != '0':
                    print(f"Error querying data for {date_str}: {rs.error_msg}")
                    continue

                # 将结果集逐条读取并添加到列表中
                while rs.next():
                    row = rs.get_row_data()
                    # row.append(date_str)  # 添加查询日期作为原始数据的 updateDate
                    all_data.append(row)

            current += timedelta(days=1)

        rs = bs.query_sz50_stocks()

        # 将结果集逐条读取并添加到列表中
        while (rs.error_code == '0') & rs.next():
            row = rs.get_row_data()
            # row.append(date_str)  # 添加查询日期作为原始数据的 updateDate
            all_data.append(row)

        print(all_data[0])
        # 构建 DataFrame
        columns = ['updateDate', 'code', 'code_name']
        df = pd.DataFrame(all_data, columns=columns)

        # 处理 code 列：转大写、去点号
        df['code'] = df['code'].str.upper().str.replace('.', '', regex=False)

        # 按 code 和 code_name 分组，获取最小和最大时间
        grouped = df.groupby(['code'])['updateDate'].agg(['min', 'max'])
        grouped.columns = ['start_time', 'end_time']
        grouped = grouped.reset_index()
        # 保存结果
        return grouped


    def get_index_constituents(self, index_type: str) -> pd.DataFrame:
        """获取指数成分股
        
        Args:
            index_type: 指数类型
                - 'sz50': 上证50
                - 'hs300': 沪深300
                - 'zz500': 中证500
                
        Returns:
            DataFrame包含以下字段：
            - code: 股票代码
            - code_name: 股票名称
            - date: 纳入日期
        """
        try:
            print(f"[INFO] 尝试获取{index_type}成分股")

            # 获取当前时间，并转换成日期字符串，格式为 '%Y-%m-%d'
            current_date_str = datetime.now().strftime('%Y-%m-%d')

            # 根据指数类型选择查询方法
            if index_type == 'sz50':
                df = self.query_sz50_stocks(start_date='2010-01-01', end_date=current_date_str)
            elif index_type == 'hs300':
                df = self.query_hs300_stocks(start_date='2010-01-01', end_date=current_date_str)
            elif index_type == 'zz500':
                df = self.query_zz500_stocks(start_date='2010-01-01', end_date=current_date_str)
            else:
                raise ValueError(f"不支持的指数类型: {index_type}")

            time.sleep(self._get_api_delay())

            print(f"[INFO] {index_type}成分股查询成功")

        except Exception as e:
            print(f"[ERROR] 失败，错误: {str(e)}")
            return pd.DataFrame()

        if not df.empty:
            # 格式化日期列
            if 'updateDate' in df.columns:
                df['updateDate'] = pd.to_datetime(df['updateDate']).dt.strftime('%Y-%m-%d')
                
        return df