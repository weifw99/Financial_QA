"""财务数据获取模块

主要功能：
1. 财务报表数据（资产负债表、利润表、现金流量表）
2. 盈利预测数据
3. 财务指标（市盈率、市净率等）
"""

from typing import Dict, List, Optional
from datetime import datetime
import time
import random

import pandas as pd
import baostock as bs

from .connection import ConnectionManager

class FinancialDataAPI:
    def __init__(self, token: str = None, connection_manager: ConnectionManager = None):
        self._max_retries = 3
        self._retry_delay = 0.1  # 重试间隔（秒）
        self._api_delay = lambda: random.uniform(0.1, 0.15)  # API调用间隔时间（秒）
        self._conn_manager = ConnectionManager()
            
    def get_income_statement(
        self,
        code: str,
        year: int = None,
        quarter: int = None
    ) -> pd.DataFrame:
        for retry in range(self._max_retries):
            try:
                rs = bs.query_profit_data(
                    code=code,
                    # year=year,
                    # quarter=quarter
                )
                time.sleep(self._api_delay())  # API调用后休眠
                
                if rs.error_code != '0':
                    raise Exception(f"获取利润表数据失败: {rs.error_msg}")
                    
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                    
                df = pd.DataFrame(data_list, columns=rs.fields)
                return df.sort_values('statDate', ascending=False)
            except Exception as e:
                if retry < self._max_retries - 1:
                    print(f"第{retry + 1}次尝试失败: {e}，将在{self._retry_delay}秒后重试")
                    time.sleep(self._retry_delay)
                    continue
                print(f"获取利润表数据失败: {e}")
                return pd.DataFrame()
            
    def get_balance_sheet(
        self,
        code: str,
        year: int = None,
        quarter: int = None
    ) -> pd.DataFrame:
        for retry in range(self._max_retries):
            try:
                rs = bs.query_balance_data(
                    code=code,
                    # year=year,
                    # quarter=quarter
                )
                time.sleep(self._api_delay())  # API调用后休眠
                
                if rs.error_code != '0':
                    raise Exception(f"获取资产负债表数据失败: {rs.error_msg}")
                    
                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())
                    
                df = pd.DataFrame(data_list, columns=rs.fields)
                return df.sort_values('statDate', ascending=False)
            except Exception as e:
                if retry < self._max_retries - 1:
                    print(f"第{retry + 1}次尝试失败: {e}，将在{self._retry_delay}秒后重试")
                    time.sleep(self._retry_delay)
                    continue
                print(f"获取资产负债表数据失败: {e}")
                return pd.DataFrame()
            