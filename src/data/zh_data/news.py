"""新闻数据获取模块

主要功能：
1. 公司公告
"""

from typing import Dict, List, Optional
from datetime import datetime
import random

import pandas as pd
import baostock as bs
import aiohttp
import time

from .connection import ConnectionManager

class NewsDataAPI:
    def __init__(self, token: str = None, connection_manager: ConnectionManager = None):
        self._max_retries = 3
        self._retry_delay = 0.1  # 重试间隔（秒）
        self._api_delay = lambda: random.uniform(0.1, 0.15)  # API调用间隔时间（秒）
        self._conn_manager = connection_manager
        
    def get_company_announcements(
        self,
        code: str,
        year: int = None,
        quarter: int = None
    ) -> pd.DataFrame:
        """获取公司公告数据
        
        Args:
            code: 股票代码，sh或sz.+6位数字代码，如：sh.601398
            year: 统计年份，为空时默认当前年
            quarter: 统计季度(1-4)，为空时默认当前季度
            
        Returns:
            DataFrame包含公司公告数据
        """
        try:
            # 参数验证
            if not code.startswith(('sh.', 'sz.')):
                print(f"无效的股票代码格式: {code}，应以'sh.'或'sz.'开头")
                return pd.DataFrame()
                
            # 验证year参数
            current_year = datetime.now().year
            if year is not None and (not isinstance(year, int) or year < 1990 or year > current_year):
                print(f"无效的年份: {year}，应为1990到{current_year}之间的整数")
                return pd.DataFrame()
                
            # 验证quarter参数
            if quarter is not None and quarter not in [1, 2, 3, 4]:
                print(f"无效的季度: {quarter}，应为1-4之间的整数")
                return pd.DataFrame()
                
            rs = bs.query_operation_data(
                code=code,
                #year=year,
                #quarter=quarter
            )
            time.sleep(self._api_delay())  # API调用后休眠
            if rs.error_code != '0':
                print(f"获取公司公告失败: {rs.error_msg}")
                return pd.DataFrame()
                
            data_list = []
            while (rs.error_code == '0') & rs.next():
                data_list.append(rs.get_row_data())
                
            df = pd.DataFrame(data_list, columns=rs.fields)
            return df.sort_values('publishDate', ascending=False) if not df.empty else df
        except Exception as e:
            print(f"获取公司公告失败: {e}")
            return pd.DataFrame()