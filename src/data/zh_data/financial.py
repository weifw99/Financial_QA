"""财务数据获取模块

主要功能：
1. 财务报表数据（资产负债表、利润表、现金流量表）
2. 盈利预测数据
3. 财务指标（市盈率、市净率等）
"""

from typing import Dict, List, Optional, Union
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
        year: Union[int, List[int]] = None,
        quarter: Union[int, List[int]] = None
    ) -> pd.DataFrame:

        import datetime
        if year is None:
            years = [i for i in range(2000, datetime.datetime.now().year + 1)]
            quarters = [1, 2, 3, 4]
        else:
            if type(year) == int:
                years = [year]
            else:
                years = year
            if quarter is None:
                quarters = [1, 2, 3, 4]
            else:
                if type(quarter) == int:
                    quarters = [quarter]
                else:
                    quarters = quarter

        data_list = []
        for year in years:
            for q in quarters:
                time.sleep(0.1)
                rs = bs.query_profit_data(code=code, year=year, quarter=q)
                if rs.error_code == '0':
                    while rs.next():
                        row = rs.get_row_data()
                        data_list.append(row)
        columns = ['code','pubDate','statDate', 'roeAvg', 'npMargin', 'gpMargin', 'netProfit', 'epsTTM', 'MBRevenue', 'totalShare', 'liqaShare']
        df_profit = pd.DataFrame(data_list, columns=columns).sort_values(['pubDate'])
        df_profit.ffill(inplace=True) # 对 null 使用前先填充， 主要是MBRevenue

        df_profit.dropna(inplace=True)

        print(code, 'query_profit_data' ,len(df_profit))

        return df_profit

            