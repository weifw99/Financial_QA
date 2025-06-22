"""数据同步模块

主要功能：
1. 全量数据同步
2. 增量数据同步
3. 数据存储管理
"""

import os
import time
from datetime import datetime, timedelta
from typing import List, Optional
import re
import multiprocessing as mp
from functools import partial

import pandas as pd

from .market import MarketDataAPI
from .financial import FinancialDataAPI
from .news import NewsDataAPI
from .connection import ConnectionManager

from src.data.zh_data.configs.config import SYNC_CONFIG
from . import BASE_DIR, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR
# 删除validate_stock_code函数

class DataSync:
    def __init__(self, base_dir: str = DATA_DIR):
        print("DataSync.__init__", base_dir)
        """初始化数据同步模块
        
        Args:
            base_dir: 数据存储根目录
        """
        self.base_dir = base_dir
        self._conn_manager = ConnectionManager()
        self._conn_manager.login()
        self.market_api = MarketDataAPI(connection_manager=self._conn_manager)
        self.financial_api = FinancialDataAPI(connection_manager=self._conn_manager)
        self.news_api = NewsDataAPI(connection_manager=self._conn_manager)
        
        
        # 创建数据存储目录
        self.market_dir = os.path.join(base_dir, 'market')
        self.financial_dir = os.path.join(base_dir, 'financial')
        self.news_dir = os.path.join(base_dir, 'news')
        
        for dir_path in [self.market_dir, self.financial_dir, self.news_dir]:
            if not os.path.exists(dir_path):
                os.makedirs(dir_path)
    def _ensure_logout(self):
        """确保登出"""
        print("DataSync._conn_manager: 登出")
        self._conn_manager.logout()  
                
    @staticmethod
    def _process_stock_batch(stock_codes: List[str]):
        """处理一批股票数据
        
        Args:
            stock_codes: 需要处理的股票代码列表
        """
        # 创建新的DataSync实例
        sync = DataSync()
        
        async def process_stocks():
            for i, code in enumerate(stock_codes):
                print(f"index: {i}, dealing stock: {code}, dealing date: {datetime.now().strftime('%Y-%m-%d:%H:%M:%S')}")
                if 'market' in SYNC_CONFIG['data_types']:
                    
                    # 添加数据获取逻辑，获取复权因子信息数据， 保存到csv文件，接口使用 
                    # rs_factor = bs.query_adjust_factor(code="sh.600000", start_date="2015-01-01", end_date="2017-12-31")
                    # 参数含义：
                        # code：股票代码，sh或sz.+6位数字代码，或者指数代码，如：sh.601398。sh：上海；sz：深圳。此参数不可为空；
                        # start_date：开始日期，为空时默认为2015-01-01，包含此日期；
                        # end_date：结束日期，为空时默认当前日期，包含此日期。
                    
                    # 获取复权因子信息
                    '''
                    adjust_factor_data = await sync.market_api.get_adjust_factor(
                        code=code,
                        start_date='2010-01-01'  # 与K线数据保持一致
                    )
                    if not adjust_factor_data.empty:
                        sync._save_data(adjust_factor_data, sync.market_dir, code, 'adjust_factor')
                    '''
                    print(f"index: {i}, stock: {code}, full sync, dealing date: {datetime.now().strftime('%Y-%m-%d:%H:%M:%S')}")
                    await sync._sync_single_stock_full(code)

                if 'financial' in SYNC_CONFIG['data_types']:
                    income_file = os.path.join(sync.financial_dir, code, 'income.csv')
                    if not os.path.exists(income_file):
                        print(f"financial index: {i}, stock: {code}, full sync, dealing date: {datetime.now().strftime('%Y-%m-%d:%H:%M:%S')}")
                        await sync._sync_single_financial_full(code)
                    else:
                        print(f"financial index: {i}, stock: {code}, incremental sync, dealing date: {datetime.now().strftime('%Y-%m-%d:%H:%M:%S')}")
                        await sync._sync_single_financial_incremental(code)

                # 检查股票文件
        
        # 在每个进程中运行异步任务
        import asyncio
        asyncio.run(process_stocks())
        
        # 在子进程完成后执行登出操作
        # sync._ensure_logout()
    
    async def sync_stock(self):
        """根据配置同步股票数据"""
        
        print(f"SYNC_CONFIG: {SYNC_CONFIG}")
        
        # 获取成分股，并保存
        index_types = ['sz50', 'hs300', 'zz500', 'zz1000']
        # index_types = [ 'zz500']
        # index_types = []
        for index_type in index_types:
            print(f"[INFO] 开始获取{index_type}成分股")
            constituents = self.market_api.get_index_constituents(index_type)
            if not constituents.empty:
                # 保存成分股数据
                from . import RAW_DATA_DIR
                raw_data_dir = RAW_DATA_DIR
                index_dir = os.path.join(raw_data_dir, 'index')
                if not os.path.exists(index_dir):
                    os.makedirs(index_dir)
                file_path = os.path.join(index_dir, f'{index_type}_constituents.csv')
                constituents.to_csv(file_path, index=False)
                print(f"[INFO] {index_type}成分股数据已保存到: {file_path}")
            else:
                print(f"[WARN] 获取{index_type}成分股失败")
        
        # 获取股票列表
        stock_list = self.market_api.get_stock_list()
        if stock_list.empty:
            print("获取股票列表失败")
            return

        stock_list = stock_list[stock_list['type'].isin(['1', '2', ])]
        # 获取所有股票代码并随机打乱顺序
        stock_codes = stock_list['code'].tolist()
        import random
        random.shuffle(stock_codes)
        
        # 根据进程数将股票列表分组
        process_num = SYNC_CONFIG.get('process_num', 4)
        print(f"进程数目: {process_num}")
        batch_size = len(stock_codes) // process_num + (1 if len(stock_codes) % process_num else 0)
        stock_batches = [stock_codes[i:i + batch_size] for i in range(0, len(stock_codes), batch_size)]
        print(f"batch_size: {batch_size}, stock_batches: {len(stock_batches)}")
        
        # 创建进程池
        with mp.Pool(process_num) as pool:
            # 使用进程池并行处理每批股票
            pool.map(partial(self._process_stock_batch), stock_batches)
            
            print("所有数据同步任务已完成")
        self._ensure_logout()

    async def _sync_single_stock_full(self, code: str):
        """全量同步单只股票的所有数据

        Args:
            code: 股票代码
        """

        type_file_map = {
            'd': 'daily',
            'a_d': 'daily_a',
            'w': 'weekly',
            'm': 'monthly',
            '5': 'min5',
            '15': 'min15',
            '30': 'min30',
            '60': 'min60'
        }

        try:
            # 同步行情数据（从2010年开始）
            # 定义所有需要同步的周期
            # frequencies = ['d', 'w', 'm', '5', '15', '30', '60']
            frequencies = ['d', 'w', 'm', '15', ]
            frequencies = ['d', 'a_d', 'w', 'm', '15', ]
            for freq in frequencies:

                _file = os.path.join(self.market_dir, code, f'{type_file_map[freq]}.csv')

                if os.path.exists(_file):
                    df = pd.read_csv(_file)
                    if 'datetime' in df.columns:
                        df['date'] = df['datetime']
                    start_date = pd.to_datetime(df['date']).max().strftime('%Y-%m-%d')
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    # today = datetime.now().strftime('%Y-%m-%d')
                    # yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                    # if today in df['date'].values or yesterday in df['date'].values:
                    print(f"股票{code}-freq:{freq} 已存在{type_file_map[freq]}增量数据，开始同步，start_date:{start_date}, end_date:{end_date}")
                    today = datetime.now().strftime('%Y-%m-%d')
                    yesterday = (datetime.now() - timedelta(days=3)).strftime('%Y-%m-%d')
                    # if today in df['date'].values or yesterday in df['date'].values:
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    if freq in ['w', 'm']  or today in df['date'].values or yesterday in df['date'].values:
                        print(f"股票{code}已有最新数据，跳过增量同步")
                        continue
                    # 获取对应周期的数据
                    kline_data = await self.market_api.get_stock_daily(
                        code,
                        start_date=start_date,
                        end_date=end_date,
                        frequency=freq
                    )
                    if not kline_data.empty:
                        # 根据周期设置保存的文件名
                        file_type = type_file_map[freq]
                        self._save_data(kline_data, self.market_dir, code, file_type, mode='a')

                else:
                    print(f"股票{code}-freq:{freq} 不存在{type_file_map[freq]}全量数据，开始同步")

                    # 获取对应周期的数据
                    kline_data = await self.market_api.get_stock_daily(
                        code,
                        start_date='2010-01-01',
                        frequency=freq
                    )
                    if not kline_data.empty:
                        # 根据周期设置保存的文件名
                        file_type = type_file_map[freq]
                        self._save_data(kline_data, self.market_dir, code, file_type)
                time.sleep(0.2)

        except Exception as e:
            print(f"同步股票{code}数据失败: {e}")
            print(f"错误详情: {str(e.__class__.__name__)}: {str(e)}")
            print(f"发生错误的位置: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")

    async def _sync_single_financial_full(self, code: str):
        """全量同步单只股票的所有数据

        Args:
            code: 股票代码
        """
        try:
            # 同步财务数据
            # 利润表
            income_data = self.financial_api.get_income_statement(
                code
            )
            if not income_data.empty:
                self._save_data(income_data, self.financial_dir, code, 'income')

        except Exception as e:
            print(f"同步股票{code}数据失败: {e}")
            print(f"错误详情: {str(e.__class__.__name__)}: {str(e)}")
            print(f"发生错误的位置: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")

    async def _sync_single_financial_incremental(self, code: str):
        """增量同步单只股票的最新数据

        Args:
            code: 股票代码
        """
        try:

            # 同步财务数据
            # 获取当前年度和季度
            now = datetime.now()
            current_year = now.year
            current_quarter = (now.month - 1) // 3 + 1

            # 利润表
            income_data = self.financial_api.get_income_statement(
                code,
                year=current_year,
                quarter=current_quarter
            )
            if not income_data.empty:
                self._save_data(income_data, self.financial_dir, code, 'income', mode='a')

        except Exception as e:
            print(f"增量同步股票{code}数据失败: {e}")
            print(f"错误详情: {str(e.__class__.__name__)}: {str(e)}")
            print(f"发生错误的位置: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")


    def _save_data(self, df: pd.DataFrame, base_dir: str, code: str,
                   data_type: str, mode: str = 'w') -> None:
        """保存数据到CSV文件
        
        Args:
            df: 数据框
            base_dir: 基础目录
            code: 股票代码
            data_type: 数据类型
            mode: 写入模式，'w'为覆盖，'a'为追加
        """
        # 创建股票专属目录
        stock_dir = os.path.join(base_dir, code)
        if not os.path.exists(stock_dir):
            os.makedirs(stock_dir)
            
        # 文件路径
        file_path = os.path.join(stock_dir, f'{data_type}.csv')
        
        try:
            # print(f"code: {code}, data_tyle: {data_type}, mode: {mode}, df columns: {df.columns}")
            # 如果文件存在且是增量模式，合并数据并去重
            if os.path.exists(file_path) and mode == 'a':
                # print(f"文件存在，合并数据并去重: {file_path}")
                # 读取现有数据
                existing_data = pd.read_csv(file_path )
                # existing_data = existing_data[4:].reset_index(drop=True)
                # 统一日期格式为YYYY-MM-DD
                if data_type == 'daily_a' or data_type == 'daily' or data_type == 'weekly' or data_type == 'monthly':
                    # print(f"existing_data: {existing_data.columns}")
                    #print(f"df: {df.columns}")
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    existing_data['date'] = pd.to_datetime(existing_data['date']).dt.strftime('%Y-%m-%d')
                if data_type == 'min5' or data_type == 'min15' or data_type == 'min30' or data_type == 'min60':
                    df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
                    existing_data['datetime'] = pd.to_datetime(existing_data['datetime']).dt.strftime('%Y-%m-%d %H:%M:%S')
                
                # 合并数据，新数据在前
                merged_data = pd.concat([df, existing_data], ignore_index=True)
                # 根据数据类型选择去重的关键字段
                if data_type == 'daily_a' or data_type == 'daily' or data_type == 'weekly' or data_type == 'monthly':
                    # 行情数据按日期去重
                    merged_data = merged_data.drop_duplicates(subset=['date'], keep='first')
                elif data_type == 'min5' or data_type == 'min15' or data_type == 'min30' or data_type == 'min60':
                    # 行情数据按日期去重
                    merged_data = merged_data.drop_duplicates(subset=['datetime'], keep='first')
                elif data_type in [ 'income']:
                    # 财务数据按年度和季度去重
                    merged_data.ffill(inplace= True)  # 对 null 使用前先填充， 主要是MBRevenue
                    merged_data = merged_data.drop_duplicates(subset=['pubDate'], keep='first')
                elif data_type in ['balance']:
                    # 财务数据按年度和季度去重
                    merged_data = merged_data.drop_duplicates(subset=['year', 'quarter'], keep='first')
                elif data_type == 'announcements':
                    # 公告数据按公告ID去重
                    merged_data = merged_data.drop_duplicates(subset=['announcement_id'], keep='first')
                # 保存合并后的数据
                merged_data.to_csv(file_path, index=False)
            else:
                # 文件不存在或覆盖模式，直接保存
                df.to_csv(file_path, mode=mode, index=False)
        except Exception as e:
            print(f"保存数据失败: {e}")
            print(f"错误详情: {str(e.__class__.__name__)}: {str(e)}")
            print(f"发生错误的位置: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")
            raise e
