"""港股数据同步模块

主要功能：
1. 全量数据同步
2. 增量数据同步
3. 数据存储管理
"""

import os
from datetime import datetime, timedelta
from typing import List, Optional
import multiprocessing as mp
from functools import partial
import logging

import pandas as pd

from .market import MarketDataAPI
from . import BASE_DIR, DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR

class DataSync:
    def __init__(self, base_dir: str = DATA_DIR, process_num: int = 4,
                 start_date: str = '2010-01-01', end_date: Optional[str] = None,
                 full_sync: bool = False):
        """初始化数据同步模块
        
        Args:
            base_dir: 数据存储根目录
            process_num: 并行进程数
            start_date: 数据开始日期
            end_date: 数据结束日期
            full_sync: 是否执行全量同步
        """
        self.base_dir = base_dir
        self.process_num = process_num
        self.start_date = start_date
        self.end_date = end_date or datetime.now().strftime('%Y-%m-%d')
        self.full_sync = full_sync
        self.market_api = MarketDataAPI()
        
        # 创建数据存储目录
        self.market_dir = os.path.join(base_dir, 'market')
        
        if not os.path.exists(self.market_dir):
            os.makedirs(self.market_dir)
            
    @staticmethod
    def _process_stock_batch(stock_codes: List[str], sync_instance):
        """处理一批股票数据
        
        Args:
            stock_codes: 需要处理的股票代码列表
            sync_instance: DataSync实例
        """
        async def process_stocks():
            for i, code in enumerate(stock_codes):
                logging.info(f"index: {i}, dealing stock: {code}, dealing date: {datetime.now().strftime('%Y-%m-%d:%H:%M:%S')}")
                
                # 检查数据文件是否存在
                daily_file = os.path.join(sync_instance.market_dir, code, 'daily.csv')
                weekly_file = os.path.join(sync_instance.market_dir, code, 'weekly.csv')
                monthly_file = os.path.join(sync_instance.market_dir, code, 'monthly.csv')
                min5_file = os.path.join(sync_instance.market_dir, code, 'min5.csv')
                min15_file = os.path.join(sync_instance.market_dir, code, 'min15.csv')
                min30_file = os.path.join(sync_instance.market_dir, code, 'min30.csv')
                min60_file = os.path.join(sync_instance.market_dir, code, 'min60.csv')
                
                if sync_instance.full_sync or (not os.path.exists(daily_file) or 
                    not os.path.exists(weekly_file) or 
                    not os.path.exists(monthly_file) or
                    not os.path.exists(min5_file) or
                    not os.path.exists(min15_file) or
                    not os.path.exists(min30_file) or
                    not os.path.exists(min60_file)):
                    # 文件不存在或需要全量同步
                    await sync_instance._sync_single_stock_full(code)
                else:
                    # 检查daily.csv文件中的最新日期
                    if os.path.exists(daily_file):
                        df = pd.read_csv(daily_file)
                        df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                        today = datetime.now().strftime('%Y-%m-%d')
                        yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                        if today in df['date'].values or yesterday in df['date'].values:
                            logging.info(f"股票{code}已有最新数据，跳过增量同步")
                            continue
                    # 文件存在，执行增量同步
                    end_date = datetime.now().strftime('%Y-%m-%d')
                    start_date = (datetime.now() - timedelta(days=7)).strftime('%Y-%m-%d')
                    await sync_instance._sync_single_stock_incremental(code, start_date, end_date)
        
        # 在每个进程中运行异步任务
        import asyncio
        asyncio.run(process_stocks())
    
    async def sync_stock(self):
        """根据配置同步港股数据"""
        
        # 获取港股列表
        stock_list = self.market_api.get_stock_list()
        if stock_list.empty:
            logging.error("获取港股列表失败")
            return
        
        # 获取所有股票代码
        stock_codes = stock_list['code'].tolist()
        
        # 根据进程数将股票列表分组
        logging.info(f"进程数目: {self.process_num}")
        batch_size = len(stock_codes) // self.process_num + (1 if len(stock_codes) % self.process_num else 0)
        stock_batches = [stock_codes[i:i + batch_size] for i in range(0, len(stock_codes), batch_size)]
        logging.info(f"batch_size: {batch_size}, stock_batches: {len(stock_batches)}")
        
        # 创建进程池
        with mp.Pool(self.process_num) as pool:
            # 使用进程池并行处理每批股票
            pool.map(partial(self._process_stock_batch, sync_instance=self), stock_batches)
            
            logging.info("所有数据同步任务已完成")
            
    async def _sync_single_stock_full(self, code: str):
        """全量同步单只股票的所有数据
        
        Args:
            code: 股票代码
        """
        try:
            # 同步行情数据
            # 定义所有需要同步的周期
            frequencies = ['d', 'w', 'm', '5m', '15m', '30m', '1h']
            for freq in frequencies:
                # 获取对应周期的数据
                kline_data = await self.market_api.get_stock_daily(
                    code,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    frequency=freq
                )
                if not kline_data.empty:
                    # 根据周期设置保存的文件名
                    file_type = {
                        'd': 'daily',
                        'w': 'weekly',
                        'm': 'monthly',
                        '5m': 'min5',
                        '15m': 'min15',
                        '30m': 'min30',
                        '1h': 'min60'
                    }[freq]
                    self._save_data(kline_data, self.market_dir, code, file_type)
                    
        except Exception as e:
            logging.error(f"同步股票{code}数据失败: {e}")
            logging.error(f"错误详情: {str(e.__class__.__name__)}: {str(e)}")
            
    async def _sync_single_stock_incremental(self, code: str, start_date: str, end_date: str):
        """增量同步单只股票的数据
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        """
        try:
            # 同步行情数据
            frequencies = ['d', 'w', 'm', '5m', '15m', '30m', '1h']
            for freq in frequencies:
                # 获取对应周期的数据
                kline_data = await self.market_api.get_stock_daily(
                    code,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=freq
                )
                if not kline_data.empty:
                    # 根据周期设置保存的文件名
                    file_type = {
                        'd': 'daily',
                        'w': 'weekly',
                        'm': 'monthly',
                        '5m': 'min5',
                        '15m': 'min15',
                        '30m': 'min30',
                        '1h': 'min60'
                    }[freq]
                    self._save_data(kline_data, self.market_dir, code, file_type, mode='a')
                    
        except Exception as e:
            logging.error(f"同步股票{code}数据失败: {e}")
            logging.error(f"错误详情: {str(e.__class__.__name__)}: {str(e)}")
            
    def _save_data(self, df: pd.DataFrame, base_dir: str, code: str,
                   data_type: str, mode: str = 'w') -> None:
        """保存数据到CSV文件
        
        Args:
            df: 要保存的数据
            base_dir: 基础目录
            code: 股票代码
            data_type: 数据类型
            mode: 写入模式，'w'为覆盖，'a'为追加
        """
        # 创建股票代码目录
        stock_dir = os.path.join(base_dir, code)
        if not os.path.exists(stock_dir):
            os.makedirs(stock_dir)
            
        # 保存数据
        file_path = os.path.join(stock_dir, f'{data_type}.csv')
        if mode == 'w' or not os.path.exists(file_path):
            df.to_csv(file_path, index=False, encoding='utf-8')
            logging.info(f"保存数据到文件: {file_path}")
        else:
            # 追加模式，需要处理重复数据
            existing_df = pd.read_csv(file_path, encoding='utf-8')
            combined_df = pd.concat([existing_df, df])
            combined_df = combined_df.drop_duplicates(subset=['date'])
            combined_df = combined_df.sort_values('date')
            combined_df.to_csv(file_path, index=False, encoding='utf-8')
            logging.info(f"追加数据到文件: {file_path}") 