"""数据同步模块

主要功能：
1. 全量数据同步
2. 增量数据同步
3. 数据存储管理
"""

import os
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
                    
                    # 检查数据文件是否存在
                    daily_file = os.path.join(sync.market_dir, code, 'daily.csv')
                    # min5_file = os.path.join(sync.market_dir, code, 'min5.csv')
                    # min15_file = os.path.join(sync.market_dir, code, 'min15.csv')
                    # min30_file = os.path.join(sync.market_dir, code, 'min30.csv')
                    # min60_file = os.path.join(sync.market_dir, code, 'min60.csv')
                    monthly_file = os.path.join(sync.market_dir, code, 'monthly.csv')
                    weekly_file = os.path.join(sync.market_dir, code, 'weekly.csv')
                    # if (not os.path.exists(daily_file) )  or (not os.path.exists(min5_file)) or (not os.path.exists(monthly_file)) or (not os.path.exists(weekly_file)) :
                    if (not os.path.exists(daily_file) ) or (not os.path.exists(monthly_file)) or (not os.path.exists(weekly_file)) :
                        # 文件不存在，执行全量同步
                        print(f"index: {i}, stock: {code}, full sync, dealing date: {datetime.now().strftime('%Y-%m-%d:%H:%M:%S')}")
                        await sync._sync_single_stock_full(code)
                    else:
                        # continue
                        # 检查daily.csv文件中的最新日期
                        daily_file = os.path.join(sync.market_dir, code, 'daily.csv')
                        if os.path.exists(daily_file):
                            print(f"股票{code}, daily_file 已存在")
                            df = pd.read_csv(daily_file)
                            df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                            today = datetime.now().strftime('%Y-%m-%d')
                            yesterday = (datetime.now() - timedelta(days=1)).strftime('%Y-%m-%d')
                            if today in df['date'].values or yesterday in df['date'].values:
                                print(f"股票{code}已有最新数据，跳过增量同步")
                                continue
                        # 文件存在，执行增量同步
                        end_date = datetime.now().strftime('%Y-%m-%d')
                        start_date = (datetime.now() - timedelta(days=SYNC_CONFIG['incremental_days'])).strftime('%Y-%m-%d')
                        await sync._sync_single_stock_incremental(code, start_date, end_date)
                    
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
        index_types = ['sz50', 'hs300', 'zz500']
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
        try:
            # 同步行情数据（从2010年开始）
            # 定义所有需要同步的周期
            # frequencies = ['d', 'w', 'm', '5', '15', '30', '60']
            frequencies = ['d', 'w', 'm', '15',]
            for freq in frequencies:
                # 获取对应周期的数据
                kline_data = await self.market_api.get_stock_daily(
                    code,
                    start_date='2010-01-01',
                    frequency=freq
                )
                if not kline_data.empty:
                    # 根据周期设置保存的文件名
                    file_type = {
                        'd': 'daily',
                        'w': 'weekly',
                        'm': 'monthly',
                        '5': 'min5',
                        '15': 'min15',
                        '30': 'min30',
                        '60': 'min60'
                    }[freq]
                    self._save_data(kline_data, self.market_dir, code, file_type)
                
            # 同步财务数据
            if 'financial' in SYNC_CONFIG['data_types']:
                current_year = datetime.now().year
                # 资产负债表
                balance_data = self.financial_api.get_balance_sheet(
                    code
                )
                if not balance_data.empty:
                    self._save_data(balance_data, self.financial_dir, code, 'balance')
                
                # 利润表
                income_data = self.financial_api.get_income_statement(
                    code
                )
                if not income_data.empty:
                    self._save_data(income_data, self.financial_dir, code, 'income')
                '''        
                for year in range(current_year - SYNC_CONFIG['financial_years'], current_year + 1):
                    for quarter in range(1, 5):
                        # 资产负债表
                        balance_data = self.financial_api.get_balance_sheet(
                            code,
                            year=year,
                            quarter=quarter
                        )
                        if not balance_data.empty:
                            self._save_data(balance_data, self.financial_dir, code, 'balance')
                        
                        # 利润表
                        income_data = self.financial_api.get_income_statement(
                            code,
                            year=year,
                            quarter=quarter
                        )
                        if not income_data.empty:
                            self._save_data(income_data, self.financial_dir, code, 'income')
                '''
                            
            # 同步新闻数据
            if 'news' in SYNC_CONFIG['data_types']:
                start_date = (datetime.now() - timedelta(days=SYNC_CONFIG['news_days'])).strftime('%Y-%m-%d')
                end_date = datetime.now().strftime('%Y-%m-%d')
                
                # 公司公告
                # 根据news_days计算起始年份和季度
                start_date_obj = datetime.now() - timedelta(days=SYNC_CONFIG['news_days'])
                end_date_obj = datetime.now()
                
                # 获取起始和结束的年份季度
                start_year = start_date_obj.year
                start_quarter = (start_date_obj.month - 1) // 3 + 1
                end_year = end_date_obj.year
                end_quarter = (end_date_obj.month - 1) // 3 + 1
                
                # 遍历每个季度获取公告数据
                announcements_list = []
                current_year = start_year
                current_quarter = start_quarter
                
                while (current_year < end_year) or \
                      (current_year == end_year and current_quarter <= end_quarter):
                    quarter_announcements = self.news_api.get_company_announcements(
                        code,
                        year=current_year,
                        quarter=current_quarter
                    )
                    if not quarter_announcements.empty:
                        announcements_list.append(quarter_announcements)
                    
                    # 更新年份和季度
                    current_quarter += 1
                    if current_quarter > 4:
                        current_quarter = 1
                        current_year += 1
                
                # 合并所有季度的数据
                announcements = pd.concat(announcements_list) if announcements_list else pd.DataFrame()
                if not announcements.empty:
                    self._save_data(announcements, self.news_dir, code, 'announcements')
                    


        except Exception as e:
            print(f"同步股票{code}数据失败: {e}")
            print(f"错误详情: {str(e.__class__.__name__)}: {str(e)}")
            print(f"发生错误的位置: {e.__traceback__.tb_frame.f_code.co_filename}:{e.__traceback__.tb_lineno}")
            
    async def _sync_single_stock_incremental(self, code: str, start_date: str, end_date: str):
        """增量同步单只股票的最新数据
        
        Args:
            code: 股票代码
            start_date: 开始日期
            end_date: 结束日期
        """
        try:
            # 同步行情数据
            # 定义所有需要同步的周期
            # frequencies = ['d', 'w', 'm', '5', '15', '30', '60']
            frequencies = ['d', 'w', 'm', '15',]
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
                        '5': 'min5',
                        '15': 'min15',
                        '30': 'min30',
                        '60': 'min60'
                    }[freq]
                    self._save_data(kline_data, self.market_dir, code, file_type, mode='a')
                
            # 同步财务数据
            # 获取当前年度和季度
            now = datetime.now()
            current_year = now.year
            current_quarter = (now.month - 1) // 3 + 1
            
            # 资产负债表
            balance_data = self.financial_api.get_balance_sheet(
                code,
                year=current_year,
                quarter=current_quarter
            )
            if not balance_data.empty:
                self._save_data(balance_data, self.financial_dir, code, 'balance', mode='a')
                
            # 利润表
            income_data = self.financial_api.get_income_statement(
                code,
                year=current_year,
                quarter=current_quarter
            )
            if not income_data.empty:
                self._save_data(income_data, self.financial_dir, code, 'income', mode='a')
                
            # 同步新闻数据
            # 公司公告
            # 根据start_date和end_date计算年份和季度
            start_date_obj = datetime.strptime(start_date, '%Y-%m-%d')
            end_date_obj = datetime.strptime(end_date, '%Y-%m-%d')
            
            # 获取起始和结束的年份季度
            start_year = start_date_obj.year
            start_quarter = (start_date_obj.month - 1) // 3 + 1
            end_year = end_date_obj.year
            end_quarter = (end_date_obj.month - 1) // 3 + 1
            
            # 遍历每个季度获取公告数据
            announcements_list = []
            current_year = start_year
            current_quarter = start_quarter
            
            while (current_year < end_year) or \
                  (current_year == end_year and current_quarter <= end_quarter):
                quarter_announcements = self.news_api.get_company_announcements(
                    code,
                    year=current_year,
                    quarter=current_quarter
                )
                if not quarter_announcements.empty:
                    announcements_list.append(quarter_announcements)
                
                # 更新年份和季度
                current_quarter += 1
                if current_quarter > 4:
                    current_quarter = 1
                    current_year += 1
            
            # 合并所有季度的数据
            announcements = pd.concat(announcements_list) if announcements_list else pd.DataFrame()
            if not announcements.empty:
                self._save_data(announcements, self.news_dir, code, 'announcements', mode='a')
                


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
            # 如果文件存在且是增量模式，合并数据并去重
            if os.path.exists(file_path) and mode == 'a':
                # print(f"文件存在，合并数据并去重: {file_path}")
                # 读取现有数据
                existing_data = pd.read_csv(file_path )
                # existing_data = existing_data[4:].reset_index(drop=True)
                # 统一日期格式为YYYY-MM-DD
                if data_type == 'daily' or data_type == 'weekly' or data_type == 'monthly':

                    # print(f"existing_data: {existing_data.columns}")
                    #print(f"df: {df.columns}")
                    df['date'] = pd.to_datetime(df['date']).dt.strftime('%Y-%m-%d')
                    existing_data['date'] = pd.to_datetime(existing_data['date']).dt.strftime('%Y-%m-%d')
                
                # 合并数据，新数据在前
                merged_data = pd.concat([df, existing_data], ignore_index=True)
                # 根据数据类型选择去重的关键字段
                if data_type == 'daily':
                    # 行情数据按日期去重
                    merged_data = merged_data.drop_duplicates(subset=['date'], keep='first')
                elif data_type in ['balance', 'income']:
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
