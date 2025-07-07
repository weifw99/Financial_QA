import random
import traceback
from typing import Union, List, Optional, Tuple

import akshare as ak
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import os
from pathlib import Path
import glob
import json
import time

from scipy.stats import linregress

import sys

from src.busi.bond_.util.baostock_connection import BaostockConnectionManager
from src.busi.bond_.constant import DataCons
import baostock as bs

            
class BondDataHandle:
    def __init__(self):
        Path(DataCons.BOND_INFO_DIR).mkdir(parents=True, exist_ok=True)
        Path(DataCons.BOND_TRADING_DAY_DIR).mkdir(parents=True, exist_ok=True)
        Path(DataCons.STOCK_TRADING_DIR).mkdir(parents=True, exist_ok=True)
        self._connct_manage = BaostockConnectionManager()
        self._connct_manage.login()

    def momentum_func(self, the_array):
        r = np.log(the_array)
        slope, _, rvalue, _, _ = linregress(np.arange(len(r)), r)
        annualized = (1 + slope) ** 252
        return annualized * (rvalue ** 2)

    def get_bond_data(self, refresh: bool = True) -> pd.DataFrame:

        if not refresh:
            if Path(DataCons.BOND_INFO_CAT).exists():
                print(f"从本地文件加载 可转债数据 数据: {DataCons.BOND_INFO_CAT}")
                return pd.read_csv(DataCons.BOND_INFO_CAT)
        # 获取可转债数据,拼接
        bond_info_dfs: list[pd.DataFrame] = self.get_and_download_bond_info()
        bond_info_df, bond_cov_df = bond_info_dfs[0], bond_info_dfs[1]
        bond_info_df = bond_info_df.rename(columns={'债券代码': '转债代码'})

        bond_info_df = bond_info_df.drop(columns=[ co_ for co_ in ['纯债价值','转股价值','纯债溢价率','转股溢价率'] if co_ in bond_info_df.columns ])

        bond_info_df['转债代码_code'] = bond_info_df['债券代码1'].apply(lambda x: f'{x.split(".")[1].lower()}{x.split(".")[0]}')
        bond_info_df['正股代码_code'] = bond_info_df[['债券代码1', '正股代码']].apply(lambda x: f'{str(x["债券代码1"]).split(".")[1].lower()}{str(x["正股代码"]).rjust(6, "0")}', axis=1)

        # 加载可转债数据-价值分析-溢价率分析数据
        bond_trading_pds = []
        for file_path in glob.glob(DataCons.BOND_ZH_COV_VALUE_ANALYSIS_FILE_PATH.format("*")):
            print(f"从本地文件加载 可转债数据-价值分析-溢价率分析 数据: {file_path}")
            df = pd.read_csv(file_path)
            code_ = str(Path(file_path).stem).lower()
            df['转债代码_code'] = code_
            df = df.rename(columns={'日期': 'date'})
            df = df.dropna()

            #
            day_file = Path(DataCons.BOND_ZH_HS_DAILY_FILE_PATH.format(code_))
            if not day_file.exists():
                continue
            trading_day_pd = pd.read_csv(DataCons.BOND_ZH_HS_DAILY_FILE_PATH.format(code_))

            # 动量策略
            data_m5 = round(trading_day_pd['close'].rolling(5).apply(self.momentum_func).to_frame('momentum').reset_index(), 2)
            trading_day_pd['momentum_5'] = data_m5['momentum']

            # Rbreak突破策略
            trading_day_pd['pivot'] = round((trading_day_pd['high'].shift() + trading_day_pd['low'].shift() + trading_day_pd['close'].shift()) / 3, 2)  # '中枢点'
            trading_day_pd['bBreak'] = round(trading_day_pd['high'].shift() + 2 * (trading_day_pd['pivot'] - trading_day_pd['low'].shift()), 2)  # 突破买入价
            trading_day_pd['bEnter'] = round(2 * trading_day_pd['pivot'] - trading_day_pd['high'].shift(), 2)  # 反转买入价

            bond_day_result = pd.merge( df, trading_day_pd, on=['date'], how='left')
            bond_day_result = pd.merge(bond_day_result, bond_info_df, on=['转债代码_code'], how='left')

            # print( bond_day_result.columns )
            # 可转债已上市时间计算，年
            bond_day_result['public_date'] = round(abs( (pd.to_datetime(bond_day_result['date']) - pd.to_datetime(bond_day_result['上市时间'])).dt.days ) / 365.25, 1)

            # bond_day_result = bond_day_result.dropna()

            path_ = DataCons.BOND_INFO_CAT_DAY.format(code_)
            Path(path_).parent.mkdir(parents=True, exist_ok=True)

            bond_day_result.to_csv(path_.format(code_), index=False)

            bond_trading_pds.append(bond_day_result)

        bond_trading_day_pd = pd.concat(bond_trading_pds, axis=0)

        # bond_trading_day_pd['双低1'] = bond_trading_day_pd['纯债价值'] + bond_trading_day_pd['转股溢价率'] * 100
        # bond_trading_day_pd['双低2'] = 2*bond_trading_day_pd['纯债价值'] - bond_trading_day_pd['转股价值']

        bond_trading_day_pd['双低1'] = bond_trading_day_pd['收盘价'] + bond_trading_day_pd['转股溢价率'] # 小数乘以100
        bond_trading_day_pd['双低2'] = 2*bond_trading_day_pd['收盘价'] - bond_trading_day_pd['转股价值']

        # 双低策略一：双低＝转债价格＋转股溢价率×100（集思录双低），或转债价格＋转股溢价（宁稳网老式双低）
        # 双低策略二：双低＝纯债溢价＋转股溢价率 = 2×转债价格 - 债底 - 转股价值（宁稳新双低）
        # 双低策略三：转债价格和溢价率分别排序获得各自名次，两者相加获得最终名次取低
        bond_trading_day_pd.index = bond_trading_day_pd['date']
        bond_trading_day_pd.to_csv(DataCons.BOND_INFO_CAT, index=False)

        return bond_trading_day_pd

        # 加载可转债 day 数据
        bond_trading_pds = []
        # for file_path in glob.glob(DataCons.BOND_ZH_HS_DAILY_FILE_PATH.format("*")):
        #     print(f"从本地文件加载 可转债数据-价值分析-溢价率分析 数据: {file_path}")
        #     df = pd.read_csv(file_path)
        #     code_ = str(Path(file_path).stem).lower()
        #     df['转债代码_code'] = code_
        #
        #     bond_day_result = pd.merge(bond_trading_day_pd, df, on=['转债代码_code'], how='left')
        #
        #     # print(bond_day_result.columns)
        #     # print(bond_day_result.head(5))
        #
        #     bond_info_result = pd.merge(bond_info_df, bond_day_result, on=['转债代码_code'], how='left')
        #
        #     path_ = DataCons.BOND_INFO_CAT_DAY.format(code_)
        #     Path(path_).parent.mkdir(parents=True, exist_ok=True)
        #
        #     bond_info_result.to_csv(path_.format(code_), index=False)

            # ond_trading_pds.append(bond_info_result)

        # bond_day_pd = pd.concat(bond_trading_pds, axis=0)
        # bond_day_pd.to_csv(DataCons.BOND_INFO_CAT, index=False)


        '''
        bond_info_result = pd.merge(bond_info_df, bond_cov_df, on=['转债代码'], how='left')

        # 加载股票数据-日线数据
        stock_trading_pds = []
        for file_path in glob.glob(DataCons.STOCK_TRADING_DAY_FILE_PATH.format("*")):
            # print(f"从本地文件加载 股票数据-日线数据 数据: {file_path}")
            df = pd.read_csv(file_path)

            df['正股代码_code'] = str(Path(file_path).stem).lower()
            stock_trading_pds.append(df)

        stock_day_pd = pd.concat(stock_trading_pds, axis=0)
        
        '''

    def down_all_data(self):
        bond_info_dfs: list[pd.DataFrame] = self.get_and_download_bond_info()

        # symbol = 'sz128039', 代码进行转换
        # symbols = [ f'{str(code_.split(".")[1]).lower()}{code_.split(".")[0]}' for code_ in bond_info_dfs[0]['债券代码1'].tolist()]
        # self.download_bond_zh_cov_value_analysis(symbol=symbols)
        # self.download_bond_trading_day_data(symbol=symbols)
        #
        # symbols = [ f'{str(bond_code.split(".")[1])}{stock_code}' for bond_code, stock_code in zip(bond_info_dfs[0]['债券代码1'].tolist(), bond_info_dfs[0]['正股代码'].tolist())]
        # self.download_stock_trading_day_data(symbol=symbols)

        for bond_code, stock_code in zip(bond_info_dfs[0]['债券代码1'].tolist(), bond_info_dfs[0]['正股代码'].tolist()):
            bond_code_temp = f'{str(bond_code.split(".")[1]).lower()}{bond_code.split(".")[0]}'
            stock_code_temp = f'{str(bond_code.split(".")[1])}.{str(stock_code).rjust(6, "0")}'
            self.download_bond_zh_cov_value_analysis(symbol=bond_code_temp)
            self.download_bond_trading_day_data(symbol_tuple=(bond_code_temp, stock_code_temp) )
            self.download_stock_trading_day_data(symbol=stock_code_temp)
            time.sleep(random.randint(1, 3))


    @staticmethod
    def get_stock_a_indicator_lg(symbol: str) -> pd.DataFrame:
        """

        :param symbol:
        :return:
        """
        # 市盈率, 市净率, 股息率数据接口
        print(f"从 ak.stock_a_indicator_lg 获取 {symbol} 的 市盈率, 市净率, 股息率数据接口")
        stock_a_indicator_df = ak.stock_a_indicator_lg(symbol=symbol[3:])
        '''
        'trade_date', 'pe', 'pe_ttm', 'pb', 'dv_ratio', 'dv_ttm', 'ps', 'ps_ttm', 'total_mv'
        '''
        # total_mv 总市值
        # print('stock_a_indicator_df ', stock_a_indicator_df.columns )
        # print('stock_a_indicator_df ', stock_a_indicator_df.head() )
        stock_a_indicator_df['total_mv'] = round(stock_a_indicator_df['total_mv'] / 10000, 2)  # 单位 亿元
        stock_a_indicator_df = stock_a_indicator_df[['trade_date', 'total_mv']]
        stock_a_indicator_df.columns = ['date', 'total_mv']
        stock_a_indicator_df['date'] = pd.to_datetime(stock_a_indicator_df['date'])

        return stock_a_indicator_df

    @staticmethod
    def get_and_download_bond_info() -> List[pd.DataFrame]:

        results: List[pd.DataFrame] = []
        if os.path.exists(DataCons.BOND_INFO_FILE_PATH):
            print(f"可转债数据 从本地文件加载债券数据: {DataCons.BOND_INFO_FILE_PATH}")
            bond_data = pd.read_csv(DataCons.BOND_INFO_FILE_PATH)
        else:
            # 获取可转债数据
            bond_data = ak.bond_zh_cov()
            # 打印数据列名，用于调试
            print("可转债数据 获取到的数据列名：")
            print(bond_data.columns.tolist())
            # 保存数据
            bond_data.to_csv(DataCons.BOND_INFO_FILE_PATH, index=False)
        print(f"可转债数据 数据已保存至: {DataCons.BOND_INFO_FILE_PATH}")

        # data_type_dict = {'债券代码': 'str', '申购代码': 'str', '正股代码': 'str'}
        # bond_data.astype(data_type_dict)

        results.append(bond_data)

        if os.path.exists(DataCons.BOND_COV_COMPARISON_FILE_PATH):
            print(f"可转债比价表 从本地文件加载债券数据: {DataCons.BOND_COV_COMPARISON_FILE_PATH}")
            bond_cov_comparison_df = pd.read_csv(DataCons.BOND_COV_COMPARISON_FILE_PATH)
        else:
            # 获取可转债比价表
            bond_cov_comparison_df = ak.bond_cov_comparison()
            # 打印数据列名，用于调试
            print("可转债比价表 获取到的数据列名：")
            print(bond_cov_comparison_df.columns.tolist())
            # 保存数据
            bond_cov_comparison_df.to_csv(DataCons.BOND_COV_COMPARISON_FILE_PATH, index=False)
            print(f"可转债比价表 数据已保存至: {DataCons.BOND_COV_COMPARISON_FILE_PATH}")
        # data_type_dict = {'转债代码': 'str', '正股代码': 'str'}
        # bond_cov_comparison_df.astype(data_type_dict)

        results.append(bond_cov_comparison_df)

        return results

    @staticmethod
    def download_bond_zh_cov_value_analysis(symbol: Union[str, List[str]] = "113527", skip_exists: bool = False) -> None:
        # 从接口获取数据
        if isinstance(symbol, str):
            symbols = [symbol]
        else:
            symbols = symbol
        for symbol in symbols:
            path_ = DataCons.BOND_ZH_COV_VALUE_ANALYSIS_FILE_PATH.format(symbol)
            Path(path_).parent.mkdir(parents=True, exist_ok=True)
            if skip_exists:
                if os.path.exists(path_):
                    print(f"从本地文件已存在 可转债数据-价值分析-溢价率分析 数据: {symbol}")
                    continue
            try:
                print(f"从接口获取 可转债数据-价值分析-溢价率分析 数据: {symbol}")
                if symbols[0] in [str(i) for i in range(10)] :
                    hist = ak.bond_zh_cov_value_analysis(symbol=symbol)
                else:
                    hist = ak.bond_zh_cov_value_analysis(symbol=symbol[2:])
                if hist is not None and not hist.empty:
                    # 保存到本地
                    hist.to_csv(path_, index=False)
            except Exception as e:
                print(f"获取债券 {symbol} 可转债数据-价值分析-溢价率分析数据失败: {e}")

    @staticmethod
    def download_bond_trading_day_data(symbol_tuple: Union[Tuple[str, str], List[Tuple[str, str]]], skip_exists: bool = False) -> None:
        """
        :param symbol_tuple:
        :return:
        """
        if isinstance(symbol_tuple, Tuple):
            symbol_tuples = [symbol_tuple]
        else:
            symbol_tuples = symbol_tuple

        for symbol, stock_code in symbol_tuples:
            path_ = DataCons.BOND_ZH_HS_DAILY_FILE_PATH.format(symbol)
            Path(path_).parent.mkdir(parents=True, exist_ok=True)
            if skip_exists:
                if os.path.exists(path_):
                    print(f"从本地文件已存在 可转债日频交易 数据: {symbol}-{stock_code}")
                    continue
            try:
                print(f"从接口获取 可转债日频交易 数据: {symbol}-{stock_code}")
                bond_zh_hs_cov_daily_df = ak.bond_zh_hs_cov_daily(symbol=symbol) # sz128039

                print('download_bond_trading_day_data', symbol, len(bond_zh_hs_cov_daily_df))

                # time.sleep(10)
                if bond_zh_hs_cov_daily_df is not None and not bond_zh_hs_cov_daily_df.empty:
                    # 保存到本地
                    # temp_df = BondDataHandle.get_stock_a_indicator_lg(symbol=stock_code)
                    # temp_df['date'] = pd.to_datetime(temp_df['date'])
                    # bond_zh_hs_cov_daily_df['date'] = pd.to_datetime(bond_zh_hs_cov_daily_df['date'])
                    #
                    # print(len(temp_df), len(bond_zh_hs_cov_daily_df))
                    #
                    # bond_zh_hs_cov_daily_df = pd.merge(bond_zh_hs_cov_daily_df, temp_df, on=['date'], how='left')
                    bond_zh_hs_cov_daily_df['factor'] = 1

                    bond_zh_hs_cov_daily_df.to_csv(path_, index=False)
            except Exception as e:
                print(f"获取债券 {symbol} 可转债日频交易数据失败: {e}")
                traceback.print_exc()



    @staticmethod
    def download_stock_trading_day_data(symbol: Union[str, List[str]],
                                        start_date: str = '2010-01-01',
                                        end_date: Optional[str] = None) -> None:
        """

        :param symbol:
        :param start_date:
        :param end_date:
        :return:
        """
        if isinstance(symbol, str):
            symbols = [symbol]
        else:
            symbols = symbol
        print(f"从接口获取 股票日频交易 数据: {symbol}")
        frequency = 'd'
        fields = "date,open,high,low,close,volume,amount,adjustflag,turn,tradestatus,pctChg,peTTM,pbMRQ,psTTM,pcfNcfTTM"
        if end_date is None:
            end_date = datetime.now().strftime('%Y-%m-%d')

        for symbol in symbols:
            convert_symbol = symbol.replace('.', '').upper()
            path_ = DataCons.STOCK_TRADING_DAY_FILE_PATH.format(convert_symbol)
            Path(path_).parent.mkdir(parents=True, exist_ok=True)
            bef_df: pd.DataFrame = None
            if os.path.exists(path_):
                print(f"从本地文件已存在 股票日频交易 数据: {symbol}, 拉取最新的数据，并 merge")
                bef_df = pd.read_csv(path_)
                if bef_df is not None and not bef_df.empty:
                    bef_df = bef_df.drop_duplicates(subset=['date'], keep='last')
                    bef_df = bef_df.sort_values(by='date', ascending=False)
                    start_date = bef_df.iloc[0]['date']
            try:

                rs = bs.query_history_k_data_plus(
                    symbol,
                    fields,
                    start_date=start_date,
                    end_date=end_date,
                    frequency=frequency.lower(),
                    adjustflag="1"  # 复权状态(1：后复权， 2：前复权，3：不复权）
                )
                # time.sleep(10)  # API调用后休眠
                if rs.error_code != '0':
                    error_msg = f"API错误 - 错误码: {rs.error_code}, 错误信息: {rs.error_msg}"
                    print(f"[ERROR] {error_msg}")
                    raise Exception(error_msg)

                data_list = []
                while (rs.error_code == '0') & rs.next():
                    data_list.append(rs.get_row_data())

                df = pd.DataFrame(data_list, columns=rs.fields)
                print(f"query_history_k_data_plus：{symbol}接口获取{start_date}-{end_date}日线数据为{len(df)}")

                if not df.empty:
                    if bef_df is not None and not bef_df.empty:
                        df = pd.concat([bef_df, df], ignore_index=True)
                else:
                    df = bef_df

                # 数值类型转换
                numeric_columns = ['open', 'high', 'low', 'close', 'volume', 'amount']
                if frequency.lower() in {'d', 'w', 'm'}:  # 日线、周线、月线额外的指标
                    numeric_columns.extend(['turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM', 'pcfNcfTTM'])

                for col in numeric_columns:
                    if col in df.columns:
                        df[col] = pd.to_numeric(df[col], errors='coerce')

                # 根据时间列排序
                df['factor'] = 1  # 赋权因子
                temp_df = df.drop_duplicates(subset=['date'], keep='last')

                # 市盈率, 市净率, 股息率数据接口
                # print(f"{symbol}-{symbol[3:]}获取市值数据-开始")
                # stock_a_indicator_df = BondDataHandle.get_stock_a_indicator_lg(symbol=symbol)
                # # print(f"{symbol}-{symbol[3:]}获取市值数据-结束")
                # if 'total_mv' in temp_df.columns:
                #     temp_df = temp_df.drop(columns=['total_mv' ], axis=1)
                # temp_df['date'] = pd.to_datetime(temp_df['date'])
                # stock_a_indicator_df['date'] = pd.to_datetime(stock_a_indicator_df['date'])

                # temp_df = pd.merge(temp_df, stock_a_indicator_df, on=['date'], how='left')

                # 格式化日期列
                # temp_df['date'] = pd.to_datetime(temp_df['date'])
                temp_df.sort_values('date', ascending=False).drop_duplicates().to_csv(path_, index=False)
            except Exception as e:
                print(f"{symbol}获取日线数据失败: {e}")
                break

    def convert_data_to_qlib(self):
        '''
        转换可转债数据和正股数据到 qlib_csv
        :return:
        '''

        # DataCons.BOND_ZH_HS_DAILY_PATH
        if os.path.exists(DataCons.BOND_ZH_HS_DAILY_PATH):
            for file in os.listdir(DataCons.BOND_ZH_HS_DAILY_PATH):
                if file.endswith(".csv"):
                    file_path = os.path.join(DataCons.BOND_ZH_HS_DAILY_PATH, file)
                    print(f"转换可转债数据: {file_path}")


        # DataCons.STOCK_TRADING_DAY_PATH
        if os.path.exists(DataCons.STOCK_TRADING_DAY_PATH):
            for file in os.listdir(DataCons.STOCK_TRADING_DAY_PATH):
                if file.endswith(".csv"):
                    file_path = os.path.join(DataCons.STOCK_TRADING_DAY_PATH, file)
                    print(f"转换可转债正股数据: {file_path}")



def main():
    data_handle = BondDataHandle()
    data_handle.down_all_data()
    # data_handle.convert_data_to_qlib()
    # BondDataHandle().get_bond_data()


if __name__ == "__main__":
    main()

