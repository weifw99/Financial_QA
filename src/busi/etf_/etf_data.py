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

from busi.etf_.constant import DataCons


class EtfDataHandle:
    def __init__(self):
        pass
        # Path(DataCons.BOND_INFO_DIR).mkdir(parents=True, exist_ok=True)
        # Path(DataCons.BOND_TRADING_DAY_DIR).mkdir(parents=True, exist_ok=True)
        # Path(DataCons.STOCK_TRADING_DIR).mkdir(parents=True, exist_ok=True)
        # self._connct_manage = BaostockConnectionManager()
        # self._connct_manage.login()

    def get_etf_data(self, refresh: bool = True) -> pd.DataFrame:
        import akshare as ak

        fund_etf_spot_em_df = ak.fund_etf_spot_em()

        fund_etf_spot_em_df.to_csv(DataCons.ETF_INFO_FILE_PATH, index=False)
        print(fund_etf_spot_em_df)

    @staticmethod
    def get_and_download_etf_info() -> pd.DataFrame:

        etf_spot_em_df: pd.DataFrame = None
        if os.path.exists(DataCons.ETF_INFO_FILE_PATH):
            print(f"etf数据 从本地文件加载债券数据: {DataCons.ETF_INFO_FILE_PATH}")
            etf_spot_em_df = pd.read_csv(DataCons.ETF_INFO_FILE_PATH)
        else:
            etf_spot_em_df = ak.fund_etf_spot_em()

            # 提取代码
            def get_exchange(code) -> str:
                code = str(code)
                if code.startswith(('51', '58', '56', '50', '52')):
                    return f'SZ{code}' # 上证
                elif code.startswith(('15', '16')):
                    return f'SH{code}' # 深证
                else:
                    return f'WZ{code}' # 未知

            etf_spot_em_df['代码1'] = etf_spot_em_df['代码'].apply(get_exchange)

            etf_spot_em_df.to_csv(DataCons.ETF_INFO_FILE_PATH, index=False)
        print(f"etf数据 数据已保存至: {DataCons.ETF_INFO_FILE_PATH}")

        return etf_spot_em_df


    @staticmethod
    def download_etf_trading_day_data(symbol: Union[str, List[str]],
                                      start_date: str = '19700101',
                                      end_date: Optional[str] = None,
                                      refresh: bool = True) -> list[pd.DataFrame]:
        """
        :param symbol:
        :param start_date: 19700101
        :param end_date: 19700101
        :return:
        """
        if isinstance(symbol, str):
            symbols = [symbol]
        else:
            symbols = symbol
        print(f"从接口获取 etf日频交易 数据: {symbol}")
        frequency = 'daily'
        if end_date is None:
            end_date = datetime.now().strftime('%Y%m%d')

        # 生成日期列表
        #date_list = pd.date_range(start=start_date, end=end_date, freq='D').strftime('%Y%m%d').tolist()

        etf_result = []
        for symbol in symbols:
            path_ = DataCons.ETF_HS_DAILY_FILE_PATH.format(symbol)
            Path(path_).parent.mkdir(parents=True, exist_ok=True)
            bef_df: pd.DataFrame = None
            if refresh and os.path.exists(path_):
                print(f"从本地文件已存在 etf日频交易 数据: {symbol}, 拉取最新的数据，并 merge")
                bef_df = pd.read_csv(path_)
                if bef_df is not None and not bef_df.empty:
                    bef_df = bef_df.drop_duplicates(subset=['date'], keep='last')
                    bef_df = bef_df.sort_values(by='date', ascending=False)
                    start_date = bef_df.iloc[0]['date']
            try:
                import akshare as ak
                df = ak.fund_etf_hist_em(symbol=symbol[2:],
                                         period=frequency,
                                         start_date=start_date,
                                         end_date=end_date,
                                         adjust="hfq")
                if bef_df is not None and not bef_df.empty:
                    df = pd.concat([bef_df, df], ignore_index=True)

                if len(df) == 0:
                    print(f"{symbol}获取日线数据为 null")
                    continue

                time.sleep(0.01)
                df['factor'] = 1  # 赋权因子

                # 日期,开盘,收盘,最高,最低,成交量,成交额,涨跌幅,换手率,factor
                # date,open,close,high,low,volume,amount,pctChg,turn,factor
                temp_df = df.drop_duplicates(subset=['日期'], keep='last')
                columns = {
                    '日期': 'date',
                    '开盘': 'open',
                    '收盘': 'close',
                    '最高': 'high',
                    '最低': 'low',
                    '成交量': 'volume',
                    '成交额': 'amount',
                    '涨跌幅': 'pctChg',
                    '换手率': 'turn',
                    'factor': 'factor'
                }
                temp_df.rename(columns=columns, inplace=True)

                temp_df['date'] = pd.to_datetime(temp_df['date'])
                temp_df = temp_df.sort_values('date', ascending=False)

                temp_df.to_csv(path_, index=False)
                etf_result.append(temp_df)
            except Exception as e:
                print(f"{symbol}获取日线数据失败: {e}")
                break

        return etf_result

    def get_down_symbols_data(self, symbols: List[str], refresh: bool = False) -> pd.DataFrame:

        if (not refresh) and os.path.exists(DataCons.ETF_INFO_CAT):
            return pd.read_csv(DataCons.ETF_INFO_CAT)

        etf_list = []
        for etf_code in symbols:
            temp_etf: list[pd.DataFrame] = self.download_etf_trading_day_data(symbol=str( etf_code ) )

            if len(temp_etf) == 0:
                continue
            temp_etf0 = temp_etf[0]
            temp_etf0['代码'] = etf_code
            etf_list.append(temp_etf0)

        etf_trading_day_pd = pd.concat(etf_list, axis=0)

        etf_trading_day_pd.index = etf_trading_day_pd['日期']
        if not os.path.exists(DataCons.ETF_INFO_DIR):
            os.makedirs(DataCons.ETF_INFO_DIR)
        etf_trading_day_pd.to_csv(DataCons.ETF_INFO_CAT, index=False)

        return etf_trading_day_pd


    def get_down_all_data(self, refresh: bool = False) -> pd.DataFrame:

        if (not refresh) and os.path.exists(DataCons.ETF_INFO_CAT):
            return pd.read_csv(DataCons.ETF_INFO_CAT)

        etf_info_dfs: pd.DataFrame = self.get_and_download_etf_info()

        # etf_info_dfs = etf_info_dfs[[]]
        etf_list = []
        for etf_code in etf_info_dfs['代码1'].tolist():
            temp_etf: list[pd.DataFrame] = self.download_etf_trading_day_data(symbol=str( etf_code ) )

            if len(temp_etf) == 0:
                continue
            temp_etf0 = temp_etf[0]
            temp_etf0['代码'] = etf_code
            etf_list.append(temp_etf0)

            time.sleep(random.randint(1, 6))

        etf_trading_day_pd = pd.concat(etf_list, axis=0)

        etf_trading_day_pd.index = etf_trading_day_pd['date']
        if not os.path.exists(DataCons.ETF_INFO_DIR):
            os.makedirs(DataCons.ETF_INFO_DIR)
        etf_trading_day_pd.to_csv(DataCons.ETF_INFO_CAT, index=False)

        return etf_trading_day_pd




def main():
    # EtfDataHandle().get_etf_data()
    EtfDataHandle().get_down_all_data(refresh=True)


if __name__ == "__main__":
    main()

