
## 将zh_data中的数据转换为qlib的格式
# zh_data 目录结构 data/zh_data/    
# ├── market/
# ├── raw/
#   ├── index/ # 指数数据
#   ├── stock_list.csv # 股票列表


# qlib_data 目录结构 data/qlib_data/
# 每个股票保存为一个csv文件。其中日期列命名为'date'。文件名为股票代码，如股票600000的价格数据，保存在'SH600000.csv'

import os
import pandas as pd
import argparse

from src.data.qlib_data.scripts.dump_bin import DumpDataAll


class ZhData2Qlib:
    def __init__(self, zh_data_dir: str, qlib_data_dir: str, type: str):
        """_summary_

        Args:
            zh_data_dir (str): _description_
            qlib_data_dir (str): _description_
            type (str): daily, min5, min15, min30, min60, monthly, weekly
        """
        self.zh_data_dir = zh_data_dir
        self.qlib_data_dir = qlib_data_dir
        self.type = type

    def convert_to_qlib(self):

        for i, stock_file in enumerate(os.listdir(self.zh_data_dir)):
            # if i > 10:
            #     break
            print(f'{i}/{stock_file}')
            
            file_path = f'{self.zh_data_dir}/{stock_file}/{self.type}.csv'
            if not os.path.exists(file_path):
                print(f'{file_path} not exists')
                continue
            
            qlib_path = self.qlib_data_dir + '/' + self.type
            if not os.path.exists(qlib_path):
               os.makedirs(qlib_path)
            save_path = os.path.join(qlib_path, stock_file.upper().replace('.', '') + '.csv')
            df = pd.read_csv(file_path)
            # 使用后复权价格，factor均设置为1， 回测使用该因子
            #open,high,low,close,volume,amount,turn
            df = df[['date',  'open', 'high', 'low', 'close','volume', 'amount']]
            df['factor'] = 1.0
            df.to_csv(save_path, index=False)
            

if __name__ == '__main__':
    # python zh_data2qlib.py --base_data_path /path/to/base_data --out_base_path /path/to/output
    parser = argparse.ArgumentParser(description="Convert ZhData to Qlib format")

    parser.add_argument('--base_data_path', type=str, default='/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data',
                        help='Base data directory')
    parser.add_argument('--out_base_path', type=str,
                        default='/Users/dabai/liepin/study/llm/Financial_QA/data/qlib_data',
                        help='Output base directory')

    args = parser.parse_args()

    base_data_path = args.base_data_path
    out_base_path = args.out_base_path

    zh_data_dir = base_data_path + '/market'
    zh_data_index_dir = base_data_path + '/raw/index'

    qlib_data_dir = out_base_path + '/cn_data'
    qlib_csv_dir = out_base_path + '/cn_data_csv/'

    # python scripts/dump_bin.py dump_all --csv_path ~/dev/stock_price_data_wind --qlib_dir ~/dev/qlib_data/cn_data_wind
    zh_data2qlib = ZhData2Qlib(zh_data_dir=zh_data_dir, qlib_data_dir=qlib_csv_dir, type='daily')
    zh_data2qlib.convert_to_qlib()
    include_fields = "open,high,low,close,volume,amount,factor"
    DumpDataAll(csv_path=qlib_csv_dir + 'daily',
                qlib_dir=qlib_data_dir,
                freq="day",
                max_workers=4,
                date_field_name="date",
                file_suffix=".csv",
                symbol_field_name="symbol",
                include_fields=include_fields,).dump()
    '''
    
    '''

    '''
    zh_data2qlib = ZhData2Qlib(zh_data_dir=zh_data_dir, qlib_data_dir=qlib_csv_dir, type='min15')
    zh_data2qlib.convert_to_qlib()
    DumpDataAll(csv_path=qlib_csv_dir + 'min15',
                qlib_dir=qlib_data_dir,
                freq="15min",
                max_workers=4,
                date_field_name="datetime",
                file_suffix=".csv",
                symbol_field_name="symbol",
                include_fields=include_fields, ).dump()
                
    '''

    # 更新 instruments 数据
    # 设置指数股票列表

    ins_map = {
        '/raw/index/sz50_constituents.csv': '/instruments/csi50.txt',
        '/raw/index/zz500_constituents.csv': '/instruments/csi500.txt',
        '/raw/index/hs300_constituents.csv': '/instruments/csi300.txt',
        '/raw/index/zz1000_constituents.csv': '/instruments/zz1000.txt',
        '/raw/index/zz2000_constituents.csv': '/instruments/zz2000.txt',
    }
    all_stock_path = '/Users/dabai/liepin/study/llm/Financial_QA/data/qlib_data/cn_data/instruments/all.txt'
    all_stock_df = pd.read_csv(all_stock_path, sep='\t')
    all_stock_df.columns = ['code', 'start_time', 'end_time']

    for k, v in ins_map.items():
        if not os.path.exists(base_data_path + k):
            continue
        data = pd.read_csv(base_data_path + k, sep=',')[['code']]

        data = pd.merge(all_stock_df, data, on='code', how='inner')
        # 过滤不存在的股票
        data.to_csv(qlib_data_dir + v, sep='\t', index=False, header=False)

    ins_map = {
        '/raw/index/中小综指-399101.csv': '/instruments/zxzz399101.txt',
        '/raw/index/微盘股-BK1158.csv': '/instruments/wpgbk1158.txt',
    }

    for k, v in ins_map.items():
        if not os.path.exists(base_data_path + k):
            continue
        data = pd.read_csv(base_data_path + k, sep=',')[['type']]
        data['code'] = data['type'].str.replace('.', '').str.upper()
        data = data[['code']]

        data = pd.merge(all_stock_df, data, on='code', how='inner')
        data.to_csv(qlib_data_dir + v, sep='\t', index=False, header=False)
