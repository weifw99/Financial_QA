
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

from models.qlib_.data.scripts.dump_bin import DumpDataAll


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
            df['factor'] = 1.0
            df.to_csv(save_path, index=False)
            

if __name__ == '__main__':

    base_data_path = '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data'
    zh_data_dir = base_data_path + '/market'
    zh_data_index_dir = base_data_path + '/raw/index'

    out_base_path = '/Users/dabai/liepin/study/llm/Financial_QA/data/qlib_data'

    qlib_data_dir = out_base_path + '/zh_qlib'



    qlib_csv_dir = out_base_path + '/zh_qlib_csv/'
    zh_data2qlib = ZhData2Qlib(zh_data_dir=zh_data_dir, qlib_data_dir=qlib_csv_dir, type='daily')
    zh_data2qlib.convert_to_qlib()

    zh_data2qlib = ZhData2Qlib(zh_data_dir=zh_data_dir, qlib_data_dir=qlib_csv_dir, type='min15')
    zh_data2qlib.convert_to_qlib()


    # python scripts/dump_bin.py dump_all --csv_path ~/dev/stock_price_data_wind --qlib_dir ~/dev/qlib_data/cn_data_wind

    include_fields = "open,high,low,close,volume,amount,turn,pctChg,factor"
    DumpDataAll(csv_path=qlib_csv_dir + 'daily',
                qlib_dir=qlib_data_dir,
                freq="day",
                max_workers=4,
                date_field_name="date",
                file_suffix=".csv",
                symbol_field_name="symbol",
                include_fields=include_fields,).dump()

    # 设置指数股票列表

    DumpDataAll(csv_path=qlib_csv_dir + 'min15',
                qlib_dir=qlib_data_dir,
                freq="15min",
                max_workers=4,
                date_field_name="datetime",
                file_suffix=".csv",
                symbol_field_name="symbol",
                include_fields=include_fields, ).dump()

    # 更新 instruments 数据

    ins_map = {
        '/raw/index/sz50_constituents.csv': '/zh_qlib/instruments/csi50.txt',
        '/raw/index/zz500_constituents.csv': '/zh_qlib/instruments/csi500.txt',
        '/raw/index/hs300_constituents.csv': '/zh_qlib/instruments/csi300.txt',
    }

    for k, v in ins_map.items():
        data = pd.read_csv(base_data_path + k, sep=',')
        data.to_csv(out_base_path + v, sep='\t', index=False, header=False)
    '''
    
    from qlib.data import D
    import pandas as pd

    # 读取新的 instruments 数据
    data = pd.read_csv('hs300_stocks.csv')

    # 将数据转换为 qlib 所需的格式
    # 这里需要根据具体的数据格式进行转换
    # 假设数据包含 'code' 列表示证券代码
    instruments = data['code'].tolist()

    # 更新 qlib 中的 instruments 数据
    D.update_instruments(instruments=instruments, start_time='2024-01-01', end_time='2024-12-31', freq='day', append=True)
    '''

