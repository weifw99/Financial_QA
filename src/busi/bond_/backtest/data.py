import backtrader as bt
import pandas as pd
from busi.bond_.bond_data import BondDataHandle


#获取数据
class Getdata():

    def dailydata(self):
        # data=pd.read_csv('可转债日线数据带溢价率.txt')
        data = BondDataHandle().get_bond_data(refresh=False)

        # 数据选择
        # date,纯债价值,转股价值,纯债溢价率,转股溢价率,转债代码_code,open,high,low,close,volume,发行规模,上市时间,信用评级,正股代码_code,双低1,双低2
        cols = ['date', '纯债价值', '转股价值', '纯债溢价率', '转股溢价率',
                '转债代码_code', 'open', 'high', 'low', 'close', 'volume',
                'momentum_5', 'pivot', 'bBreak', 'bEnter',
                '发行规模', '上市时间', '信用评级', '正股代码_code', '双低1', '双低2', 'public_date', '']
        # print(data.columns)
        data = data[cols]

        rename_cols = {
            '转债代码_code': 'symbol',
            '双低1': 'double_low1',
            '双低2': 'double_low2',
        }
        data.rename(columns=rename_cols, inplace=True)

        data['openinterest'] = 0

        # 添加索引
        data.index = pd.to_datetime(data['date'])
        # 添加排序

        data = data[['open', 'high', 'low', 'close', 'volume', 'openinterest',
                     'symbol', 'public_date', 'momentum_5', 'pivot', 'bBreak', 'bEnter', 'double_low1', 'double_low2']]
        return data


#拓展数据
class Dailydataextend(bt.feeds.PandasData):
    # 增加线
    lines = ('public_date', 'momentum_5','bBreak','bEnter','double_low1', 'double_low2' )
    params = (('public_date', -1),
              ('momentum_5', -1),
              ('bBreak', -1),
              ('bEnter', -1),
              ('double_low1', -1),
              ('double_low2', -1),
              ('dtformat', '%Y-%m-%d'),)

