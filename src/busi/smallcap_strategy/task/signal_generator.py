import numpy as np
from datetime import datetime, timedelta
from utils.momentum_utils import get_momentum


class SmallCapSignalGenerator:
    def __init__(self, config):
        self.config = config
        self.stock_data = {}     # name -> DataFrame
        self.today = None        # 当前日期

    def load_data(self, stock_data_dict: dict, today: datetime):
        self.stock_data = stock_data_dict
        self.today = today

    def check_trend_crash(self):
        df = self.stock_data[self.config['smallcap_index']]
        recent = df.tail(4)
        if len(recent) < 4:
            return False
        daily_ret = recent['close'] / recent['open'] - 1
        crash_days = (daily_ret < -0.03).sum()
        avg_ret = daily_ret.mean()
        vol = np.std(np.diff(np.log(df['close'].tail(11)))) * np.sqrt(252)
        return (crash_days >= 2 or avg_ret < -0.04) and vol < 0.2

    def check_momentum_rank(self, top_k=2):
        ranks = []
        for name in [self.config['smallcap_index']] + self.config['large_indices']:
            df = self.stock_data.get(name)
            if df is None or len(df) < self.config['momentum_days'] + 1:
                continue
            prices = df['close'].values[-(self.config['momentum_days'] + 1):]
            score = get_momentum(prices, method='log', days=self.config['momentum_days'])
            ranks.append((name, score))
        ranks.sort(key=lambda x: x[1], reverse=True)
        return self.config['smallcap_index'] in [x[0] for x in ranks[:top_k]], ranks

    def filter_candidates(self):
        results = []
        for name, df in self.stock_data.items():
            row = df.iloc[-1]
            try:
                if (
                    row['mv'] > self.config['min_mv']
                    and row['profit'] > 0
                    and 2 < row['close'] < self.config['hight_price']
                    and row['amount'] > 4000000
                    and row['turn'] >=1
                    and row['roeAvg'] > 0
                    and row['profit_ttm'] > 0
                    and row['revenue'] > self.config['min_revenue']
                    and row['is_st'] == 0
                ):
                    results.append((name, row['mv']))
            except:
                continue
        results.sort(key=lambda x: x[1])
        return [(x[0], x[1]) for x in results[:self.config['hold_count_high']]]

    def generate_signals(self, current_hold=None):
        """
        返回：
            - 是否趋势熔断
            - 是否动量领先
            - 建议买入列表
            - 建议卖出列表
        """
        trend_crash = self.check_trend_crash()
        momentum_ok, momentum_rank = self.check_momentum_rank(top_k=2)

        if trend_crash:
            return {
                'trend_crash': True,
                'momentum_ok': momentum_ok,
                'momentum_rank': momentum_rank,
                'buy': [],
                'current_hold': list(current_hold or []),
                'sell': list(current_hold or []),
            }

        if not momentum_ok:
            return {
                'trend_crash': False,
                'momentum_ok': False,
                'momentum_rank': momentum_rank,
                'buy': [],
                'current_hold': list(current_hold or []),
                'sell': list(current_hold or []),
            }

        candidates = self.filter_candidates()
        to_buy = [(x[0], x[1], 1 if x[0] in current_hold else 0) for x in candidates]
        return {
            'trend_crash': False,
            'momentum_ok': True,
            'momentum_rank': momentum_rank,
            'buy': list(to_buy),
            'current_hold': current_hold,
        }