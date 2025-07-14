import numpy as np
from datetime import datetime, timedelta
from .utils.momentum_utils import get_momentum
from ..utils.filter_stocks_by_forecast import filter_stocks_by_forecast


class SmallCapSignalGenerator:
    def __init__(self, config):
        self.config = config
        self.stock_data = {}     # name -> DataFrame
        self.today = None        # 当前日期

    def load_data(self, stock_data_dict: dict, today: datetime):
        self.stock_data = stock_data_dict
        self.today = today

    def check_trend_crash(self):
        df = self.stock_data[self.config['smallcap_index'][0]]
        recent = df.tail(4)
        if len(recent) < 4:
            return False
        daily_ret = recent['close'] / recent['open'] - 1
        crash_days = (daily_ret < -0.03).sum()
        avg_ret = daily_ret.mean()
        vol = np.std(np.diff(np.log(df['close'].tail(11)))) * np.sqrt(252)
        return (crash_days >= 2 or avg_ret < -0.04) and vol < 0.2

    def check_combo_trend_crash(self):
        """
        使用多个小市值指数 DataFrame 组合判断趋势熔断：
        - 若过去3天内有至少2天下跌超3%
        - 或平均涨跌幅小于 -4%
        - 且波动率较低（<20%）
        则触发组合趋势止损。
        """
        close_list = []
        open_list = []

        for name in self.config['smallcap_index']:
            df = self.stock_data.get(name)
            if df is None or len(df) < 4 or df[['open', 'close']].isnull().tail(4).any().any():
                print(f"⚠️ 指数 {name} 数据缺失或不足")
                return False
            recent = df.tail(4)
            close_list.append(recent['close'].values)
            open_list.append(recent['open'].values)

        # 构造组合收盘价/开盘价序列
        close_avg = np.mean(close_list, axis=0)
        open_avg = np.mean(open_list, axis=0)
        daily_ret = close_avg / open_avg - 1

        crash_days = (daily_ret < -0.03).sum()
        avg_ret = daily_ret.mean()

        # 波动率计算使用组合指数的最近11个收盘价
        close_series = np.mean(
            [self.stock_data[name]['close'].tail(11).values for name in self.config['smallcap_index']],
            axis=0
        )
        if np.any(np.isnan(close_series)) or len(close_series) < 11:
            print("⚠️ 波动率计算数据缺失")
            return False

        vol = np.std(np.diff(np.log(close_series))) * np.sqrt(252)

        print(f"📉 组合趋势止损判断：3日组合涨跌={daily_ret}, avg={avg_ret:.2%}, vol={vol:.2%}")
        if (crash_days >= 2 or avg_ret < -0.04) and vol < 0.2:
            print("🚨 触发组合小市值指数的趋势熔断机制")
            return True

        return False
    def check_recent_recovery(self):
        recovery_scores = []

        for i in range(3):
            day_scores = []
            for name in self.config['smallcap_index']:
                df = self.stock_data.get(name)  # 获取 DataFrame
                if df is None or 'close' not in df.columns:
                    print(f"⚠️ {name} 数据缺失或无 close 列")
                    return False

                # 要求数据长度至少为 momentum_days + i + 1
                if len(df) < self.config['momentum_days'] + i + 1:
                    print(f"⚠️ {name} 数据不足 {self.config['momentum_days'] + i + 1} 行")
                    return False

                # 取对应的价格区间，注意 pandas 的 index 是正向递增的
                end = -i if i != 0 else None
                price_slice = df['close'].iloc[-(self.config['momentum_days'] + i + 1):end]

                if price_slice.isnull().any():
                    print(f"⚠️ {name} 包含缺失值")
                    return False

                score = get_momentum(price_slice.values, method="log", days=self.config['momentum_days'])
                day_scores.append(score)

            # 每天所有小市值指数动量均值
            recovery_scores.append(np.mean(day_scores))

        print(f'📊 最近三个动量: {recovery_scores}')
        return recovery_scores[2] > recovery_scores[1] > recovery_scores[0]
    def check_momentum_rank(self, top_k=2):
        ranks = []
        for name in self.config['smallcap_index'] + self.config['large_indices']:
            df = self.stock_data.get(name)
            if df is None or len(df) < self.config['momentum_days'] + 1:
                print(f"⚠️ {name} 数据缺失或不足 {self.config['momentum_days'] + 1} 行")
                continue
            prices = df['close'].values[-(self.config['momentum_days'] + 1):]
            score = get_momentum(prices, method='log', days=self.config['momentum_days'])
            momentum_log = get_momentum(prices, method='log', days=self.config['momentum_days'])
            momentum_slope = get_momentum(prices, method='slope_r2', days=self.config['momentum_days'])
            # 组合方式（例如加权平均）
            combo_score = 0.5 * momentum_log + 0.5 * momentum_slope
            ranks.append((name, combo_score))

        combo_score = np.mean([ x[1] for x in ranks if x[0] in self.config['smallcap_index']] )
        ranks.append(('__smallcap_combo__', combo_score))

        ranks_comp = ranks[len(self.config['smallcap_index']):]
        ranks_comp.sort(key=lambda x: x[1], reverse=True)
        ranks.sort(key=lambda x: x[1], reverse=True)
        in_top_k = '__smallcap_combo__' in [x[0] for x in ranks_comp[:top_k]]
        is_recovering = self.check_recent_recovery()

        if not in_top_k and not is_recovering:
            return False, ranks
        else:
            return True, ranks

    def filter_candidates(self):
        results = []
        for name, df in self.stock_data.items():
            row = df.iloc[-1]

            try:
                if (
                    # ['date', 'open', 'high', 'low', 'close',
                        # 'volume', 'amount', 'turn', 'mv', 'is_st', 'profit_ttm_y',
                        # 'profit_y', 'revenue_y', 'roeAvg_y',
                        # 'profit_ttm_q', 'profit_q', 'revenue_single_q', 'roeAvg_q',
                        # 'openinterest', ]
                    # row['mv'] > self.config['min_mv']
                    row['lt_mv'] > self.config['min_mv']
                    and row['lt_share_rate'] >= 0.8
                    and row['is_st'] == 0
                    and 2 < row['close'] < self.config['hight_price']
                    and row['amount'] > 4000000
                    and row['turn'] > 1.5

                    and row['profit_y'] > 0
                    and row['roeAvg_y'] > 0
                    and row['profit_ttm_y'] > 0
                    and row['revenue_y'] > self.config['min_revenue']

                    # and row['profit_q'] > 0
                    # and row['roeAvg_q'] > 0
                    # and row['profit_ttm_q'] > 0
                    # and row['revenue_single_q'] > self.config['min_revenue']

                ):
                    results.append((name, row['lt_mv']))
                    # results.append((name, row['mv']))
            except:
                continue
        results.sort(key=lambda x: x[1])
        return [(x[0], x[1]) for x in results[:self.config['hold_count_high']]]

    def generate_signals(self, current_hold=None):
        """
        返回：
            - 是否趋势熔断
            - 是否动量领先
            - 建议买入列表（包含：股票名、市值、是否已持仓、收盘价）
            - 建议卖出列表（为当前持仓列表）
        """
        # trend_crash = self.check_trend_crash()
        trend_crash = self.check_combo_trend_crash()
        momentum_ok, momentum_rank = self.check_momentum_rank(top_k=2)

        candidates = self.filter_candidates()

        filter_names = filter_stocks_by_forecast([name for name, mv in candidates])
        print(f"🔍 筛选股票：{filter_names}")

        # ➕ 添加收盘价字段
        to_buy = []
        for name, mv in candidates:
            df = self.stock_data.get(name)
            if df is None or df.empty or 'close' not in df.columns:
                close_price = None
            else:
                close_price = df['close'].iloc[-1]  # 最新收盘价
            in_hold = 1 if current_hold and name in current_hold else 0
            to_buy.append((name, mv, in_hold, close_price,  name in filter_names))

        if trend_crash:
            return {
                'trend_crash': True,
                'momentum_ok': momentum_ok,
                'momentum_rank': momentum_rank,
                'buy': to_buy,
                'current_hold': list(current_hold or []),
                'sell': list(current_hold or []),
            }

        if not momentum_ok:
            return {
                'trend_crash': False,
                'momentum_ok': False,
                'momentum_rank': momentum_rank,
                'buy': to_buy,
                'current_hold': list(current_hold or []),
                'sell': list(current_hold or []),
            }


        return {
            'trend_crash': False,
            'momentum_ok': True,
            'momentum_rank': momentum_rank,
            'buy': to_buy,  # [(name, mv, in_hold, close_price)]
            'current_hold': list(current_hold or []),
        }