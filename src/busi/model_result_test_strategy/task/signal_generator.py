import numpy as np
from datetime import datetime, timedelta


class SmallCapSignalGenerator:
    def __init__(self, config):
        self.config = config
        self.stock_data = {}     # name -> DataFrame
        self.today = None        # 当前日期

    def load_data(self, stock_data_dict: dict, today: datetime):
        self.today = today
        self.stock_data_date = today
        temp_dict= {}
        for name, df in stock_data_dict.items():
            df_until_today = df[df.index <= today]
            temp_dict[ name ] = df_until_today
            data_index = df_until_today.index.unique().to_list()
            self.stock_data_date = data_index[-1]
        self.stock_data = temp_dict



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

        crash_days = (daily_ret < -0.025).sum()
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
        if (crash_days >= 2 or avg_ret < -0.03) and vol < 0.2:
            # 最近 3 天至少 2 天跌超 2.5%，或者平均跌超 3%。
            # 且波动率较低。
            print("🚨 触发组合小市值指数的趋势熔断机制")
            return True

        return False
    def check_recent_recovery(self, momentum_days=15):
        recovery_scores = []

        for i in range(4):
            day_scores = []
            for name in self.config['smallcap_index']:
                df = self.stock_data.get(name)  # 获取 DataFrame
                if df is None or 'close' not in df.columns:
                    print(f"⚠️ {name} 数据缺失或无 close 列")
                    return False

                # 要求数据长度至少为 momentum_days + i + 1
                if len(df) < momentum_days + i + 1:
                    print(f"⚠️ {name} 数据不足 {momentum_days + i + 1} 行")
                    return False

                # 取对应的价格区间，注意 pandas 的 index 是正向递增的
                end = -i if i != 0 else None
                price_slice = df['close'].iloc[-(momentum_days + i ):end]

                if price_slice.isnull().any():
                    print(f"⚠️ {name} 包含缺失值")
                    return False

                score = get_momentum(price_slice.values, method="log", days=momentum_days)
                day_scores.append(score)

            # 每天所有小市值指数动量均值
            day_scores = [ s*w for s, w in zip(day_scores, self.config['smallcap_weight'])]
            recovery_scores.append(np.mean(day_scores))

        print(f'📊 最近四个动量: {recovery_scores}')
        return (recovery_scores[0] > recovery_scores[1] > recovery_scores[2] > recovery_scores[3]
                or ( recovery_scores[0] > recovery_scores[1] > recovery_scores[2]
                     and recovery_scores[0] > recovery_scores[1] > recovery_scores[3]
                     )
                or ( recovery_scores[0] > recovery_scores[1] > recovery_scores[3]
                     and recovery_scores[0] > recovery_scores[2] > recovery_scores[3]
                     )
                ) , recovery_scores
    def check_momentum_rank(self, top_k=2, momentum_days=15):
        ranks = []
        for name in self.config['smallcap_index'] + self.config['large_indices']:
            df = self.stock_data.get(name)
            if df is None or len(df) < momentum_days + 1:
                print(f"⚠️ {name} 数据缺失或不足 {momentum_days + 1} 行")
                continue
            # prices = df['close'].values[-(momentum_days + 1):]
            prices = df['close'].values[-(momentum_days):]
            # print('get_index_return:', name, prices)
            score = get_momentum(prices, method='log', days=momentum_days)
            momentum_log = get_momentum(prices, method='log', days=momentum_days)
            momentum_slope = get_momentum(prices, method='return', days=momentum_days)
            # 组合方式（例如加权平均）
            combo_score = 0.5 * momentum_log + 0.5 * momentum_slope
            # print('get_index_return:', name, combo_score, momentum_log, momentum_slope)
            ranks.append((name, combo_score))
        # print(ranks)
        combo_scores = [s*w for s, w in zip([ x[1] for x in ranks if x[0] in self.config['smallcap_index']], self.config['smallcap_weight'])]
        # combo_score = np.mean([ x[1] for x in ranks if x[0] in self.config['smallcap_index']] )
        combo_score = np.mean(combo_scores)
        ranks.append(('__smallcap_combo__', combo_score))

        ranks_comp = ranks[len(self.config['smallcap_index']):]
        ranks_comp.sort(key=lambda x: x[1], reverse=True)
        ranks.sort(key=lambda x: x[1], reverse=True)
        in_top_k = '__smallcap_combo__' in [x[0] for x in ranks_comp[:top_k]]
        top_n = [x[0] for x in ranks_comp].index('__smallcap_combo__') + 1
        is_recovering, recovery_scores = self.check_recent_recovery(momentum_days=momentum_days)

        # if not in_top_k and not is_recovering:
        if not in_top_k :
            return False, ranks, ranks_comp, recovery_scores, top_n
        else:
            return True, ranks, ranks_comp, recovery_scores, top_n

    def get_small_mem_return(self, window_size=5, momentum_days=15):
        scores = []
        for name in self.config['smallcap_index']:
            df = self.stock_data.get(name)
            if df is None or len(df) < momentum_days + 1:
                print(f"⚠️ {name} 数据缺失或不足 {momentum_days + 1} 行")
                continue
            # prices = df['close'].values[-(momentum_days + 1):]
            mems = []
            prices = df['close'].values[-(momentum_days+window_size-1):]
            print('get_small_mem_return:', name, prices)
            for i in range(window_size):
                prices1 = prices[i:momentum_days + i]
                # print('get_index_return:', i, name, prices1)
                momentum_log = get_momentum(prices1, method='log', days=momentum_days)
                momentum_slope = get_momentum(prices1, method='return', days=momentum_days)
                # 组合方式（例如加权平均）
                combo_score = 0.5 * momentum_log + 0.5 * momentum_slope
                mems.append(combo_score)
            if len(mems) > 0:
                scores.append(mems)
        print(f'📊 小市值动量get_small_mem_return: {scores} ')
        if len(scores) > 0:
            # return np.mean(scores, axis=0)

            # 转成 numpy 并匹配长度
            arrays = [np.array(a, dtype=float) for a in scores]

            length_set = {len(a) for a in arrays}
            if len(length_set) != 1:
                raise ValueError("所有数组长度必须一致")

            # 加权相加
            weighted_sum = np.zeros_like(arrays[0])
            for arr, w in zip(arrays, self.config['smallcap_weight']):
                weighted_sum += arr * w

            # 求均值（对加权后的 N 组求平均）
            result = weighted_sum / len(scores)
            return result
        return []


    def smallcap_price_change(self, days=3):
        """
        计算小市值组合指数最近 days 天的涨跌幅，返回最小值
        使用 pandas DataFrame 数据计算
        """
        pcts = []

        for name in self.config['smallcap_index']:
            df = self.stock_data.get(name)
            if df is None or len(df) < days + 1:
                print(f"⚠️ {name} 数据缺失或不足 {days + 1} 行")
                continue
            # 取最近 days + 1 天的数据
            recent_df = df.iloc[-(days):]
            print(f"{name}: {recent_df}")
            print(recent_df[['close', 'open']].head())
            # 昨日收盘 vs days 天前开盘
            pct = (recent_df['close'].iloc[-1] - recent_df['open'].iloc[0]) / recent_df['open'].iloc[0]
            pcts.append(pct)
            # 可选打印调试
            print(f"{name}: pct={pct:.4f}, open0={recent_df['open'].iloc[0]}, close_last={recent_df['close'].iloc[-1]}")
        if pcts:
            return np.min(pcts)  # 返回最小跌幅
        return 0

    def filter_candidates(self):
        results = []
        for name, df in self.stock_data.items():
            row = df.iloc[-1]
            print(f"{name} , mv:{row['mv']}, lt_share_rate:{row['lt_share_rate']}, is_st:{row['is_st']}, amount:{row['amount']}, turn:{row['turn']}, profit_y:{row['profit_y']}, roeAvg_y:{row['roeAvg_y']}, profit_ttm_y:{row['profit_ttm_y']}, revenue_y:{row['revenue_y']}")
            # sz.003003 , mv:2320333600.0,
            # lt_share_rate:0.6665704221367135,
            # is_st:0.0, amount:41451396.92, turn:2.7083,
            # profit_y:46398355.85, roeAvg_y:0.040917,
            # profit_ttm_y:50131100.384399995, revenue_y:1416582813.52
            try:
                if (
                    # ['date', 'open', 'high', 'low', 'close',
                        # 'volume', 'amount', 'turn', 'mv', 'is_st', 'profit_ttm_y',
                        # 'profit_y', 'revenue_y', 'roeAvg_y',
                        # 'profit_ttm_q', 'profit_q', 'revenue_single_q', 'roeAvg_q',
                        # 'openinterest', ]
                    row['mv'] > self.config['min_mv']
                    # and row['lt_mv'] > self.config['min_mv']
                    and row['lt_share_rate'] >= 0.85  # 流通市值占比
                    and row['is_st'] == 0
                    and 5 < row['close'] < self.config['hight_price']
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
                    results.append((name, row['lt_mv'], row['mv']))
                    # results.append((name, row['mv']))
                    print(f"✅ {name} 通过过滤")

            except:
                continue
        # results.sort(key=lambda x: x[2], reverse=False)
        results.sort(key=lambda x: (x[2], x[1], id(x[0]) ), reverse=False)
        return [(x[0], x[2]) for x in results[:self.config['hold_count_high']]]

    def generate_signals(self, current_hold=None):
        """
        返回：
            - 是否趋势熔断
            - 是否动量领先
            - 建议买入列表（包含：股票名、市值、是否已持仓、收盘价）
            - 建议卖出列表（为当前持仓列表）
        """

        score = self.get_small_mem_return(window_size=6, momentum_days=3)
        slope = get_momentum(score[1:], method='slope', days=5)
        print(f"get_small_mem_return score: {score}, slope: {slope}")

        # trend_crash = self.check_trend_crash()
        trend_crash = self.check_combo_trend_crash()
        momentum_ok, momentum_rank, ranks_comp, recovery_scores, top_n = self.check_momentum_rank(top_k=1, momentum_days=self.config['momentum_days'])
        momentum_ok2, _ ,_, _, _= self.check_momentum_rank(top_k=2, momentum_days=self.config['momentum_days'])
        momentum_ok2_short, _ ,_, _, _= self.check_momentum_rank(top_k=2, momentum_days=self.config['momentum_days_short'])

        pct_1 = self.smallcap_price_change(days=1)
        pct_2 = self.smallcap_price_change(days=2)

        candidates = self.filter_candidates()

        # ➕ 添加收盘价字段
        to_buy = []
        for name, mv in candidates:
            df = self.stock_data.get(name)
            if df is None or df.empty or 'close' not in df.columns:
                close_price = None
            else:
                close_price = df['close'].iloc[-1]  # 最新收盘价
            in_hold = 1 if current_hold and name in current_hold else 0
            to_buy.append((name, mv, in_hold, close_price,  False))

        sing = {
            'trend_crash': trend_crash,
            'recovery_scores': recovery_scores,
            'momentum_ok': momentum_ok,
            'momentum_ok2': momentum_ok2,
            'momentum_ok2_short': momentum_ok2_short,
            'small_pct_1': pct_1,
            'small_pct_2': pct_2,
            'slope': slope,
            'top_n': top_n,
            'momentum_rank': [list(t) for t in momentum_rank],
            'ranks_comp': [list(t) for t in ranks_comp],
            'buy': [list(t) for t in to_buy],
            'current_hold': list(current_hold or []),
            'sell': list(current_hold or []),
        }
        print(f"🚀 策略信号：{sing}")
        return  sing

