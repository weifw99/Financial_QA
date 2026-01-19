import numpy as np
from datetime import datetime, timedelta

import pandas as pd


class SmallCapSignalGenerator:
    def __init__(self, config):
        self.config = config
        self.stock_data = {}     # name -> DataFrame
        self.today = None        # 当前日期

    def load_data(self, stock_data_dict: list[pd.DataFrame], today: datetime):
        self.today = today
        self.stock_data_date = today
        temp_dict= {}
        for df in stock_data_dict:
            print(f"数据长度：{len(df)} 行, 列数: {len(df.columns)}, 数据列: {df.columns.tolist()}")
            if len( df) < 1:
                print(f"⚠️ 无数据")
                continue
            name = df['code'].tolist()[0]
            print(f"数据: {name}")
            df_until_today = df[df.index <= today]
            temp_dict[ name ] = df_until_today
            data_index = df_until_today.index.unique().to_list()
            self.stock_data_date = data_index[-1]
        self.stock_data = temp_dict

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

                    row['lt_share_rate'] >= 0.8  # 流通市值占比
                    and row['mv'] > self.config['min_mv']
                    and row['is_st'] == 0
                    and row['turn'] > 1.5
                    and row['amount'] > 4000000
                    and 2 < row['close'] < self.config['hight_price']

                    and row['profit_y'] > 0
                    and row['roeAvg_y'] > 0
                    and row['profit_ttm_y'] > 0
                    and row['revenue_y'] > self.config['min_revenue']

                    # and row['profit_q'] > 0
                    # and row['roeAvg_q'] > 0
                    and row['profit_ttm_q'] > 0
                    # and row['revenue_single_q'] > self.config['min_revenue']
                    and row['score'] > 0

                ):
                    results.append((name, row['mv'], row['score'], row['class_p']))
                    print(f"✅ {name} 通过过滤")

            except:
                continue
        candidates = sorted(results, key=lambda x: x[2], reverse=True)
        print('candidates:', candidates)
        print("filter_stocks stage 1 len：", len(candidates))
        # candidates1 = candidates[:100]
        # print("filter_stocks stage 2 len：", len(candidates1))
        candidates1 = sorted(candidates, key=lambda x: x[1], reverse=False)

        return candidates1
        # return [(x[0], x[1], x[2]) for x in candidates1[:self.config['hold_count_high']]]

    def generate_signals(self):
        """
        返回：
            - 是否趋势熔断
            - 是否动量领先
            - 建议买入列表（包含：股票名、市值、是否已持仓、收盘价）
            - 建议卖出列表（为当前持仓列表）
        """

        candidates = self.filter_candidates()

        hold_num = self.config['hold_count_high']

        # 拆数据
        stocks = [d for d, mv, score, class_p in candidates]
        mvs = np.array([mv for d, mv, score, class_p in candidates], dtype=float)
        scores = np.array([score for d, mv, score, class_p in candidates], dtype=float)
        class_ps = np.array([score for d, mv, score, class_p in candidates], dtype=float)

        # 横截面 Rank（归一到 [0,1]）
        cap_rank = (-mvs).argsort().argsort() / (len(mvs) - 1)
        score_rank = scores.argsort().argsort() / (len(scores) - 1)
        class_p_rank = class_ps.argsort().argsort() / (len(class_ps) - 1)

        # 加权（建议 0.7~0.8 给市值）
        final_score = 0.9 * cap_rank + 0.06 * score_rank + 0.04 * class_p_rank
        # final_score = 0.95 * cap_rank + 0.05 * score_rank
        # final_score = 0.1 * cap_rank + 0.9 * score_rank
        # final_score = 0.9 * cap_rank + 0.05 * score_rank + 0.05 * class_p_rank

        # 排序选股
        idx = np.argsort(-final_score)
        selected = idx[:hold_num]

        to_hold = list()
        for i in selected:
            d = stocks[i]
            to_hold.append((d, mvs[i], scores[i], class_ps[ i], final_score[i]))
            print(
                f"BUY {d} | final={final_score[i]:.4f} "
                f"mv={mvs[i]:.3f} score={scores[i]:.3f}"
                f"cap_rank={cap_rank[i]:.3f} model_rank={score_rank[i]:.3f}"
            )

        # ➕ 添加收盘价字段
        to_buy = []
        for name, mv, score, class_p, final_score in to_hold:
            df = self.stock_data.get(name)
            if df is None or df.empty or 'close' not in df.columns:
                close_price = None
            else:
                close_price = df['close'].iloc[-1]  # 最新收盘价
            to_buy.append((name, mv, final_score, close_price))

        sing = {
            'buy': [list(t) for t in to_buy],
        }
        print(f"🚀 策略信号：{sing}")
        return sing

