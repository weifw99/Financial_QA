import pandas as pd
import numpy as np
import matplotlib.pyplot as plt


class IndustryFactorResearch:
    def __init__(self, df_price, df_flow):
        """
        输入数据：
        df_price: 包含 收盘、行业代码、日期 等
        df_flow:  包含 主力净流入 等
        """
        self.df = self._prepare_data(df_price, df_flow)

    # -------------------------------
    # 数据准备与合并
    # -------------------------------
    def _prepare_data(self, df_price, df_flow):
        dfp = df_price.copy()
        dff = df_flow.copy()

        dfp['日期'] = pd.to_datetime(dfp['日期'])
        dff['日期'] = pd.to_datetime(dff['日期'])

        df = pd.merge(dfp, dff, on=['日期', '行业代码', '行业名称'], how='inner')
        df = df.sort_values(['行业代码', '日期']).reset_index(drop=True)
        return df

    # -------------------------------
    # Z-score
    # -------------------------------
    @staticmethod
    def zscore(s):
        return (s - s.mean()) / (s.std() + 1e-9)

    # -------------------------------
    # 构建热度指标
    # -------------------------------
    def build_hot_factors(self):
        df = self.df

        # 主力连续净流入天数（5日）
        df['主力连续净流入天数'] = df.groupby('行业代码')['主力净流入-净额'].transform(
            lambda x: x.gt(0).rolling(5).sum()
        )

        # 超大单爆发（相对20日均值）
        df['超大单爆发'] = df.groupby('行业代码')['超大单净流入-净额'].transform(
            lambda x: x / (x.rolling(20).mean() + 1e-9)
        )

        # 主力占比变化
        df['主力占比变化'] = df.groupby('行业代码')['主力净流入-净占比'].transform(lambda x: x.diff())

        # 资金流 RPS（横截面）
        df['flow_RPS'] = df.groupby('日期')['主力净流入-净额'].transform(lambda x: x.rank(pct=True))

        # 综合热度
        df['hot_score'] = (
                0.4 * self.zscore(df['flow_RPS']) +
                0.1 * self.zscore(df['主力连续净流入天数']) +
                0.05 * self.zscore(df['超大单爆发']) +
                0.05 * self.zscore(df['主力占比变化'])
        )

        self.df = df

    # -------------------------------
    # 计算 RPS（给定周期）
    # -------------------------------
    def build_rps(self, N):
        df = self.df

        df[f'return_{N}'] = df.groupby('行业代码')['收盘'].pct_change(N)
        df['RPS'] = df.groupby('日期')[f'return_{N}'].transform(lambda x: x.rank(pct=True))

        self.df = df

    # -------------------------------
    # future return
    # -------------------------------
    def compute_future_ret(self, future_day):
        df = self.df
        df['future_ret'] = df.groupby('行业代码')['收盘'].shift(-future_day) / df['收盘'] - 1
        self.df = df

    # -------------------------------
    # 构建组合因子
    # -------------------------------
    def build_combo_score(self, w_hot=0.6, w_rps=0.4):
        df = self.df

        df['combo_score'] = (
                w_hot * self.zscore(df['hot_score']) +
                w_rps * self.zscore(df['RPS'])
        )

        self.df = df

    # -------------------------------
    # IC 分析
    # -------------------------------
    def calc_ic(self):
        df = self.df
        ic_series = df.groupby('日期', group_keys=False).apply(
            lambda x: x['combo_score'].corr(x['future_ret']), include_groups=False
        )
        return {
            'IC_mean': ic_series.mean(),
            'IC_std': ic_series.std(),
            'IC_IR': ic_series.mean() / (ic_series.std() + 1e-9)
        }

    # -------------------------------
    # 分层收益
    # -------------------------------
    def layer_analysis(self, n_bins=5, labels=None):
        """
        更稳健的分层分析。
        n_bins: 分层数量，默认为 5（Q1..Q5），可以改为 3 等。
        labels: 标签列表，默认 ['Q1',...]
        """
        if labels is None:
            labels = [f"Q{i}" for i in range(1, n_bins + 1)]

        df = self.df.copy()

        # safe qcut for a single group's Series; 返回与 x.index 对齐的 Series（标签）
        def safe_qcut_for_group(x):
            # x 是 Series（可能包含 NaN）
            x = x.copy()
            x_no_na = x.dropna()

            # fallback：用 rank-based 分桶（保证不会报错）
            def fallback_rank():
                # 对原始 index 返回 Series（标签），NaN 保持 NaN
                ranks = x.rank(method='first')  # 保证唯一顺序
                try:
                    lab = pd.qcut(ranks, n_bins, labels=labels)
                except Exception:
                    # 最保险：按百分位手动分配
                    pct = ranks.rank(pct=True)
                    bins = pd.cut(pct, bins=np.linspace(0, 1, n_bins + 1), labels=labels, include_lowest=True)
                    lab = bins
                return pd.Series(lab, index=x.index)

            # 情况1：有效值太少
            if len(x_no_na) < n_bins:
                return fallback_rank()

            # 情况2：有效值全部相同
            if x_no_na.nunique() == 1:
                return fallback_rank()

            # 情况3：NaN 比例过高（可调，避免 qcut 在边界全部 NaN）
            if x.isna().mean() > 0.5:
                return fallback_rank()

            # 正常情况：对无 NaN 的子序列做 qcut，然后 reindex 回原 index
            try:
                lab_no_na = pd.qcut(x_no_na, n_bins, labels=labels, duplicates='drop')
                # lab_no_na 是一个按 x_no_na.index 对应的分类 Series
                out = pd.Series(index=x.index, dtype=object)
                out.loc[lab_no_na.index] = lab_no_na.values
                # 剩下的（NaN 的位置）保持为 NaN
                return out
            except Exception:
                # 一切失败 -> fallback
                return fallback_rank()

        # 对每个日期分组并 apply，结果是一个 Series（索引为原 df 的行索引）
        quant_series = df.groupby('日期')['combo_score'].apply(lambda grp: safe_qcut_for_group(grp)).reset_index(
            level=0, drop=True)

        # groupby.apply 返回时，若每次返回 Series，会得到 MultiIndex；reset_index(level=0, drop=True) 已处理
        # 但有时 pandas 会把结果放成 object dtype 的 Series of Series —— 规范化为单列
        if isinstance(quant_series.iloc[0], pd.Series):
            # 展开：quant_series 是由子 Series 组成的 Series，逐个合并
            quant = pd.concat(quant_series.values)
        else:
            quant = quant_series

        # 确保 quant 的索引与 df 的索引对齐
        quant = quant.reindex(df.index)

        df['quantile'] = quant.values

        # 计算 future_ret（如果还未计算）
        if 'future_ret' not in df.columns:
            # 这里默认 future_day 已在 self.df 中或由外部调用 compute_future_ret 预先计算
            raise ValueError(
                "请先调用 compute_future_ret(future_day) 以生成 future_ret 列，或确保 self.df 已含 'future_ret'")

        # 返回分层统计
        return df.groupby('quantile')['future_ret'].agg(
            样本数='count',
            平均收益='mean',
            胜率=lambda x: (x > 0).mean(),
            上涨幅度=lambda x: x[x > 0].mean(),
            下跌幅度=lambda x: x[x <= 0].mean()
        )

    # -------------------------------
    # 权重搜索（w_hot, w_rps）
    # -------------------------------
    def search_best_weights(self, future_day):
        results = []
        for w_hot in [0.1, 0.3, 0.5, 0.7, 0.9]:
            w_rps = 1 - w_hot

            self.build_combo_score(w_hot, w_rps)
            self.compute_future_ret(future_day)

            ic_res = self.calc_ic()
            results.append([w_hot, w_rps, ic_res['IC_mean']])

        return pd.DataFrame(results, columns=['w_hot', 'w_rps', 'IC']).sort_values('IC', ascending=False)

    import matplotlib.pyplot as plt
    import numpy as np
    import pandas as pd

    def factor_ic_analysis(self, future_day=10):
        """
        拆解因子，计算每个因子对未来收益的 IC（信息系数）
        仅对每日期有效数据 >= 2 计算
        """
        df = self.df.copy()
        if 'future_ret' not in df.columns:
            self.compute_future_ret(future_day)

        factor_list = [
            'hot_score', 'flow_RPS', '主力连续净流入天数',
            '超大单爆发', '主力占比变化', 'RPS', '板块资金聚集度', '主力资金动量', '超大单占比',
            '超大单净流入-净额_趋势', '大单净流入-净额_趋势', '中单净流入-净额_趋势', '小单净流入-净额_趋势',
            '主力连续净流入天数_3', '主力连续净流入天数_5','主力连续净流入天数_7', '主力连续净流入天数_10',
            '主力连续净流入天数_15', '主力连续净流入天数_20','主力连续净流入天数_30', '主力连续净流入天数_40',
            '主力连续净流入天数_60',
            # [3, 5, 7, 10, 15, 20, 30, 40, 60, ]
        ]

        results = []
        for fac in factor_list:
            ic_vals = []
            for date, group in df.groupby('日期'):
                if group[fac].notna().sum() >= 2 and group['future_ret'].notna().sum() >= 2:
                    ic = group[fac].corr(group['future_ret'])
                else:
                    ic = np.nan
                ic_vals.append(ic)
            ic_vals = np.array(ic_vals)
            results.append([fac, np.nanmean(ic_vals), np.nanstd(ic_vals)])
        return pd.DataFrame(results, columns=['因子', 'IC均值', 'IC_std']).sort_values('IC均值', ascending=False)

    # -------------------------------
    # RPS 周期搜索
    # -------------------------------
    def search_best_rps(self, future_day):
        results = []
        for N in [20, 40, 55, 80, 120]:
            self.build_rps(N)
            self.build_combo_score(0.6, 0.4)
            self.compute_future_ret(future_day)

            ic_res = self.calc_ic()
            layer = self.layer_analysis()
            q_diff = layer.loc['Q5', '平均收益'] - layer.loc['Q1', '平均收益']

            results.append([N, ic_res['IC_mean'], q_diff])

        return pd.DataFrame(results, columns=['RPS周期', 'IC', 'Q5-Q1']).sort_values('IC', ascending=False)

    # -------------------------------
    # 持有周期搜索
    # -------------------------------
    def search_best_future_day(self, rps_N):
        results = []
        self.build_rps(rps_N)

        for future_day in [5, 10, 20, 30]:
            self.build_combo_score(0.6, 0.4)
            self.compute_future_ret(future_day)

            ic_res = self.calc_ic()
            layer = self.layer_analysis()
            q_diff = layer.loc['Q5', '平均收益'] - layer.loc['Q1', '平均收益']

            results.append([future_day, ic_res['IC_mean'], q_diff])

        return pd.DataFrame(results, columns=['未来天数', 'IC', 'Q5-Q1']).sort_values('IC', ascending=False)

    # -------------------------------
    # 分层收益可视化
    # -------------------------------
    def plot_layers(self):
        layer = self.layer_analysis()
        plt.figure(figsize=(8, 5))
        layer['平均收益'].plot(kind='bar')
        plt.title("Q1-Q5 分层未来收益")
        plt.show()

    # -------------------------------
    # 资金流衍生特征计算
    # -------------------------------
    def build_advanced_flow_features(self, window_trend=3):
        """
        生成资金流趋势特征，包括：
        - 连续净流入天数（不同窗口）
        - 主力资金动量
        - 超大单占比
        - 板块聚集度（多股同时流入）
        window_trend: 用于计算滚动趋势的窗口
        """
        df = self.df.copy()

        # -------------------
        # 1. 连续净流入天数（可变窗口）
        # -------------------
        df[f'主力连续净流入天数_{window_trend}'] = df.groupby('行业代码')['主力净流入-净额'].transform(
            lambda x: x.gt(0).rolling(window_trend).sum()
        )

        for window in [3, 5, 7, 10, 15, 20, 30, 40, 60, ]:
            df[f'主力连续净流入天数_{window}'] = df.groupby('行业代码')['主力净流入-净额'].transform(
                lambda x: x.gt(0).rolling(window).sum()
            )

        # -------------------
        # 2. 主力资金动量（今日流入 - 昨日流入）
        # -------------------
        df['主力资金动量'] = df.groupby('行业代码')['主力净流入-净额'].transform(lambda x: x.diff())

        # -------------------
        # 3. 超大单占比（相对于成交额）
        # -------------------
        df['超大单占比'] = df['超大单净流入-净额'] / (df['成交额'] + 1e-9)

        # -------------------
        # 4. 板块聚集度（同日多股同时流入）
        # -------------------
        # 每日板块内资金流入股票数 / 总股票数
        df['板块资金聚集度'] = df.groupby(['日期', '行业代码'])['主力净流入-净额'].transform(
            lambda x: (x > 0).sum() / len(x)
        )

        # -------------------
        # 5. 分类型资金流趋势（超大/大/中/小单）
        # -------------------
        for col in ['超大单净流入-净额', '大单净流入-净额', '中单净流入-净额', '小单净流入-净额']:
            df[f'{col}_趋势'] = df.groupby('行业代码')[col].transform(lambda x: x.rolling(window_trend).mean())

        # 更新 df
        self.df = df

    def build_combo_score_advanced(self, w_hot=0.5, w_advanced=0.5):
        """
        使用增强版资金流因子构建 combo_score_advanced：
        - w_hot: 原始 hot_score 权重
        - w_advanced: 新增资金流衍生特征权重
        """
        df = self.df

        # 构建增强版热度
        # df['hot_score_advanced'] = (
        #         0.2 * self.zscore(df['大单净流入-净额_趋势']) +
        #         0.2 * self.zscore(df['超大单净流入-净额_趋势']) +
        #         0.2 * self.zscore(df['flow_RPS']) +
        #         0.1 * self.zscore(df['超大单占比']) +
        #         0.1 * self.zscore(df['主力连续净流入天数_5'])
        # )
        df['hot_score_advanced'] = (
                0.9 * self.zscore(df['大单净流入-净额_趋势']) +
                0.1 * self.zscore(df['超大单净流入-净额_趋势']) +
                0.2 * self.zscore(df['flow_RPS']) +
                0.1 * self.zscore(df['超大单占比']) +
                0.1 * self.zscore(df['主力连续净流入天数_5'])
        )

        # 组合成新的 combo_score_advanced
        df['combo_score_advanced'] = (
                w_hot * self.zscore(df['hot_score']) +
                w_advanced * self.zscore(df['hot_score_advanced'])
        )
        df['combo_score'] = (
                w_hot * self.zscore(df['hot_score']) +
                w_advanced * self.zscore(df['hot_score_advanced'])
        )

        self.df = df

    def get_daily_quantile_details(self, combo_col='combo_score_advanced', n_bins=5):
        """
        按日期对 combo_score_advanced 分层，返回每个分层的详细列表：
        {日期1: {'Q1': [(行业代码, 行业名称, future_ret), ...], ..., 'Q5': [...]}, ...}
        """
        labels = [f"Q{i}" for i in range(1, n_bins + 1)]
        result_dict = {}

        for date, group in self.df.groupby('日期'):
            x = group[combo_col].copy()
            x_no_na = x.dropna()

            # safe qcut
            if len(x_no_na) < n_bins or x_no_na.nunique() == 1:
                ranks = x.rank(method='first', pct=True)
                group['quantile'] = pd.cut(ranks, bins=np.linspace(0, 1, n_bins + 1), labels=labels,
                                           include_lowest=True)
            else:
                group['quantile'] = pd.qcut(x_no_na, n_bins, labels=labels, duplicates='drop')
                group['quantile'] = group['quantile'].reindex(group.index)

            # 为每个分层生成详细列表
            daily_dict = {q: [] for q in labels}
            for q, sub in group.groupby('quantile'):
                if pd.isna(q):
                    continue
                for _, row in sub.iterrows():
                    daily_dict[q].append((row['行业代码'], row['行业名称'], row[combo_col]))

            result_dict[date] = daily_dict

        return result_dict


    def build_code_industry_dict(self, df: pd.DataFrame=None) -> dict:
        """
        根据股票 code 转换前缀，生成 {转换后code: industry_code} 的字典。
        """
        if df is None:
            base_path = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/industry"
            # 加载数据
            import pandas as pd
            df = pd.read_csv(f'{base_path}/board_industry/data.csv', dtype={'code': 'string'})

        def convert_code(code: str) -> str:
            if code.startswith("00"):
                return f"sz.{code}"
            elif code.startswith("399"):
                return f"sz.{code}"
            elif code.startswith("93"):
                return f"csi{code}"
            else:
                return f"sh.{code}"  # 默认用上交所，如果你需要可修改

        result = {}
        for _, row in df.iterrows():
            raw_code = row["code"]
            industry_code = row["industry_code"]
            conv_code = convert_code(raw_code)
            result[conv_code] = industry_code

        return result
