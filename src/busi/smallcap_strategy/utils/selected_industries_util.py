import os
from pathlib import Path

# industry_filter.py

import pandas as pd
import numpy as np

# ---------------- 策略 A ----------------
def strategy_A_recent_top(df, top_k=3):
    df = df.sort_values(['日期', '主力净流入-净额'], ascending=[True, False])
    grouped = df.groupby('日期')
    result = {
        date: group.head(top_k)['行业名称'].tolist()
        for date, group in grouped
    }
    return result

# ---------------- 策略 B ----------------
def strategy_B_recent_topk(df, top_k=3, window=5):
    df = df.copy()
    df = df.sort_values(['日期', '主力净流入-净额'], ascending=[True, False])
    date_list = sorted(df['日期'].unique())
    result = {}
    for i in range(window, len(date_list)):
        win_dates = date_list[i - window:i]
        recent_df = df[df['日期'].isin(win_dates)]
        top_counts = recent_df.groupby('行业名称').head(top_k).groupby('行业名称').size()
        top_industries = top_counts.sort_values(ascending=False).head(top_k).index.tolist()
        result[date_list[i]] = top_industries
    return result

# ---------------- 策略 C ----------------
def strategy_C_today_confirm(df, threshold=1e7):
    df = df.copy()
    df = df[df['主力净流入-净额'] > threshold]
    result = df.groupby('日期')['行业名称'].apply(list).to_dict()
    return result

# ---------------- 策略 D ----------------
def strategy_D_smooth(df, top_k=3, min_appear=3, window=5):
    counts = strategy_B_recent_topk(df, top_k=top_k, window=window)
    result = {}
    for date, inds in counts.items():
        result[date] = [i for i in inds if inds.count(i) >= min_appear]
    return result

# ---------------- 策略 E ----------------
def strategy_E_combined_score(df, price_df, top_k=5, window=5, momentum_days=5):
    df = df.copy()
    df['score'] = 0

    # 获取每个行业近 N 天的资金流均值和动量
    result = {}
    date_list = sorted(df['日期'].unique())
    for i in range(window, len(date_list)):
        win_dates = date_list[i - window:i]
        today = date_list[i]
        sub_df = df[df['日期'].isin(win_dates)]

        avg_flow = sub_df.groupby('行业名称')['主力净流入-净额'].mean()

        mom = {}
        for industry in avg_flow.index:
            price_series = price_df.get(industry)
            if price_series is not None and len(price_series) >= momentum_days + i:
                try:
                    window_prices = price_series.iloc[i - momentum_days:i].values
                    if np.all(~np.isnan(window_prices)):
                        ret = window_prices[-1] / window_prices[0] - 1
                        mom[industry] = ret
                except:
                    continue

        scores = []
        for ind in avg_flow.index:
            flow_score = avg_flow[ind]
            momentum_score = mom.get(ind, 0)
            combo = 0.5 * flow_score + 0.5 * momentum_score * 1e9
            scores.append((ind, combo))

        sorted_ind = sorted(scores, key=lambda x: x[1], reverse=True)[:top_k]
        result[today] = [x[0] for x in sorted_ind]
    return result

# ---------------- 统一封装 ----------------
def get_industry_ranking_by_strategy(df_fundflow, df_price=None, strategy='A', **kwargs):
    if strategy == 'A':
        return strategy_A_recent_top(df_fundflow, **kwargs)
    elif strategy == 'B':
        return strategy_B_recent_topk(df_fundflow, **kwargs)
    elif strategy == 'C':
        return strategy_C_today_confirm(df_fundflow, **kwargs)
    elif strategy == 'D':
        return strategy_D_smooth(df_fundflow, **kwargs)
    elif strategy == 'E':
        return strategy_E_combined_score(df_fundflow, df_price, **kwargs)
    else:
        raise ValueError(f"Unknown strategy: {strategy}")




def strategy_A_today_topk(df_flow: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    records = []
    for date, group in df_flow.groupby('日期'):
        top_industries = (
            group.sort_values(by='主力净流入-净额', ascending=False)
            .head(top_k)['行业名称']
            .tolist()
        )
        records.append({'日期': date, 'Top行业': top_industries})
    return pd.DataFrame(records)


def strategy_A_avg_net_inflow(df_flow: pd.DataFrame, top_k: int = 5, ma_window=5) -> pd.DataFrame:
    """
    🅰️ 策略 A（改进版）：过去 N 日主力净流入均值 TopN，输出格式与 strategy_A_today_topk 相同
    返回 DataFrame:
        - 日期
        - Top行业（行业名称列表）
    """
    df = df_flow.copy()

    # 计算行业滚动平均
    grouped = []
    for name, group in df.groupby('行业名称'):
        group = group.sort_values('日期')
        group[f'{ma_window}日均净流入'] = group['主力净流入-净额'].rolling(window=ma_window, min_periods=1).mean()
        group['行业名称'] = name
        grouped.append(group)

    df_ma = pd.concat(grouped)
    df_ma = df_ma.sort_values(['日期', f'{ma_window}日均净流入'], ascending=[True, False])

    # 按日期取 TopN 行业
    result = []
    for date, group in df_ma.groupby('日期'):
        top_industries = group.head(top_k)['行业名称'].tolist()
        result.append({'日期': date, 'Top行业': top_industries})

    return pd.DataFrame(result)


def strategy_B_recent_topk(df_flow: pd.DataFrame, top_k: int = 5, window: int = 5) -> pd.DataFrame:
    df = df_flow.copy()
    df = df.sort_values(['日期', '主力净流入-净额'], ascending=[True, False])
    date_list = sorted(df['日期'].unique())
    records = []

    for i in range(window, len(date_list)):
        win_dates = date_list[i - window:i]
        recent_df = df[df['日期'].isin(win_dates)]
        top_counts = (
            recent_df.groupby('日期')
            .apply(lambda x: x.head(top_k))
            .reset_index(drop=True)
            .groupby('行业名称').size()
        )
        top_industries = top_counts.sort_values(ascending=False).head(top_k).index.tolist()
        records.append({'日期': date_list[i], 'Top行业': top_industries})

    return pd.DataFrame(records)

def strategy_C_stable_topk(df_flow: pd.DataFrame, top_k: int = 5, window: int = 5) -> pd.DataFrame:
    df = df_flow.sort_values(['日期', '主力净流入-净额'], ascending=[True, False])
    date_list = sorted(df['日期'].unique())
    records = []

    for i in range(window, len(date_list)):
        win_dates = date_list[i - window:i]
        industry_counts = {}

        for date in win_dates:
            top = df[df['日期'] == date].head(top_k)['行业名称'].tolist()
            for name in top:
                industry_counts[name] = industry_counts.get(name, 0) + 1

        # 连续天数 = 上榜次数 == window
        stable_industries = [k for k, v in industry_counts.items() if v == window]
        records.append({'日期': date_list[i], 'Top行业': stable_industries[:top_k]})

    return pd.DataFrame(records)

def strategy_D_rank_avg(df_flow: pd.DataFrame, top_k: int = 5, window: int = 5) -> pd.DataFrame:
    df = df_flow.copy()
    date_list = sorted(df['日期'].unique())
    records = []

    for i in range(window, len(date_list)):
        win_dates = date_list[i - window:i]
        rank_dict = {}

        for date in win_dates:
            day_df = df[df['日期'] == date].sort_values(by='主力净流入-净额', ascending=False)
            for rank, name in enumerate(day_df['行业名称'], start=1):
                rank_dict.setdefault(name, []).append(rank)

        avg_rank = {k: np.mean(v) for k, v in rank_dict.items() if len(v) == window}
        top_industries = sorted(avg_rank, key=avg_rank.get)[:top_k]
        records.append({'日期': date_list[i], 'Top行业': top_industries})

    return pd.DataFrame(records)

def strategy_E_combined_score(
    df_flow: pd.DataFrame,
    df_price: pd.DataFrame,
    top_k: int = 5,
    window: int = 5,
    momentum_days: int = 5
) -> pd.DataFrame:
    from collections import defaultdict
    import numpy as np

    df_flow = df_flow.copy()
    df_price = df_price.copy()
    df_flow = df_flow.sort_values(['日期', '行业名称'])
    df_price = df_price.sort_values(['日期', '行业名称'])

    date_list = sorted(set(df_flow['日期']) & set(df_price['日期']))
    records = []

    for i in range(window, len(date_list)):
        d = date_list[i]
        win_dates = date_list[i - window:i]

        flow_df = df_flow[df_flow['日期'].isin(win_dates)]
        price_window = df_price[df_price['日期'].isin(win_dates + [d])]
        momentum_score = {}

        for name, group in price_window.groupby('行业名称'):
            group = group.sort_values('日期')
            if len(group) < momentum_days + 1:
                continue
            close = group['收盘'].values[-(momentum_days+1):]
            if np.any(np.isnan(close)) or close[0] <= 0:
                continue
            score = (close[-1] - close[0]) / close[0]
            momentum_score[name] = score

        top_counts = (
            flow_df.groupby('日期')
            .apply(lambda x: x.sort_values(by='主力净流入-净额', ascending=False).head(top_k))
            .reset_index(drop=True)
            .groupby('行业名称').size()
        )

        # 加权得分
        total_score = {}
        for name in top_counts.index:
            score = 0.5 * top_counts[name] / window + 0.5 * momentum_score.get(name, 0)
            total_score[name] = score

        top_industries = sorted(total_score.items(), key=lambda x: x[1], reverse=True)
        records.append({
            '日期': d,
            'Top行业': [i[0] for i in top_industries[:top_k]]
        })

    return pd.DataFrame(records)

def strategy_E_preheat_industries(
    df_flow: pd.DataFrame,
    top_k: int = 5,
    window: int = 5,
    rise_days: int = 3,
    max_rank_threshold: int = 30
) -> pd.DataFrame:
    """
    🧠 策略 E++：识别 “预热型” 热门行业（即还没进入主流Top行业，但正在快速爬升的行业）

    📌 目标：
        - 通过识别主力资金净流入排名“持续爬升”的行业，挖掘下一阶段可能成为热点的候选行业。

    📈 原理：
        - 主力净流入排名如果连续几天逐步升高（数字变小），且当前首次进入前30名之内，
          则可能是主力正在布局但还未爆发的行业。

    🧩 参数说明：
        - df_flow : pd.DataFrame
            包含字段 ['日期', '行业名称', '主力净流入-净额'] 的行业资金流入数据
        - top_k : int
            最终输出的“潜力行业”数量（默认输出Top5）
        - window : int
            最近多少天内的数据用于分析（需至少包含 rise_days 的数据）
        - rise_days : int
            连续几天资金流入“排名上升”才认为是持续爬升（如3表示最近3天排名依次上升）
        - max_rank_threshold : int
            最后一天的资金净流入排名必须进入前 max_rank_threshold 内（默认30），
            避免选择太边缘的行业

    📌 返回：
        - DataFrame，每行包含：
            - '日期'：当天日期
            - 'Top行业'：当日识别出的潜在热点行业（最多 top_k 个）

    📊 示例返回结构：
        日期        | Top行业
        ------------|-------------------------------
        2025-07-10 | ['物流行业', '工程机械', '芯片', ...]
    """
    df = df_flow.copy()
    all_dates = sorted(df['日期'].unique())
    industry_ranks = {}

    # 步骤1：计算每个行业每天的主力净流入排名
    for date in all_dates:
        daily_df = df[df['日期'] == date].copy()
        daily_df = daily_df.sort_values('主力净流入-净额', ascending=False)
        daily_df['rank'] = range(1, len(daily_df) + 1)
        for _, row in daily_df.iterrows():
            industry_ranks.setdefault(row['行业名称'], []).append((date, row['rank']))

    selected = []
    for name, rank_series in industry_ranks.items():
        if len(rank_series) < window:
            continue
        last_ranks = [r for _, r in rank_series[-rise_days:]]
        if len(last_ranks) < rise_days:
            continue

        # 步骤2：检查是否持续上升（排名持续变小）
        if all(earlier > later for earlier, later in zip(last_ranks, last_ranks[1:])):
            latest_rank = last_ranks[-1]
            if latest_rank <= max_rank_threshold:
                selected.append(name)

    # 步骤3：构建输出 DataFrame
    today = all_dates[-1]
    return pd.DataFrame([{'日期': today, 'Top行业': selected[:top_k]}])


def strategy_E_combined_with_filter(
    df_flow: pd.DataFrame,
    top_k: int = 5,
    window: int = 5,
    rise_days: int = 3,
    max_rank_threshold: int = 30,
    exclude_recent_top: int = 2,      # 🚫 最近几天的热门行业
    exclude_top_k: int = 5
) -> pd.DataFrame:
    """
    🧠 整合策略 E（预热行业检测）+ 剔除近期爆发行业

    参数说明：
    - df_flow: 主力净流入行业数据 ['日期', '行业名称', '主力净流入-净额']
    - top_k: 每日返回行业个数
    - window: slope 动量与排名窗口
    - rise_days: 连续上涨排名的天数
    - max_rank_threshold: 最后一天行业排名不得高于此阈值（越小越靠前）
    - exclude_recent_top: 向前排除多少天内的高排名行业
    - exclude_top_k: 每天排除多少个高排名行业

    返回：DataFrame，格式 ['日期', 'Top行业']
    """
    import numpy as np
    from sklearn.linear_model import LinearRegression

    df = df_flow.copy()
    df = df.sort_values(['日期', '主力净流入-净额'], ascending=[True, False])
    all_dates = sorted(df['日期'].unique())

    # 1️⃣ 生成行业每日排名
    industry_rank_df = []
    for date in all_dates:
        daily = df[df['日期'] == date].copy()
        daily['rank'] = range(1, len(daily) + 1)
        industry_rank_df.append(daily[['日期', '行业名称', 'rank']])
    rank_df = pd.concat(industry_rank_df)

    # 2️⃣ 行业 → 日期 → 排名序列
    industry_ranks = {}
    for name, group in rank_df.groupby('行业名称'):
        group = group.sort_values('日期')
        industry_ranks[name] = list(zip(group['日期'], group['rank']))

    records = []

    for i in range(window, len(all_dates)):
        date = all_dates[i]
        excluded_set = set()

        # 🚫 过去 exclude_recent_top 天内的 top 排名行业
        recent_dates = all_dates[i - exclude_recent_top:i]
        for d in recent_dates:
            recent_top = rank_df[rank_df['日期'] == d].nsmallest(exclude_top_k, 'rank')
            excluded_set.update(recent_top['行业名称'].tolist())

        selected_strict = []

        # 🔍 连续排名上升的行业
        for name, rank_series in industry_ranks.items():
            if name in excluded_set:
                continue
            series_dict = dict(rank_series)
            sub_dates = all_dates[i - rise_days:i]
            try:
                ranks = [series_dict[d] for d in sub_dates]
            except KeyError:
                continue
            if len(ranks) < rise_days:
                continue
            if all(earlier > later for earlier, later in zip(ranks, ranks[1:])) and ranks[-1] <= max_rank_threshold:
                selected_strict.append((name, ranks[-1]))

        selected_strict = sorted(set(selected_strict), key=lambda x: x[1])
        top_strict_names = [x[0] for x in selected_strict[:top_k]]

        print(date, 'top_strict_names:', top_strict_names)

        # 🧪 不足补充动量 slope
        if len(top_strict_names) < top_k:
            slope_scores = []

            for name, rank_series in industry_ranks.items():
                if name in excluded_set or name in top_strict_names:
                    continue

                series_dict = dict(rank_series)
                sub_dates = all_dates[i - window:i]

                try:
                    ranks = [series_dict[d] for d in sub_dates]
                except KeyError:
                    continue

                if len(ranks) < window:
                    continue

                x = np.arange(len(ranks)).reshape(-1, 1)
                y = np.array(ranks)

                if np.any(np.isnan(y)) or np.any(np.isinf(y)):
                    continue

                # 当前排名不能过热（例如前3），防止“已爆发”
                if y[-1] <= 3:
                    continue

                # 拟合趋势线：判断排名变化趋势
                model = LinearRegression()
                model.fit(x, y)
                slope = model.coef_[0]

                # 限制最大当前排名，防止尾部行业
                if y[-1] <= max_rank_threshold:
                    slope_scores.append((name, slope))

            # 排序：slope 越负越好（排名下降最快）
            slope_scores = sorted(slope_scores, key=lambda x: x[1])

            print(f"[{date}] 📉 slope_scores (排名趋势候选):", slope_scores)

            for name, _ in slope_scores:
                if name not in top_strict_names:
                    top_strict_names.append(name)
                if len(top_strict_names) >= top_k:
                    break

        records.append({'日期': date, 'Top行业': top_strict_names})

    return pd.DataFrame(records)
def get_selected_industries_from_methods(df_flow: pd.DataFrame, df_price: pd.DataFrame,
                                          top_k=5, window=5, momentum_days=5) -> dict:
    """
    返回包含 A/B/C/D/E 每日行业选择结果 以及融合策略结果
    """
    print("📦 正在批量计算 A-E 行业轮动策略...")

    # 分别计算各策略 DataFrame
    res_A = strategy_A_today_topk(df_flow, top_k=top_k)
    res_A1 = strategy_A_avg_net_inflow(df_flow, top_k=top_k, ma_window=window)
    res_B = strategy_B_recent_topk(df_flow, top_k=top_k, window=window)
    res_C = strategy_C_stable_topk(df_flow, top_k=top_k, window=window)
    res_D = strategy_D_rank_avg(df_flow, top_k=top_k, window=window)
    res_E = strategy_E_combined_score(df_flow, df_price, top_k=top_k, window=window, momentum_days=momentum_days)
    res_E1 = strategy_E_combined_with_filter(df_flow, top_k=top_k, window=window,
                                             rise_days=2, max_rank_threshold=30,
                                             exclude_recent_top=2, exclude_top_k=3)

    def df_to_map(df):
        return {row['日期'].strftime('%Y-%m-%d'): row['Top行业'] for _, row in df.iterrows()}

    map_A = df_to_map(res_A)
    map_A1 = df_to_map(res_A1)
    map_B = df_to_map(res_B)
    map_C = df_to_map(res_C)
    map_D = df_to_map(res_D)
    map_E = df_to_map(res_E)
    map_E1 = df_to_map(res_E1)

    print(f"✅ 策略完成，共生成 {len(map_A)} 天的候选行业")
    return {
        "A": map_A,
        "A1": map_A1,
        "B": map_B,
        "C": map_C,
        "D": map_D,
        "E": map_E,
        "E1": map_E1,
    }



def load_industry_fundflow(filepath):
    df = pd.read_csv(filepath, encoding='utf-8')
    df['日期'] = pd.to_datetime(df['日期'])
    df.sort_values(by=['日期', '主力净流入-净额'], ascending=[True, False], inplace=True)
    return df


def load_industry_price(filepaths):

    pd_list = []
    for filepath in os.listdir(filepaths):
        df_price = pd.read_csv(os.path.join(filepaths, filepath), encoding='utf-8')
        pd_list.append( df_price )
        df_price['日期'] = pd.to_datetime(df_price['日期'])

    return pd.concat(pd_list, ignore_index=True)



def load_stock_industry_map():
    """
    读取股票行业映射 CSV 文件，生成字典：{code: industry_name}

    参数：
        csv_path: str，CSV 文件路径，要求包含列 'code', 'industry_name'

    返回：
        dict：{标准化股票代码（如 sh.000001）: 行业名称}
    """

    base_data_path = '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data'
    board_industry_dir = Path(base_data_path).parent / 'zh_data/industry/board_industry'

    # 1. 找到行业目录中最新的CSV文件
    files = [f for f in os.listdir(board_industry_dir) if f.endswith('.csv')]
    if not files:
        raise FileNotFoundError(f"⚠️ 行业目录中没有找到CSV文件: {board_industry_dir}")
    files.sort(key=lambda f: os.path.getmtime(os.path.join(board_industry_dir, f)), reverse=True)
    latest_file = os.path.join(board_industry_dir, files[0])
    print(f"📄 使用行业文件: {latest_file}")

    # 2. 读取行业数据
    industry_df = pd.read_csv(latest_file, dtype={'code': str})
    industry_df = industry_df[['code', 'name', 'industry_code', 'industry_name']]

    # 处理股票代码（补足前缀，如 sh.000001）
    def standardize_code(raw_code):
        if raw_code.startswith('6'):
            return 'sh.' + raw_code
        elif raw_code.startswith(('0', '3')):
            return 'sz.' + raw_code
        elif raw_code.startswith('4') or raw_code.startswith('8'):
            return 'bj.' + raw_code
        else:
            return raw_code

    # df['standard_code'] = df['code'].apply(standardize_code)

    # 构建映射
    stock_industry_map = dict(zip(industry_df['code'], industry_df['industry_name']))

    print(f"✅ 成功生成股票行业映射，共 {len(stock_industry_map)} 条")
    return stock_industry_map

def get_indu_data():
    base_price_path = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/industry/industry_price"
    # 加载数据
    price_df = load_industry_price(base_price_path)
    base_path = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/industry"
    # 加载数据
    df = load_industry_fundflow(f'{base_path}/industry_flow.csv')
    result = get_selected_industries_from_methods(df, price_df, top_k=10, window=10, momentum_days=7)

    return result


if __name__ == '__main__':

    base_price_path = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/industry/industry_price"

    # 加载数据
    price_df = load_industry_price(base_price_path)
    base_path = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/industry"

    # 加载数据
    df = load_industry_fundflow(f'{base_path}/industry_flow.csv')

    result = get_selected_industries_from_methods(df, price_df, top_k=5, window=20, momentum_days=7)
    # res_e = strategy_E_combined_score(df, price_df, top_k=5, window=7, momentum_days=7)

    print( result)

