"""
基于 AKShare：批量计算北向/沪深港通对股票的近10日/20日净流入及占比
说明：
- 需要 pip install akshare pandas openpyxl tqdm
- ak.stock_hsgt_individual_detail_em 的 symbol 参数是单只股票代码（例如 "600519" 或 "002008"）
- 接口只能返回最近 ~90 个交易日内的数据，请注意 start_date 不可太早
"""

import akshare as ak
import pandas as pd
import time
from datetime import datetime, timedelta
from tqdm import tqdm

def safe_sum_last_n(series, n):
    """对 series 的最后 n 个非空值求和（若不足 n 行则求已有的）"""
    s = series.dropna()
    if len(s) == 0:
        return 0.0
    return s.tail(n).sum()

def get_flow_ratios_for_codes(codes,
                              end_date=None,
                              lookback_calendar_days=180,
                              days_short=10,
                              days_long=20,
                              consecutive_days=5,
                              sleep_per_request=0.4):
    """
    codes: list of stock codes, e.g. ["600519","002008"]
    end_date: str "YYYYMMDD" or None (默认今天)
    lookback_calendar_days: 向前取多少日历日的数据以确保包含足够交易日
    days_short/days_long: 计算近N交易日净流入占比（10日/20日）
    consecutive_days: 判断是否连续 N 日净买入（使用持股市值变化-1日 字段）
    返回: DataFrame
    """
    if end_date is None:
        end_dt = datetime.today()
    else:
        end_dt = datetime.strptime(end_date, "%Y%m%d")
    start_dt = end_dt - timedelta(days=lookback_calendar_days)
    start_date_str = start_dt.strftime("%Y%m%d")
    end_date_str = end_dt.strftime("%Y%m%d")
    print(f"抓取区间：{start_date_str} -> {end_date_str}（注意：akshare 的持股详情接口最多能返回最近 ~90 个交易日）")

    rows = []
    for code in tqdm(codes, desc="处理股票"):
        try:
            # 1) 北向/沪深港通个股持股明细（返回多日记录）
            hsgt_df = ak.stock_hsgt_individual_detail_em(
                symbol=str(code),
                start_date=start_date_str,
                end_date=end_date_str
            )

            print("✅ 获取数据：{}".format(code))
            print(hsgt_df.head(10))
            time.sleep(sleep_per_request)  # 切忌太频繁请求

            if hsgt_df is None or hsgt_df.empty:
                # 无数据跳过
                rows.append({
                    "代码": code,
                    "名称": None,
                    "近{}日净流入(元)".format(days_short): None,
                    "近{}日流入占比".format(days_short): None,
                    "近{}日净流入(元)".format(days_long): None,
                    "近{}日流入占比".format(days_long): None,
                    "是否连续{}日净买入".format(consecutive_days): False,
                    "备注": "hsgt 无数据"
                })
                continue

            # 常见列名检查（不同 akshare 版本列名可能略有差异）
            # 期望列示例：['持股日期','当日收盘价','机构名称','持股市值','持股市值变化-1日','持股市值变化-5日','持股市值变化-10日',...]
            # 先把日期列标准化
            date_col_candidates = [c for c in hsgt_df.columns if '持股日期' in c or '日期' == c or '交易日期' in c]
            if len(date_col_candidates) > 0:
                date_col = date_col_candidates[0]
            else:
                # 退而求其次：第一列作为日期
                date_col = hsgt_df.columns[0]

            hsgt_df[date_col] = pd.to_datetime(hsgt_df[date_col])
            hsgt_df = hsgt_df.sort_values(by=date_col).reset_index(drop=True)

            # 名称列（尝试检测）
            name_col = None
            for cand in ["机构名称","机构名称/股东名称",'股票名称','股票简称']:
                if cand in hsgt_df.columns:
                    name_col = cand
                    break

            # 日度净流入列（1日变化）
            change_1d_col = None
            for cand in hsgt_df.columns:
                if "持股市值变化-1" in str(cand) or "持股市值变化-1日" in str(cand) or "持股市值变动-1日" in str(cand):
                    change_1d_col = cand
                    break
            # 10日直观字段（某些版本存在）
            change_10d_col = None
            for cand in hsgt_df.columns:
                if "持股市值变化-10" in str(cand) or "持股市值变化-10日" in str(cand):
                    change_10d_col = cand
                    break

            # If name of the stock itself isn't in this table, try to extract from a different place:
            stock_name = None
            # some versions include '证券简称' or '股票简称' etc
            for cand in ["证券简称","股票简称","股票名称","名称"]:
                if cand in hsgt_df.columns:
                    stock_name = hsgt_df[cand].iloc[-1]
                    break
            # fallback: take first non-null from any string column
            if stock_name is None:
                possible_name_cols = [c for c in hsgt_df.columns if hsgt_df[c].dtype == object]
                if possible_name_cols:
                    stock_name = hsgt_df[possible_name_cols[0]].dropna().astype(str).iloc[0]

            # 计算近10日/20日北向净流入（元）
            # 优先使用直接的 '持股市值变化-10日'（若存在），否则用日度变化累加
            if change_10d_col is not None:
                net_in_10 = float(hsgt_df[change_10d_col].iloc[-1])
            elif change_1d_col is not None:
                net_in_10 = float(safe_sum_last_n(hsgt_df[change_1d_col], days_short))
            else:
                net_in_10 = None

            if change_1d_col is not None:
                net_in_20 = float(safe_sum_last_n(hsgt_df[change_1d_col], days_long))
            else:
                # 如果连1日字段都没有，就无法算 20 日净流入
                net_in_20 = None

            # 是否连续 N 日净买入（使用持股市值变化-1日列）
            is_consecutive = False
            if change_1d_col is not None:
                last_k = hsgt_df[change_1d_col].dropna().tail(consecutive_days)
                if len(last_k) == consecutive_days and (last_k > 0).all():
                    is_consecutive = True

            # 2) 获取成交额（用于分母）
            try:
                hist = ak.stock_zh_a_hist(symbol=str(code),
                                          period="daily",
                                          start_date=start_date_str,
                                          end_date=end_date_str,
                                          adjust="")  # 不复权就行，成交额不受复权影响
                time.sleep(sleep_per_request)
            except Exception as e:
                hist = pd.DataFrame()

            total_amt_10 = None
            total_amt_20 = None
            if hist is not None and (not hist.empty):
                # 确认成交额列名（不同版本列名可能为 '成交额'）
                amt_col = None
                for cand in hist.columns:
                    if "成交额" in str(cand):
                        amt_col = cand
                        break
                if amt_col is None:
                    # 退一步：找 numeric 列作为最后手段
                    numeric_cols = hist.select_dtypes(include=["number"]).columns.tolist()
                    if '成交额' in numeric_cols:
                        amt_col = '成交额'
                if amt_col is not None:
                    total_amt_10 = float(safe_sum_last_n(hist[amt_col], days_short))
                    total_amt_20 = float(safe_sum_last_n(hist[amt_col], days_long))
                else:
                    # 无成交额列
                    total_amt_10 = None
                    total_amt_20 = None

            # 计算占比（防止除以0）
            ratio_10 = None
            ratio_20 = None
            if net_in_10 is not None and total_amt_10 not in (None, 0):
                ratio_10 = net_in_10 / total_amt_10
            if net_in_20 is not None and total_amt_20 not in (None, 0):
                ratio_20 = net_in_20 / total_amt_20

            rows.append({
                "代码": code,
                "名称": stock_name,
                f"近{days_short}日净流入(元)": net_in_10,
                f"近{days_short}日流入占比": ratio_10,
                f"近{days_long}日净流入(元)": net_in_20,
                f"近{days_long}日流入占比": ratio_20,
                f"是否连续{consecutive_days}日净买入": is_consecutive,
                "备注": None
            })

        except Exception as err:
            rows.append({
                "代码": code,
                "名称": None,
                f"近{days_short}日净流入(元)": None,
                f"近{days_short}日流入占比": None,
                f"近{days_long}日净流入(元)": None,
                f"近{days_long}日流入占比": None,
                f"是否连续{consecutive_days}日净买入": False,
                "备注": f"异常: {err}"
            })
            # 轻暂停，避免被封
            time.sleep(1.0)

    result_df = pd.DataFrame(rows)
    # 排序：按 近20日流入占比 降序（NA 放后面）
    result_df = result_df.sort_values(by=f"近{days_long}日流入占比", ascending=False, na_position='last').reset_index(drop=True)
    return result_df


if __name__ == "__main__":
    # 示例：若你有候选池 codes 列表，把它读进来即可
    sample_codes = ["600519", "002008", "300750"]  # 示例代码
    out_df = get_flow_ratios_for_codes(sample_codes, end_date=None, lookback_calendar_days=120)
    out_df.to_excel("北向资金_10_20日净流入占比.xlsx", index=False)
    print(out_df.head())
