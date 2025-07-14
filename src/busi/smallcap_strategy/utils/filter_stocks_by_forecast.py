import os
from datetime import datetime

import akshare as ak
import pandas as pd

NEGATIVE_FORECAST_KEYWORDS = ['预亏', '首亏', '增亏', '续亏', '略减', '减亏']
NEGATIVE_FORECAST_KEYWORDS = ['预亏', '首亏', '增亏', '续亏', '预减', '略减', '减亏', '不确定']
# POSITIVE_FORECAST = ['预增', '扭亏', '续盈']
# NEUTRAL_FORECAST = ['略增', '减亏']  # 可选
def load_earnings_forecast(dates: list) -> pd.DataFrame:
    """
    加载多个季度的业绩预告数据，支持本地缓存。
    :param dates: 预告日期列表，例如 ['20240331', '20240630']
    :param base_path: 本地缓存目录
    :return: 合并后的业绩预告 DataFrame
    """

    base_path = '/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/yjyg/'

    os.makedirs(base_path, exist_ok=True)
    all_dfs = []

    for date in dates:
        file_path = os.path.join(base_path, f'yjyg_{date}.csv')

        # 优先尝试从本地加载
        if os.path.exists(file_path):
            print(f'📂 读取本地文件: {file_path}')
            df = pd.read_csv(file_path, dtype={'股票代码': str})
        else:
            try:
                print(f'🌐 请求接口数据: {date}')
                df = ak.stock_yjyg_em(date=date)
                df.to_csv(file_path, index=False, encoding='utf-8-sig')
                print(f'✅ 已保存到本地: {file_path}')
            except Exception as e:
                print(f'⚠️ 获取接口数据失败: {date}, 错误: {e}')
                continue

        # 标准化列和格式
        df = df.rename(columns={
            "股票代码": "code",
            "股票简称": "name",
            "预告类型": "forecast_type",
            "预测数值": "forecast_value",  # 单位是元
            "业绩变动幅度": "change_pct",
            "公告日期": "announcement_date"
        })
        df['code'] = df['code'].apply(lambda x: f'sh.{x}' if x.startswith('6') else f'sz.{x}')
        df['forecast_date'] = date
        all_dfs.append(df)

    if all_dfs:
        return pd.concat(all_dfs, ignore_index=True)
    else:
        print("❌ 无任何有效业绩预告数据")
        return pd.DataFrame()



from datetime import datetime

def get_quarter_end(date: datetime) -> str:
    """
    根据传入的 datetime 对象返回该日期所在季度的季度末日期（格式为 'YYYYMMDD'）

    :param date: datetime 对象，如 datetime(2025, 7, 13)
    :return: 对应季度末的日期字符串，如 '20250630'
    """
    year = date.year
    month = date.month

    if month <= 3:
        return f"{year-1}1231"
    elif month <= 6:
        return f"{year}0331"
    elif month <= 9:
        return f"{year}0630"
    else:
        return f"{year}0930"

def filter_stocks_by_forecast(stock_codes: list) -> list:

    # end = pd.to_datetime(datetime.today().strftime("%Y%m%d"))
    # print( get_quarter_end(end) )
    # print(get_quarter_end(datetime(2025, 1, 10)))  # 输出: 20250331
    # print(get_quarter_end(datetime(2025, 7, 13)))  # 输出: 20250630
    # print(get_quarter_end(datetime(2025, 10, 15)))
    # print(get_quarter_end(datetime(2025, 12, 17)))
    # print(get_quarter_end(datetime(2026, 1, 10)))
    # 指定日期列表（你可以选取近两个季度）
    # dates = ['20250331', '20250630', '20250930']
    dates = [  get_quarter_end(pd.to_datetime(datetime.today().strftime("%Y%m%d")) )]

    forecast_df = load_earnings_forecast(dates)

    print("📊 总数据量:", len(forecast_df))
    print(forecast_df.columns)
    print(forecast_df[['code', 'name', 'forecast_type', 'forecast_value']].head())


    # 2. 假设你当前候选池如下（格式必须是 sh./sz. 开头）
    # stock_pool = ['sh.600000', 'sz.000002', 'sh.603993', 'sz.300002']

    # 3. 过滤掉预亏等负面业绩预告
    """
        从候选股票中移除有不良业绩预告的股票
        :param stock_codes: 原始股票代码列表（格式：sh.600000）
        :param forecast_df: 业绩预告数据 DataFrame
        :return: 过滤后的股票列表
        """
    # 找到所有负面类型
    risky_df = forecast_df[forecast_df['forecast_type'].isin(NEGATIVE_FORECAST_KEYWORDS)]

    # 可选：也可以根据 forecast_value < 0 过滤
    # risky_df = risky_df[risky_df['forecast_value'] < 0]

    risky_set = set(risky_df['code'].unique())
    print(f"⚠️ 发现 {len(risky_set)} 只有业绩预警的股票: {risky_set}")

    filtered = [code for code in stock_codes if code not in risky_set]
    print(f"\n✅ 剔除业绩风险后的股票：{filtered}")
    return filtered


if __name__ == '__main__':
    end = pd.to_datetime(datetime.today().strftime("%Y%m%d"))
    print( get_quarter_end(end) )

    print(get_quarter_end(datetime(2025, 1, 10)))  # 输出: 20250331
    print(get_quarter_end(datetime(2025, 7, 13)))  # 输出: 20250630
    print(get_quarter_end(datetime(2025, 10, 15)))
    print(get_quarter_end(datetime(2025, 12, 17)))
    print(get_quarter_end(datetime(2026, 1, 10)))
    # 指定日期列表（你可以选取近两个季度）
    dates = ['20250331', '20250630', '20250930']
    dates = [    get_quarter_end(datetime(2025, 1, 10)),  # 输出: 20250331
            get_quarter_end(datetime(2025, 7, 13)),  # 输出: 20250630
            get_quarter_end(datetime(2025, 10, 15)),
            get_quarter_end(datetime(2025, 12, 17)),
            get_quarter_end(datetime(2026, 1, 10))]

    forecast_df = load_earnings_forecast(dates)

    print("📊 总数据量:", len(forecast_df))
    print(forecast_df[['code', 'name', 'forecast_type', 'forecast_value']].head())


    # 2. 假设你当前候选池如下（格式必须是 sh./sz. 开头）
    stock_pool = ['sh.600000', 'sz.000002', 'sh.603993', 'sz.300002']

    # 3. 过滤掉预亏等负面业绩预告
    safe_stocks = filter_stocks_by_forecast(stock_pool)

    print(f"\n✅ 剔除业绩风险后的股票：{safe_stocks}")

