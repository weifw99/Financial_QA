import akshare as ak
import pandas as pd
import os
from datetime import datetime, timedelta
import time

def download_etf_data(etf_code, etf_name, start_date, end_date, save_path='data'):
    try:
        df = ak.fund_etf_hist_em(etf_code, start_date=start_date, end_date=end_date)
        if not df.empty:
            df = df[['日期', '开盘', '最高', '最低', '收盘', '成交量']]
            df.columns = ['date', 'open', 'high', 'low', 'close', 'volume']
            df['date'] = pd.to_datetime(df['date'])
            df.set_index('date', inplace=True)
            os.makedirs(save_path, exist_ok=True)
            df.to_csv(os.path.join(save_path, f"{etf_code}.csv"))
            print(f"[✓] 下载成功：{etf_name}（{etf_code}）")
    except Exception as e:
        print(f"[✗] 下载失败：{etf_name}（{etf_code}） - {e}")

def load_all_candidates():
    dfs = []
    for file in ['top_etf_simple.csv', 'top_etf_r2.csv']:
        if os.path.exists(file):
            dfs.append(pd.read_csv(file))
    if not dfs:
        raise FileNotFoundError("未找到候选池文件，请先运行 etf_filter.py")
    combined = pd.concat(dfs, axis=0).drop_duplicates(subset='基金代码')
    return combined

def batch_download():
    today = datetime.today()
    start = today - timedelta(days=365)
    start_str = start.strftime('%Y%m%d')
    end_str = today.strftime('%Y%m%d')

    df = load_all_candidates()
    for _, row in df.iterrows():
        code = row['基金代码']
        name = row['基金简称']
        download_etf_data(code, name, start_str, end_str)
        time.sleep(1.2)

if __name__ == "__main__":
    batch_download()