import os.path
import random
import time

import requests
import pandas as pd
from bs4 import BeautifulSoup
import re

import akshare as ak


def fetch_index_components_sina(indexid='399101', max_pages=100):
    """
    爬取新浪财经某个指数的最新成分股信息。

    参数：
    - indexid: 指数ID，如 '399101'
    - max_pages: 最多爬取的分页数

    返回：
    - DataFrame，包含列：品种代码、品种名称、纳入日期/ '证券标识'
    """
    all_data = []

    for page in range(max_pages):
        url = f"https://vip.stock.finance.sina.com.cn/corp/view/vII_NewestComponent.php?page={page}&indexid={indexid}"
        res = requests.get(url, timeout=10)
        res.encoding = 'gbk'  # 注意新浪使用 GBK 编码

        soup = BeautifulSoup(res.text, 'html.parser')
        table = soup.find('table', id='NewStockTable')

        if not table:
            print(f"❌ 页面 {page} 无有效表格，跳过")
            break

        rows = table.find_all('tr')[2:]  # 跳过表头
        if not rows:
            break

        for row in rows:
            cols = row.find_all('td')
            if len(cols) >= 3:
                code = cols[0].text.strip()
                name_tag = cols[1].find('a')
                name = name_tag.text.strip() if name_tag else cols[1].text.strip()
                date = cols[2].text.strip()
                href = name_tag['href'] if name_tag and name_tag.has_attr('href') else ''
                match = re.search(r'/company/(\w+)/', href)
                sec_id = match.group(1) if match else ''

                all_data.append((code, name, date, sec_id))

        time.sleep(random.randint(1, 3))

    df = pd.DataFrame(all_data, columns=['品种代码', '品种名称', '纳入日期', '证券标识'])
    # 使用正则在字母和数字之间加点
    df['type'] = df['证券标识'].str.replace(r'([a-zA-Z]+)(\d+)', r'\1.\2', regex=True)
    return df


def load_index_stock_cons(index_code):
    index_stock_cons_csindex_df = ak.index_stock_cons_csindex(symbol=index_code)
    index_stock_cons_csindex_df['type'] = index_stock_cons_csindex_df['交易所'].apply(
        lambda x: 'sz.' if x == '深圳证券交易所' else 'sh.') + index_stock_cons_csindex_df['成分券代码']
    return index_stock_cons_csindex_df


def load_index_stock_cons_dfcf(index_code = 'BK1158'):
    '''            if code.startswith("000"):
                query_code = 'sh' + query_code
            elif code.startswith("399"):
                query_code = 'sz' + query_code
    '''


    def get_exchange_prefix(code):
        code = str(code)
        if code.startswith('60') or code.startswith('688'):
            return 'sh'
        elif (code.startswith('00') or code.startswith('002')
              or code.startswith('300') or code.startswith('301')):
            return 'sz'
        else:
            return 'unknown'

    df = ak.stock_board_industry_cons_em(symbol=index_code)
    df['type'] = df['代码'].astype(str).apply(
        lambda c: f"{get_exchange_prefix(c)}.{c}" if get_exchange_prefix(c) != 'unknown' else None)
    return df

def get_exchange_prefix(code):
    code = str(code)
    if code.startswith('60') or code.startswith('688'):
        return 'sh'
    elif (code.startswith('00') or code.startswith('002')
          or code.startswith('300') or code.startswith('301')):
        return 'sz'
    else:
        return 'unknown'


if __name__ == '__main__':

    base_path = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/raw/index"
    base_market_path = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/market"

    zz_index_dict = {
        '000852': '中证1000',
        '932000': '中证2000',
    }
    for code, name in zz_index_dict.items():
        df = load_index_stock_cons(code)
        df.to_csv(f"{ base_path }/{ name}-{code}.csv", index=False)
        if code in ['000852']:
            continue
        # code_path = f"{base_market_path}/csi{code}"
        # if not os.path.exists(code_path):
        #     os.mkdir(code_path)
        # df.to_csv(f"{code_path}/daily.csv", index=False)
        # df.to_csv(f"{code_path}/daily_a.csv", index=False)


    sina_index_dict = {
        '399101': '中小综指',
        '399005': '中小板指数-中小100',
    }
    for code, name in sina_index_dict.items():
        df = fetch_index_components_sina(code)
        df.to_csv(f"{base_path}/{name}-{code}.csv", index=False)
        # code_path = f"{base_market_path}/{get_exchange_prefix( code)}{code}"
        # if not os.path.exists(code_path):
        #     os.mkdir(code_path)
        # df.to_csv(f"{code_path}/daily.csv", index=False)
        # df.to_csv(f"{code_path}/daily_a.csv", index=False)

    dfcf_index_dict = {
        'BK1158': '微盘股',
    }
    for code, name in dfcf_index_dict.items():
        df = load_index_stock_cons_dfcf(code)
        df.to_csv(f"{base_path}/{name}-{code}.csv", index=False)
        # code_path = f"{base_market_path}/{code}"
        # if not os.path.exists(code_path):
        #     os.mkdir(code_path)
        # df.to_csv(f"{code_path}/daily.csv", index=False)
        # df.to_csv(f"{code_path}/daily_a.csv", index=False)

