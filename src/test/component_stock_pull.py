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




if __name__ == '__main__':
    import akshare as ak

    # 获取微盘股 BK1158 的成分股
    bk1158_stock_list = ak.stock_board_industry_cons_ths(symbol="BK1158")
    print(bk1158_stock_list.head())
    import akshare as ak

    # 获取 BK1158 的日线行情（微盘股指数）
    stock_board_industry_index_df = ak.stock_board_industry_index_ths(symbol="BK1158")
    print(stock_board_industry_index_df.head())

    base_path = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/raw/index"

    zz_index_dict = {
        'BK1158': '微盘股',
        # '000852': '中证1000',
        # '932000': '中证2000',
    }
    for code, name in zz_index_dict.items():
        load_index_stock_cons(code).to_csv(f"{ base_path }/{ name}-{code}.csv", index=False)

    sina_index_dict = {
        # '399101': '中小综指',
    }
    for code, name in sina_index_dict.items():
        fetch_index_components_sina(code).to_csv(f"{ base_path }/{ name}-{code}.csv", index=False)

