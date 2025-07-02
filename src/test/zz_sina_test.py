import random
import time

import requests
import pandas as pd
from bs4 import BeautifulSoup
import re


def fetch_index_components(indexid='399101', max_pages=50):
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
    return df

if __name__ == '__main__':
    # 示例使用：获取中小综指最新成分股（399101）
    df = fetch_index_components('399101', max_pages=100)
    print(df.head())