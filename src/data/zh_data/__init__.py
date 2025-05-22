"""数据获取模块

主要功能：
1. 市场数据获取（A股K线、逐笔成交、盘口数据）
2. 财务数据（财务报表、盈利预测）
3. 新闻数据（财经新闻、公告、研报）
4. 基金/ETF/指数数据

数据源：
- Tushare API
- 东方财富/同花顺
- 新浪财经/网易财经

技术栈：
- 数据拉取：requests, aiohttp, CCXT
- 数据存储：Parquet, ClickHouse, Kafka, Redis
- ETL处理：Pandas, Dask
"""

from pathlib import Path

# 定义数据存储路径
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / 'data/zh_data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# 创建必要的目录
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    dir_path.mkdir(exist_ok=True)