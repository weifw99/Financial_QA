"""配置文件

主要配置：
1. 数据源访问凭证
2. 数据存储路径
3. 其他系统参数
"""

from pathlib import Path

# 项目根目录
BASE_DIR = Path(__file__).parent.parent

# 数据存储目录
DATA_DIR = BASE_DIR / 'data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# 数据源配置
MARKET_DATA = {
    'provider': 'baostock',  # 使用baostock作为行情数据源
    'token': None,  # baostock不需要token
}

# 数据同步配置
SYNC_CONFIG = {
    # 'mode': 'incremental',  # 同步模式: full(全量) / incremental(增量)
    # 'data_types': ['market', 'financial', 'news'],  # 需要同步的数据类型
    # 'data_types': ['market', 'financial'],  # 需要同步的数据类型
    'data_types': ['market'],  # 需要同步的数据类型
    # 'data_types': ['financial'],  # 需要同步的数据类型
    'incremental_days': 3,  # 增量同步时的时间范围（天）
    'financial_years': 5,  # 财务数据同步的年限范围
    'news_days': 365,  # 新闻数据同步的时间范围（天）
    'process_num': 3,  # 并行处理的进程数
}

# 其他配置参数
CONFIG = {
    'cache_duration': 86400,  # 缓存时间（秒）
    'retry_times': 3,  # API调用重试次数
    'timeout': 30,  # API调用超时时间（秒）
}