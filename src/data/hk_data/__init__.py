"""港股数据模块"""

import os
from pathlib import Path


# 定义数据存储路径
BASE_DIR = Path(__file__).parent.parent.parent.parent
DATA_DIR = BASE_DIR / 'data/hk_data'
RAW_DATA_DIR = DATA_DIR / 'raw'
PROCESSED_DATA_DIR = DATA_DIR / 'processed'

# 创建必要的目录
for dir_path in [DATA_DIR, RAW_DATA_DIR, PROCESSED_DATA_DIR]:
    if not os.path.exists(dir_path):
        os.makedirs(dir_path) 