"""运行数据同步脚本

用于启动数据同步任务，同步最近7天的股票数据
"""

from src.data.zh_data.sync import DataSync
import asyncio

def main():
    # 初始化数据同步模块
    sync = DataSync()
    
    # 运行增量同步任务
    asyncio.run(sync.sync_stock())
    
if __name__ == '__main__':
    main()  