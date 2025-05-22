"""港股数据同步入口文件"""

import asyncio
import os
import logging
from datetime import datetime
import argparse

from .sync import DataSync
from . import DATA_DIR

# 配置日志
def setup_logging():
    """配置日志"""
    log_dir = os.path.join(os.path.dirname(DATA_DIR), 'logs')
    if not os.path.exists(log_dir):
        os.makedirs(log_dir)
        
    log_file = os.path.join(log_dir, f'hk_sync_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log')
    
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_file, encoding='utf-8'),
            logging.StreamHandler()
        ]
    )
    return log_file

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='港股数据同步工具')
    parser.add_argument('--process-num', type=int, default=4, help='并行进程数')
    parser.add_argument('--start-date', type=str, default='2010-01-01', help='数据开始日期')
    parser.add_argument('--end-date', type=str, default=None, help='数据结束日期')
    parser.add_argument('--data-dir', type=str, default=DATA_DIR, help='数据存储目录')
    parser.add_argument('--full-sync', action='store_true', help='是否执行全量同步')
    return parser.parse_args()

async def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置日志
    log_file = setup_logging()
    logging.info(f"开始同步港股数据 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logging.info(f"日志文件: {log_file}")
    logging.info(f"命令行参数: {args}")
    
    try:
        # 创建数据同步实例
        sync = DataSync(
            base_dir=args.data_dir,
            process_num=args.process_num,
            start_date=args.start_date,
            end_date=args.end_date,
            full_sync=args.full_sync
        )
        
        # 同步数据
        await sync.sync_stock()
        
        logging.info(f"港股数据同步完成 - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    except Exception as e:
        logging.error(f"同步过程中发生错误: {str(e)}", exc_info=True)
        raise

if __name__ == "__main__":
    # 运行主函数
    asyncio.run(main()) 