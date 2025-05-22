import logging
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.processor import ZScoreNorm, Fillna, DropnaLabel, DropnaProcessor
from .data_loader import get_data_loader

# 配置日志，忽略低于 ERROR 级别的日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def get_data_handler():
    qdl = get_data_loader()
    
    dh_params = {
        'instruments': ["sh600519"],
        'start_time': "20170101",
        'end_time': "20191231",
        'learn_processors': [
            DropnaLabel(),
            DropnaProcessor(),
        ],
        'infer_processors': [
            ZScoreNorm(fit_start_time="20170101", fit_end_time="20181231"),
            DropnaProcessor(),
            # Fillna(),  # Fillna 如果所有列都 NaN

        ],
        'data_loader': qdl,
    }

    logger.info("初始化 DataHandlerLP, 参数: %s", dh_params)
    dh = DataHandlerLP(**dh_params)

    logger.info("检查处理后数据是否存在 NaN 值...")
    df = dh.fetch(selector=slice("20170101", "20191231"))
    df.head()
    if df.isnull().values.any():
        logger.warning("处理后的数据中包含 NaN 值。请检查数据处理步骤。")
    
    return dh