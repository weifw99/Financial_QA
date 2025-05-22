# 使用 qlib 构造 dataset 数据集，功能类似dataset = init_instance_by_config(config["task"]["dataset"])，但是更加灵活
# 构造dataset 依赖类：DataHandlerLP、DatasetH、QlibDataLoader

import logging
from typing import Dict, Any, Optional, Union, List
from qlib.data.dataset import DatasetH
from qlib.data.dataset.handler import DataHandlerLP
from qlib.data.dataset.loader import QlibDataLoader
from qlib.data.dataset.processor import Processor

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def create_dataset(
    instruments: Union[str, List[str]] = "csi500",
    start_time: Optional[str] = None,
    end_time: Optional[str] = None,
    freq: str = "day",
    segments: Optional[Dict[str, List[str]]] = None,
    data_loader_config: Optional[Dict[str, Any]] = None,
    learn_processors: Optional[List[Union[Processor, Dict[str, Any]]]] = None,
    infer_processors: Optional[List[Union[Processor, Dict[str, Any]]]] = None,
    fit_start_time: Optional[str] = None,
    fit_end_time: Optional[str] = None,
    **kwargs
) -> DatasetH:
    """
    创建一个灵活的 Qlib 数据集
    
    Args:
        instruments: 股票池，可以是单个字符串或列表
        start_time: 开始时间
        end_time: 结束时间
        freq: 数据频率
        segments: 数据集分段配置，例如 {"train": ["2018-01-01", "2018-12-31"]}
        data_loader_config: 数据加载器配置
        learn_processors: 训练集处理器列表
        infer_processors: 推理集处理器列表
        fit_start_time: 处理器拟合开始时间
        fit_end_time: 处理器拟合结束时间
        **kwargs: 其他参数
        
    Returns:
        DatasetH: 构造好的数据集
    """
    # 设置默认的数据加载器配置
    if data_loader_config is None:
        data_loader_config = {
            "class": "QlibDataLoader",
            "kwargs": {
                "config": {
                    "feature": ["$close", "$open", "$high", "$low", "$volume"],
                    "label": ["Ref($close, -2)/Ref($close, -1) - 1"]
                }
            }
        }
    
    # 设置默认的处理器
    if learn_processors is None:
        learn_processors = []
    if infer_processors is None:
        infer_processors = []
    
    # 设置默认的数据集分段
    if segments is None:
        segments = {
            "train": ["2018-01-01", "2018-12-31"],
            "valid": ["2019-01-01", "2019-12-31"],
            "test": ["2020-01-01", "2020-12-31"]
        }
    
    # 创建数据处理器
    handler = DataHandlerLP(
        instruments=instruments,
        start_time=start_time,
        end_time=end_time,
        freq=freq,
        data_loader=data_loader_config,
        learn_processors=learn_processors,
        infer_processors=infer_processors,
        fit_start_time=fit_start_time,
        fit_end_time=fit_end_time,
        **kwargs
    )
    
    # 创建数据集
    dataset = DatasetH(
        handler=handler,
        segments=segments
    )
    
    logger.info("成功创建数据集，包含以下分段: %s", list(segments.keys()))
    return dataset