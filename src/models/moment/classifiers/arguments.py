"""参数管理"""

import os
from dataclasses import dataclass, field
from typing import Optional
from transformers import TrainingArguments


@dataclass
class ModelArguments(TrainingArguments):
    """模型相关参数"""
    
    model_name_or_path: str = field(
        default=None, metadata={"help": "预训练模型路径或huggingface模型ID"}
    )
    task_name: str = field(
        default="forecasting",
        metadata={"help": "任务名称"}
    )
    seq_len: int = field(
        default=512,
        metadata={"help": "序列长度"}
    )
    patch_len: int = field(
        default=8,
        metadata={"help": "长度"}
    )
    patch_stride_len: int = field(
        default=8,
        metadata={"help": "步长"}
    )
    forecast_horizon: int = field(
        default=192,
        metadata={"help": "预测步长"}
    )
    head_dropout: float = field(
        default=0.1,
        metadata={"help": "预测头dropout率"}
    )
    freeze_encoder: bool = field(
        default=True,
        metadata={"help": "是否冻结编码器"}
    )
    freeze_embedder: bool = field(
        default=True,
        metadata={"help": "是否冻结嵌入层"}
    )
    freeze_head: bool = field(
        default=False,
        metadata={"help": "是否冻结预测头"}
    )

@dataclass
class DataArguments:
    """数据相关参数"""
    
    train_data_path: str = field(
        metadata={"help": "数据路径"}
    )
    data_type: str = field(
        metadata={"help": "数据类型：min60, day, weekly"}
    )
    batch_size: int = field(
        default=32,
        metadata={"help": "批次大小"}
    )
    num_workers: int = field(
        default=4,
        metadata={"help": "数据加载线程数"}
    )
    random_seed: int = field(
        default=13,
        metadata={"help": "随机种子"}
    )

    def __post_init__(self):
        """验证参数"""
        if not os.path.exists(self.train_data_path):
            raise FileNotFoundError(f"训练数据文件不存在: {self.train_data_path}")
        if self.batch_size <= 0:
            raise ValueError("batch_size必须大于0")
        if self.num_workers < 0:
            raise ValueError("num_workers必须大于等于0") 