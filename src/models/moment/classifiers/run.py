"""训练入口文件"""

import logging
import os
import sys
from pathlib import Path

import torch
from transformers import HfArgumentParser, set_seed

from models.moment.forecast.arguments import ModelArguments, DataArguments
from models.moment.forecast.data import get_data_sets, MomentDataCollator
from models.moment.forecast.modeling import MomentModel
from models.moment.forecast.trainer import MomentTrainer

logger = logging.getLogger(__name__)

def setup_logging(output_dir: str) -> None:
    """设置日志
    
    Args:
        output_dir: 输出目录
    """
    os.makedirs(output_dir, exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO,
        handlers=[
            logging.FileHandler(os.path.join(output_dir, "train.log")),
            logging.StreamHandler()
        ]
    )

def main():
    """主函数"""
    # 解析参数
    parser = HfArgumentParser((ModelArguments, DataArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args = parser.parse_json_file(
            json_file=os.path.abspath(sys.argv[1])
        )
    elif len(sys.argv) == 2 and (
            sys.argv[1].endswith(".yaml") or sys.argv[1].endswith(".yml")
    ):
        model_args, data_args = parser.parse_yaml_file(
            yaml_file=os.path.abspath(sys.argv[1])
        )
    else:
        model_args, data_args = parser.parse_args_into_dataclasses()

    # 设置输出目录
    output_dir = "outputs"
    os.makedirs(output_dir, exist_ok=True)
    
    # 设置日志
    setup_logging(output_dir)
    
    # 设置随机种子
    set_seed(data_args.random_seed)

    # 初始化数据集
    train_dataset, eval_dataset = get_data_sets(data_args, seq_len=model_args.seq_len, forecast_horizon=model_args.forecast_horizon)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    # 初始化模型
    model = MomentModel.from_pretrained(model_args, device=device)

    # 检查模型参数是否被冻结
    for name, param in model.named_parameters():
        if param.requires_grad:
            logger.info(f"可训练参数: {name}")
        else:
            logger.warning(f"冻结参数: {name}")

    # 创建训练器
    trainer = MomentTrainer(
        model=model,
        args=model_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        data_collator=MomentDataCollator(),
    )

    # 开始训练
    logger.info("开始训练...")
    trainer.train()
    logger.info("训练完成!")
    trainer.save_model()

if __name__ == "__main__":
    main() 