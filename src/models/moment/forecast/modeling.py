"""模型模块"""
import json
import logging
import os
from pathlib import Path

import torch
import torch.nn as nn
from models.moment.momentfm import MOMENTPipeline
from models.moment.forecast.arguments import ModelArguments

logger = logging.getLogger(__name__)


class MomentModel(nn.Module):
    """MOMENT模型"""
    
    def __init__(self, model: MOMENTPipeline, args: ModelArguments, device: torch.device=None):
        """初始化模型
        
        Args:
            model: MOMENT模型
            args: 模型参数
        """
        super().__init__()
        self.args = args
        if device is not None:
            self.device = device
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        # 损失函数
        self.criterion = nn.MSELoss(reduction='mean').to(self.device)

    def forward(self, batch):
        """前向传播
        
        Args:
            batch: 输入数据
            
        Returns:
            dict: 模型输出
        """
        # 将数据移到设备
        # print("batch:", batch)
        timeseries = batch['timeseries'].to(self.device)
        input_mask = batch['input_mask'].to(self.device)
        forecast = batch['forecast'].to(self.device)

        # 前向传播
        # with torch.cuda.amp.autocast():
        #     output = self.model(x_enc=timeseries, input_mask=input_mask)
        #     loss = self.criterion(output.forecast, forecast)
        output = self.model(x_enc=timeseries, input_mask=input_mask)
        loss = self.criterion(output.forecast, forecast)
            
        return {
            'loss': loss,
            'forecast': output.forecast,
            'true': forecast
        }
        
    @classmethod
    def from_pretrained(cls, args: ModelArguments, device: torch.device=None):
        """从预训练模型加载
        
        Args:
            args: 模型参数
            
        Returns:
            MomentModel: 模型实例
        """
        logger.info(f"从预训练模型加载... {args}")
        # 加载预训练模型
        model_kwargs = {
            'task_name': args.task_name,
            'forecast_horizon': args.forecast_horizon,
            'head_dropout': args.head_dropout,
            'weight_decay': args.weight_decay,
            'freeze_encoder': args.freeze_encoder,
            'freeze_encoder_layers': args.freeze_encoder_layers,
            'freeze_embedder': args.freeze_embedder,
            'freeze_head': args.freeze_head,
            'seq_len': args.seq_len,
            "patch_len": args.patch_len,
            "patch_stride_len": args.patch_stride_len,
        }
        print('model_kwargs', model_kwargs)
        model = MOMENTPipeline.from_pretrained(
            args.model_name_or_path,
            model_kwargs=model_kwargs
        )
        model.config.model_name_or_path = args.model_name_or_path
        model.init()

        logger.info(f"model config ... {model.config}")
        # 创建模型实例
        return cls(model, args, device)
        
    def save_pretrained(self, output_dir: str):
        """保存模型
        
        Args:
            output_dir: 输出目录
        """
        # self.model.save_pretrained(output_dir)
        torch.save(self.model.state_dict(), os.path.join(output_dir, 'model.pth'))
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))

        logger.info(f"Saving model checkpoint to {self.model.config}")
        # 更新config.json
        config_path = Path(output_dir) / 'config.json'
        config_path.unlink(missing_ok=True)

        # 将 NamespaceWithDefaults 对象转换为字典
        config_dict = vars(self.model.config)
        config_str = json.dumps(config_dict, sort_keys=True, indent=2)
        config_path.write_text(config_str)

    @staticmethod
    def load(input_path: str, device: torch.device, model_type: str="ft_model") -> MOMENTPipeline:

        logging.info("Loading model from {}".format(input_path))
        config_path = os.path.join(input_path, "config.json")
        config = json.loads(open(config_path, "r").read())
        logging.info(f"Loading model config {config}")
        print(config)
        model_kwargs = {
            'task_name': config['model_kwargs']['task_name'],
            'forecast_horizon': config['model_kwargs']['forecast_horizon'],
            # 'head_dropout': config['model_kwargs']['head_dropout'],
            'head_dropout': 0,
            'weight_decay': config['model_kwargs']['weight_decay'],
            'freeze_encoder': config['model_kwargs']['freeze_encoder'],
            'freeze_encoder_layers': config['model_kwargs']['freeze_encoder_layers'],
            'freeze_embedder': config['model_kwargs']['freeze_embedder'],
            'freeze_head': config['model_kwargs']['freeze_head'],
            'seq_len': config['model_kwargs']['seq_len'],
            "patch_len": config['model_kwargs']['patch_len'],
            "patch_stride_len": config['model_kwargs']['patch_stride_len'],
        }
        print('model_kwargs', model_kwargs)

        model = MOMENTPipeline(config=config, model_type=model_type, **{'model_kwargs': model_kwargs})

        model.load_state_dict(torch.load(os.path.join(input_path, 'model.pth'), weights_only=False, map_location=torch.device('cpu')) )
        return model.to(device)
