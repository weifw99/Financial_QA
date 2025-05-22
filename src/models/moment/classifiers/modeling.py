"""模型模块"""

import logging
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
        with torch.cuda.amp.autocast():
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
        # 加载预训练模型
        model = MOMENTPipeline.from_pretrained(
            args.model_name_or_path,
            model_kwargs={
                'task_name': args.task_name,
                'forecast_horizon': args.forecast_horizon,
                'head_dropout': args.head_dropout,
                'weight_decay': args.weight_decay,
                'freeze_encoder': args.freeze_encoder,
                'freeze_embedder': args.freeze_embedder,
                'freeze_head': args.freeze_head,
                'seq_len': args.seq_len,
                "patch_len": args.patch_len,
                "patch_stride_len": args.patch_stride_len,
            }
        )
        model.init()
        
        # 创建模型实例
        return cls(model, args, device)
        
    def save_pretrained(self, output_dir: str):
        """保存模型
        
        Args:
            output_dir: 输出目录
        """
        self.model.save_pretrained(output_dir) 