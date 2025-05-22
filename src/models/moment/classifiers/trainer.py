"""训练器模块"""

import logging
import os
from typing import Optional, Dict, Union, List

import numpy as np
import torch
from datasets import Dataset
from tqdm import tqdm
from transformers.trainer import Trainer
from transformers import TrainingArguments

from models.moment.momentfm.utils.forecasting_metrics import get_forecasting_metrics

logger = logging.getLogger(__name__)


class MomentTrainer(Trainer):
    """MOMENT模型训练器"""
    def __init__(
            self,
            model = None,
            args: TrainingArguments = None,
            data_collator = None,
            train_dataset = None,
            eval_dataset= None,
    ):
        super().__init__(
            model=model,
            args=args,
            train_dataset=train_dataset,
            eval_dataset=eval_dataset,
            data_collator=data_collator,
        )

    def get_train_dataloader(self) -> DataLoader:
        """
        Returns the training [`~torch.utils.data.DataLoader`].

        Will use no sampler if `train_dataset` does not implement `__len__`, a random sampler (adapted to distributed
        training if necessary) otherwise.

        Subclass and override this method if you want to inject some custom behavior.
        """
        if self.train_dataset is None:
            raise ValueError("Trainer: training requires a train_dataset.")

        train_dataset = self.train_dataset
        data_collator = self.data_collator
        if is_datasets_available() and isinstance(train_dataset, datasets.Dataset):
            train_dataset = self._remove_unused_columns(train_dataset, description="training")
        else:
            data_collator = self._get_collator_with_removed_columns(data_collator, description="training")

        dataloader_params = {
            "batch_size": self._train_batch_size,
            "collate_fn": data_collator,
            "num_workers": self.args.dataloader_num_workers,
            "pin_memory": self.args.dataloader_pin_memory,
            "persistent_workers": self.args.dataloader_persistent_workers,
        }

        if not isinstance(train_dataset, torch.utils.data.IterableDataset):
            dataloader_params["sampler"] = self._get_train_sampler()
            dataloader_params["drop_last"] = self.args.dataloader_drop_last
            dataloader_params["worker_init_fn"] = seed_worker
            dataloader_params["prefetch_factor"] = self.args.dataloader_prefetch_factor

        return self.accelerator.prepare(DataLoader(train_dataset, **dataloader_params))

    def compute_metrics(self, eval_pred):
        """计算评估指标

        Args:
            eval_pred: 评估预测结果

        Returns:
            评估指标字典
        """
        trues, preds = eval_pred
        metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction='mean')

        return {
            "mse": metrics.mse,
            "mae": metrics.mae
        }

    def _save_checkpoint(self, model, trial, metrics=None):
        # Save model checkpoint
        PREFIX_CHECKPOINT_DIR = "checkpoint"
        checkpoint_folder = f"{PREFIX_CHECKPOINT_DIR}-{self.state.global_step}"

        if self.hp_search_backend is None and trial is None:
            self.store_flos()

        run_dir = self._get_output_dir(trial=trial)
        output_dir = os.path.join(run_dir, checkpoint_folder)
        self.save_model(output_dir, _internal_call=True)


    def _save(self, output_dir: Optional[str] = None):
        """保存模型

        Args:
            output_dir: 输出目录
        """
        output_dir = output_dir if output_dir is not None else self.args.output_dir
        os.makedirs(output_dir, exist_ok=True)
        logger.info("Saving model checkpoint to %s", output_dir)

        # 保存模型
        self.model.save_pretrained(output_dir)
        torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        # 保存训练参数
        # torch.save(self.args, os.path.join(output_dir, "training_args.bin"))
        # 保存模型参数
        # torch.save(self.args, os.path.join(output_dir, "model_args.bin"))
        # 保存数据参数
        # torch.save(self.args, os.path.join(output_dir, "data_args.bin"))


    def training_step(self, model, inputs, num_items_in_batch):
        """训练步骤

        Args:
            model: 模型
            inputs: 输入数据
            num_items_in_batch: 批次中的样本数

        Returns:
            训练损失
        """
        model.train()

        with self.compute_loss_context_manager():
            loss = self.compute_loss(model, inputs)

        del inputs

        if self.args.n_gpu > 1:
            loss = loss.mean()  # 多 GPU 并行训练时取平均
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps

        # 假设我们要添加 L2 正则化
        # l2_lambda = 0.001
        # l2_reg = torch.tensor(0., requires_grad=True).to(self.args.device)
        # for name, param in model.named_parameters():
        #     if 'weight' in name:
        #         l2_reg = l2_reg + torch.norm(param)
        # loss = loss + l2_lambda * l2_reg

        # 检查损失值是否有效
        if torch.isnan(loss) or torch.isinf(loss):
            # logger.warning(f"Invalid loss value: {loss.item()}, outputs: {outputs}")
            logger.warning(f"Invalid loss value: {loss.item()} ")
            return torch.tensor(0.0, device=model.device)

        # # 反向传播
        loss.backward()

        return loss.detach()


    def compute_loss(self, model, inputs, return_outputs=False):
        """计算损失

        Args:
            model: 模型
            inputs: 输入数据
            return_outputs: 是否返回输出

        Returns:
            损失值或(损失值, 输出)
        """
        # 将输入数据移到正确的设备上
        inputs = {k: v.to(model.device) for k, v in inputs.items()}

        outputs = model(inputs)
        loss = outputs['loss']

        if return_outputs:
            return loss, outputs
        return loss

    def create_optimizer(self):
        # 这里使用 Adam 优化器
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=self.args.learning_rate)
        return self.optimizer

    def evaluate(
            self,
            eval_dataset: Optional[Union[Dataset, Dict[str, Dataset]]] = None,
            ignore_keys: Optional[List[str]] = None,
            metric_key_prefix: str = "eval",
    ) -> Dict[str, float]:
        # Evaluate the model on the test split
        trues, preds, histories, losses = [], [], [], []
        self.model.eval()
        test_loader = self.get_eval_dataloader()
        with torch.no_grad():
            # for timeseries, forecast, input_mask in tqdm(test_loader, total=len(test_loader)):
            for batch in tqdm(test_loader, total=len(test_loader)):
                output = self.model(batch)

                loss = output['loss']
                forecast = output['forecast']
                true_ = output['true']

                losses.append(loss.item())

                trues.append(true_.detach().cpu().numpy())
                preds.append(forecast.detach().cpu().numpy())
                histories.append(batch['timeseries'].detach().cpu().numpy())

        losses = np.array(losses)
        average_loss = np.average(losses)

        trues = np.concatenate(trues, axis=0)
        preds = np.concatenate(preds, axis=0)
        histories = np.concatenate(histories, axis=0)

        metrics = get_forecasting_metrics(y=trues, y_hat=preds, reduction='mean')


        metrics_dict = dict()
        metrics_dict[f'{metric_key_prefix}_average_loss'] = average_loss
        metrics_dict[f'{metric_key_prefix}_mae'] = metrics.mae
        metrics_dict[f'{metric_key_prefix}_mse'] = metrics.mse
        metrics_dict[f'{metric_key_prefix}_rmse'] = metrics.rmse
        metrics_dict[f'{metric_key_prefix}_mape'] = metrics.mape
        metrics_dict[f'{metric_key_prefix}_smape'] = metrics.smape
        print(f"Epoch {self.state.epoch}: Test MSE: {metrics.mse:.3f} | Test MAE: {metrics.mae:.3f}, {metrics}")

        return metrics_dict