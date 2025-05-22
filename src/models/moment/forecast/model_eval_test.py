"""训练入口文件"""

import logging
import os
import sys
from pathlib import Path
from typing import Optional, Union, Dict, List

import numpy as np
import pandas as pd
import torch
from datasets import Dataset
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import HfArgumentParser, set_seed, TrainingArguments

from models.moment.forecast.arguments import ModelArguments, DataArguments
from models.moment.forecast.data import MomentDataset, MomentDataCollator, InformerDataset
from models.moment.forecast.modeling import MomentModel
from models.moment.forecast.show_k import show_k_2
from models.moment.forecast.trainer import MomentTrainer
from models.moment.momentfm import MOMENTPipeline
from models.moment.momentfm.utils.forecasting_metrics import get_forecasting_metrics

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


def evaluate(model: MOMENTPipeline = None,
        eval_datasets: Optional[List[Union[Dataset, Dict[str, Dataset]]]] = None, device: torch.device = None
) -> Dict[str, float]:
    # Evaluate the model on the test split
    model.eval()
    trues_, preds_ = [], []
    for eval_dataset in eval_datasets:
        eval_dataset: InformerDataset

        trues, preds, histories, histories_orig, trues_orig = [], [], [], [], []
        test_loader = DataLoader(eval_dataset, batch_size=8, shuffle=True, collate_fn=MomentDataCollator())

        scaler = eval_dataset.scaler
        ds_file_path = eval_dataset.full_file_path_and_name
        # 获取父目录信息，然后解密
        if eval_dataset.encrypted_data:
            encode_name = Path(ds_file_path).parent.name
            import base64
            # Base64解码
            code_name = base64.b64decode(encode_name).decode()
        else:
            encode_name = ''
            code_name = Path(ds_file_path).parent.name

        # print( code_name , Path(ds_file_path).parent.parent.parent)
        # print( code_name , Path(ds_file_path).parent.parent.parent.suffix)
        # 预测结果保存目录
        output_dir_ = f'{Path(ds_file_path).parent.parent}_test_predict/{code_name}_{encode_name}'
        # print(output_dir_)
        if not os.path.exists(output_dir_):
            os.makedirs(output_dir_)

        print(f"Evaluating on {ds_file_path}")
        print(f"eval_dataset.columns_names {eval_dataset.columns_names}")
        with torch.no_grad():
            for batch in tqdm(test_loader, total=len(test_loader)):
                # print(batch)
                timeseries = batch['timeseries'].to(device)
                timeseries_orig = batch['timeseries_orig'].to(device)
                forecast = batch['forecast'].to(device)
                forecast_orig = batch['forecast_orig'].to(device)
                input_mask = batch['input_mask'].to(device)

                # print(timeseries.shape, input_mask.shape)
                output = model.forecast(x_enc=timeseries, input_mask=input_mask)
                # print(output)

                true_ = forecast
                predict_forecast = output.forecast

                # 数据转换到cpu
                predict = predict_forecast.detach().cpu().numpy()
                # batch * feature * forecast_ho
                true_ = true_.detach().cpu().numpy()
                timeseries = timeseries.detach().cpu().numpy()
                timeseries_orig = timeseries_orig.detach().cpu().numpy()
                forecast_orig = forecast_orig.detach().cpu().numpy()

                # 对输出进行反归一化，使用训练集的scaler
                temp_array = []
                # 分batch存储数据
                for i in range(timeseries_orig.shape[0]):
                    predict = predict_forecast.detach().cpu().numpy()

                    preds_reshaped = predict[i].T
                    # 使用 scaler 进行反归一化
                    preds_convert_batch = scaler.inverse_transform(preds_reshaped)
                    # 将反归一化后的数据重塑回原来的形状
                    # temp_array.append(preds_convert_batch.T)

                    # print(preds_convert_batch.shape)

                    # 原始输入
                    input_ = timeseries_orig[i].T
                    result_ = forecast_orig[i].T
                    predict_ = preds_convert_batch

                    # print(input_.shape, result_.shape, predict_.shape)
                    input_df = pd.DataFrame(input_, columns=eval_dataset.columns_names)
                    input_df.to_csv(f'{output_dir_}//{eval_dataset.end_date}_{i}_input.csv')
                    result_df = pd.DataFrame(result_, columns=eval_dataset.columns_names)
                    result_df.to_csv(f'{output_dir_}//{eval_dataset.end_date}_{i}_result.csv')
                    predict_df = pd.DataFrame(predict_, columns=eval_dataset.columns_names)
                    predict_df.to_csv(f'{output_dir_}//{eval_dataset.end_date}_{i}_predict.csv')

                    show_k_2(input_df, result_df, f'{output_dir_}//{eval_dataset.end_date}_{i}_result.png')
                    show_k_2(input_df, predict_df, f'{output_dir_}//{eval_dataset.end_date}_{i}_predict.png')

                preds_convert = np.array(temp_array)

                trues_.append(true_)
                preds_.append(predict)


                trues.append(true_)
                preds.append(predict)
                histories.append(timeseries)

                histories_orig.append(timeseries_orig)
                trues_orig.append(forecast_orig)

            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)

            # preds_convert = scaler.inverse_transform(preds)

            histories = np.concatenate(histories, axis=0)
            histories_orig = np.concatenate(histories_orig, axis=0)
            trues_orig = np.concatenate(trues_orig, axis=0)
            # print(histories.shape, )

    trues_ = np.concatenate(trues_, axis=0)
    preds_ = np.concatenate(preds_, axis=0)

    metrics = get_forecasting_metrics(y=trues_, y_hat=preds_, reduction='mean')

    metrics_dict = dict()
    metrics_dict[f'mae'] = metrics.mae
    metrics_dict[f'mse'] = metrics.mse
    metrics_dict[f'rmse'] = metrics.rmse
    metrics_dict[f'mape'] = metrics.mape
    metrics_dict[f'smape'] = metrics.smape
    print(f"Test MSE: {metrics.mse:.3f} | Test MAE: {metrics.mae:.3f}, {metrics}")

    return metrics_dict


def evaluate1(model: MOMENTPipeline = None,
        eval_datasets: Optional[List[Union[Dataset, Dict[str, Dataset]]]] = None, device: torch.device = None
) -> Dict[str, float]:
    # Evaluate the model on the test split
    model.eval()
    trues_, preds_ = [], []
    for eval_dataset in eval_datasets:
        eval_dataset: InformerDataset

        trues, preds, histories, histories_orig, trues_orig = [], [], [], [], []
        test_loader = DataLoader(eval_dataset, batch_size=8, shuffle=True, collate_fn=MomentDataCollator())

        scaler = eval_dataset.scaler
        ds_file_path = eval_dataset.full_file_path_and_name
        # 获取父目录信息，然后解密
        if eval_dataset.encrypted_data:
            encode_name = Path(ds_file_path).parent.name
            import base64
            # Base64解码
            code_name = base64.b64decode(encode_name).decode()
        else:
            encode_name = ''
            code_name = Path(ds_file_path).parent.name

        # print( code_name , Path(ds_file_path).parent.parent.parent)
        # print( code_name , Path(ds_file_path).parent.parent.parent.suffix)
        # 预测结果保存目录
        output_dir_ = f'{Path(ds_file_path).parent.parent}_test_predict1/{code_name}_{encode_name}'
        # print(output_dir_)
        if not os.path.exists(output_dir_):
            os.makedirs(output_dir_)

        print(f"Evaluating on {ds_file_path}")
        print(f"eval_dataset.columns_names {eval_dataset.columns_names}")
        with torch.no_grad():
            for batch in tqdm(test_loader, total=len(test_loader)):
                # print(batch)
                timeseries = batch['timeseries'].to(device)
                timeseries_orig = batch['timeseries_orig'].to(device)
                forecast = batch['forecast'].to(device)
                forecast_orig = batch['forecast_orig'].to(device)
                input_mask = batch['input_mask'].to(device)


                true_ = forecast

                # 数据转换到cpu
                predict_batch = []
                # batch * feature * forecast_ho
                true_ = true_.detach().cpu().numpy()
                timeseries = timeseries.detach().cpu().numpy()
                timeseries_orig = timeseries_orig.detach().cpu().numpy()
                forecast_orig = forecast_orig.detach().cpu().numpy()

                # 对输出进行反归一化，使用训练集的scaler
                temp_array = []
                # 分batch存储数据
                for i in range(timeseries_orig.shape[0]):
                    scaler = StandardScaler()
                    scaler.fit(timeseries_orig[i].T)
                    timeseries_i = scaler.transform(timeseries_orig[i].T)
                    # print(input_mask.shape, input_mask[0:1, :].shape, torch.tensor([timeseries_i.T], dtype=torch.float32).shape)
                    output1 = model.forecast(x_enc=torch.tensor([timeseries_i.T], dtype=torch.float32, device=device),
                                             input_mask=input_mask[0:1, :])
                    # print(output1)
                    predict_forecast = output1.forecast
                    predict = predict_forecast.detach().cpu().numpy()

                    # preds_reshaped = predict[i].T
                    preds_reshaped = predict[0].T
                    # 使用 scaler 进行反归一化
                    preds_convert_batch = scaler.inverse_transform(preds_reshaped)
                    predict_batch.append(preds_convert_batch)

                    # 将反归一化后的数据重塑回原来的形状
                    # temp_array.append(preds_convert_batch.T)

                    # print(preds_convert_batch.shape)

                    # 原始输入
                    input_ = timeseries_orig[i].T
                    result_ = forecast_orig[i].T
                    predict_ = preds_convert_batch

                    # print(input_.shape, result_.shape, predict_.shape)
                    input_df = pd.DataFrame(input_, columns=eval_dataset.columns_names)
                    input_df.to_csv(f'{output_dir_}//{eval_dataset.end_date}_{i}_input.csv')
                    result_df = pd.DataFrame(result_, columns=eval_dataset.columns_names)
                    result_df.to_csv(f'{output_dir_}//{eval_dataset.end_date}_{i}_result.csv')
                    predict_df = pd.DataFrame(predict_, columns=eval_dataset.columns_names)
                    predict_df.to_csv(f'{output_dir_}//{eval_dataset.end_date}_{i}_predict.csv')

                    show_k_2(input_df[-20:], result_df, f'{output_dir_}//{eval_dataset.end_date}_{i}_result.png')
                    show_k_2(input_df[-20:], predict_df, f'{output_dir_}//{eval_dataset.end_date}_{i}_predict.png')
                preds_convert = np.array(temp_array)

                trues_.append(true_)
                preds_.append(predict)


                trues.append(true_)
                preds.append(predict)
                histories.append(timeseries)

                histories_orig.append(timeseries_orig)
                trues_orig.append(forecast_orig)

            trues = np.concatenate(trues, axis=0)
            preds = np.concatenate(preds, axis=0)

            # preds_convert = scaler.inverse_transform(preds)

            histories = np.concatenate(histories, axis=0)
            histories_orig = np.concatenate(histories_orig, axis=0)
            trues_orig = np.concatenate(trues_orig, axis=0)
            # print(histories.shape, )

    trues_ = np.concatenate(trues_, axis=0)
    preds_ = np.concatenate(preds_, axis=0)

    metrics = get_forecasting_metrics(y=trues_, y_hat=preds_, reduction='mean')

    metrics_dict = dict()
    metrics_dict[f'mae'] = metrics.mae
    metrics_dict[f'mse'] = metrics.mse
    metrics_dict[f'rmse'] = metrics.rmse
    metrics_dict[f'mape'] = metrics.mape
    metrics_dict[f'smape'] = metrics.smape
    print(f"Test MSE: {metrics.mse:.3f} | Test MAE: {metrics.mae:.3f}, {metrics}")

    return metrics_dict

def main():
    """主函数"""
    train_data_path = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/market"
    train_data_path = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data_copy"

    data_args: DataArguments = DataArguments(train_data_path=train_data_path,
                                             data_type="daily",
                                             limit=20,
                                             encrypted_data=True)
    seq_len = 384
    forecast_horizon = 64

    print(data_args)

    # 初始化数据集
    train_datasets, eval_datasets = MomentDataset.get_informer_datasets(data_args, seq_len=seq_len, forecast_horizon=forecast_horizon)
    device = torch.device("cuda" if torch.cuda.is_available() else "mps")

    # 初始化模型
    pretrained_model_name_or_path = '/Users/dabai/liepin/study/llm/Financial_QA/src/models/moment/forecast/outputs'
    pretrained_model_name_or_path = '/Users/dabai/liepin/study/llm/Financial_QA/src/models/moment/forecast/outputs/daily'
    pretrained_model_name_or_path = '/Users/dabai/liepin/study/llm/Financial_QA/src/models/moment/forecast/outputs/export/ck-06-6000'
    model = MomentModel.load(pretrained_model_name_or_path, device)
    model: MOMENTPipeline

    # evaluate(model, eval_datasets, device)
    evaluate1(model, eval_datasets, device)
    # Epoch 1.0: Test MSE: 1.563 | Test MAE: 0.893, ForecastingMetrics(mae=0.89260036, mse=1.5633519, mape=135.92817783355713, smape=98.49332571029663, rmse=1.2503407)



if __name__ == "__main__":
    main()