"""数据加载模块"""

import os
import torch
from torch.utils.data import Dataset
from dataclasses import dataclass

from transformers.data.data_collator import DataCollatorMixin

from models.moment.forecast.arguments import DataArguments

from typing import Optional, List

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from models.moment.util.openssl_decrypted import decrypt_file_to_pandas


class InformerDataset(Dataset):
    def __init__(
        self,
        forecast_horizon: Optional[int] = 45,
        data_split: str = "train",
        seq_len: int = 512,
        start_date: str = "2010-01-01",
        end_date: str = "2024-11-01",
        data_stride_len: int = 1,
        task_name: str = "forecasting",
        random_seed: int = 42,
        file_path: str = None,
        encrypted_data: bool = False,
        columns_names: List[str]=['open', 'high', 'low', 'close', 'volume', 'amount']
    ):
        """
        Parameters
        ----------
        forecast_horizon : int
            Length of the prediction sequence.
        data_split : str
            Split of the dataset, 'train' or 'test'.
        data_stride_len : int
            Stride length when generating consecutive
            time series windows.
        task_name : str
            The task that the dataset is used for. One of
            'forecasting', or  'imputation'.
        random_seed : int
            Random seed for reproducibility.
        file_path : str
            Random seed for reproducibility.
        """

        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon
        self.full_file_path_and_name = file_path
        self.data_split = data_split
        self.start_date = start_date
        self.end_date = end_date
        self.data_stride_len = data_stride_len
        self.task_name = task_name
        self.random_seed = random_seed
        self.encrypted_data = encrypted_data

        self.scaler = StandardScaler()

        self.columns_names = columns_names

        # Read data
        self._read_data()

    def _get_borders(self):
        n_train = 12 * 30 * 24  # 一天 24条数据，每小时一条，训练数据取：一年数据
        n_val = 4 * 30 * 24  # 4个月数据
        n_test = 4 * 30 * 24  # 4个月数据

        train_end = n_train
        val_end = n_train + n_val
        test_start = val_end - self.seq_len
        test_end = test_start + n_test + self.seq_len

        train = slice(0, train_end)
        test = slice(test_start, test_end)

        return train, test

    def _read_data(self):
        if self.encrypted_data:
            df = decrypt_file_to_pandas(self.full_file_path_and_name)
        else:
            df = pd.read_csv(self.full_file_path_and_name)

        if df is None:
            self.length_timeseries = 0
            print('df is None', self.full_file_path_and_name)
            return
        # 删除缺失值
        df.dropna(inplace=True)
        # 交易状态	1：正常交易 0：停牌, 删除未交易的数据
        if 'tradestatus' in df.columns:
            df = df[df['tradestatus'] == 1].copy()

        # 时间转换
        if 'date' in df.columns:
            df['date'] = pd.to_datetime(df['date'])
        if 'datetime' in df.columns:
            df['datetime'] = pd.to_datetime(df['datetime'])

        # 根据时间列排序
        sort_column = 'datetime' if 'datetime' in df.columns else 'date'
        df = df.sort_values(sort_column, ascending=True).reset_index(drop=True)
        # 获取指定列
        # date,open,high,low,close,volume,amount,turn
        # columns_name = [sort_column, 'open', 'high', 'low', 'close', 'volume', 'amount', 'turn']
        columns_name = [sort_column, 'open', 'high', 'low', 'close', 'volume', 'amount' ]
        columns_name = [sort_column, 'open', 'high', 'low', 'close', 'volume' ]
        columns_name = [sort_column ] + self.columns_names
        df = df[columns_name]

        # print('df', len(df), self.full_file_path_and_name)

        # 数据拆分训练集和测试集
        start_date_dt = pd.to_datetime(self.start_date)
        end_date_dt = pd.to_datetime(self.end_date)
        # 数据截取
        # 直接根据时间段截取， start_date 和 end_date
        train_data = df[ (df[sort_column] >= start_date_dt) & (df[sort_column] < end_date_dt) ].drop(columns=[sort_column])
        # 使用 infer_objects(copy=False) 方法推断并优化数据类型，避免复制数据以节省内存。
        # 使用 interpolate(method="cubic") 方法对数据进行三次样条插值，填补缺失值或平滑数据。
        train_data = train_data.infer_objects(copy=False).interpolate(method="cubic")
        if len(train_data)>0:
            self.scaler.fit(train_data.values)
            # print(self.scaler, self.scaler.mean_, self.scaler.scale_)
            # train_data = self.scaler.transform(train_data.values)
            train_data = train_data.values

            # 根据时间段截取， 开始位置为end_date所在数据的下标, 往前推 seq_len条记录（df 已经按照时间正序排好序）
            end_index = df[df[sort_column] >= end_date_dt].index[0] if not df[df[sort_column] == end_date_dt].empty else len(df)
            start_index = max(0, end_index - self.seq_len)
            test_data = df.iloc[start_index:].drop(columns=[sort_column] )
            if len(test_data):
                test_data = test_data.values
                # test_data = self.scaler.transform(test_data.values)

            # print('df', len(df), 'train_data', len(train_data), 'test_data', len(test_data), self.full_file_path_and_name)

            # df.drop(columns=[sort_column], inplace=True)
            # 将数据框中的列类型进行智能推断，优化存储类型，且不创建副本； 对数据框中的缺失值使用三次样条插值法进行填充
            # df = df.infer_objects(copy=False).interpolate(method="cubic")

            if self.data_split == "train":
                self.data = train_data
            elif self.data_split == "test":
                self.data = test_data

        else:
            self.data = train_data
        self.length_timeseries_original = df.shape[0]
        self.n_channels = df.shape[1] - 1
        self.length_timeseries = self.data.shape[0]

        # lens = (self.length_timeseries - self.seq_len - self.forecast_horizon) // self.data_stride_len + 1
        # print(f"length_timeseries: {self.length_timeseries}  seq_len: {self.seq_len}, forecast_horizon: {self.forecast_horizon}, data_stride_len: {self.data_stride_len}" )
        # print(f"{self.data_split} data shape: {self.data.shape}, lens: {lens}, file: {self.full_file_path_and_name}" )

    def __getitem__(self, index):
        seq_start = self.data_stride_len * index
        seq_end = seq_start + self.seq_len
        input_mask = np.ones(self.seq_len)

        if self.task_name == "forecasting":
            pred_end = seq_end + self.forecast_horizon

            if pred_end > self.length_timeseries:
                pred_end = self.length_timeseries
                seq_end = seq_end - self.forecast_horizon
                seq_start = seq_end - self.seq_len

            timeseries_temp = self.data[seq_start:seq_end, :]
            # timeseries_orig = self.scaler.inverse_transform(timeseries_temp)
            timeseries_orig = timeseries_temp
            forecast_temp = self.data[seq_end:pred_end, :]
            # forecast_orig = self.scaler.inverse_transform(forecast_temp)
            forecast_orig = forecast_temp

            timeseries = self.data[seq_start:seq_end, :].T
            forecast = self.data[seq_end:pred_end, :].T

            # return timeseries, forecast, input_mask, timeseries_orig.T, forecast_orig.T
            return MomentDataItem(
                timeseries=torch.tensor(timeseries, dtype=torch.float32),
                forecast=torch.tensor(forecast, dtype=torch.float32),
                input_mask=torch.tensor(input_mask, dtype=torch.float32),
                timeseries_orig=torch.tensor(timeseries_orig.T, dtype=torch.float32),
                forecast_orig=torch.tensor(forecast_orig.T, dtype=torch.float32),
            )

        elif self.task_name == "imputation":
            if seq_end > self.length_timeseries:
                seq_end = self.length_timeseries
                seq_end = seq_end - self.seq_len

            timeseries_temp = self.data[seq_start:seq_end, :]
            # timeseries_orig = self.scaler.inverse_transform(timeseries_temp)
            timeseries_orig = timeseries_temp

            timeseries = timeseries_temp.T

            # return timeseries, input_mask, timeseries_orig.T
            return MomentDataItem(
                timeseries=torch.tensor(timeseries, dtype=torch.float32),
                forecast=None,
                input_mask=torch.tensor(input_mask, dtype=torch.float32),
                timeseries_orig=torch.tensor(timeseries_orig.T, dtype=torch.float32),
                forecast_orig=None,
            )

    def __len__(self):
        if self.task_name == "imputation":
            lens_ = (self.length_timeseries - self.seq_len) // self.data_stride_len + 1
            if lens_ > 0:
                return lens_
            else:
                return 0
        elif self.task_name == "forecasting":
            lens_ = (self.length_timeseries - self.seq_len - self.forecast_horizon) // self.data_stride_len + 1
            if lens_ > 0:
                return lens_
            else:
                return 0


def check_file(daily_file: str, min_lines: int=1000, encrypted_data=False):
    """
    检查文件的行数是否达到最小要求

    Args:
        daily_file (str): 文件路径
        min_lines (int): 最小行数要求，默认为1000行

    Returns:
        bool: 如果文件行数大于等于最小要求返回True，否则返回False
    """
    try:
        if encrypted_data:
            df = decrypt_file_to_pandas(daily_file)
            return len(df) >= min_lines
        with open(daily_file, 'r', encoding='utf-8') as f:
            line_count = sum(1 for _ in f)
        return line_count >= min_lines
    except Exception as e:
        print(f"检查文件 {daily_file} 时发生错误: {str(e)}")
        return False


@dataclass
class MomentDataItem:
    """MOMENT数据项"""
    timeseries: torch.Tensor
    forecast: torch.Tensor
    input_mask: torch.Tensor
    timeseries_orig: torch.Tensor = None
    forecast_orig: torch.Tensor = None


class MomentDataset(Dataset):
    """MOMENT数据集"""

    def __init__(self, data_split: str,
                 train_data_path: str,
                 data_type: str,
                 random_seed: int,
                 seq_len: int,
                 forecast_horizon: int,
                 data_stride_len: int = 1,
                 start_date="2010-01-01",
                 end_date="2024-11-01",
                 encrypted_data=True,
                 limit=1000000,
                 ):
        """初始化数据集

        Args:
            data_split: 数据集划分(train/test)

        """
        self.data_split = data_split
        self.seq_len = seq_len
        self.forecast_horizon = forecast_horizon

        self.train_data_path = train_data_path
        self.data_type = data_type
        self.random_seed = random_seed

        self.start_date = start_date
        self.end_date = end_date
        self.data_stride_len = data_stride_len
        self.encrypted_data = encrypted_data
        self.limit = limit

        # 初始化数据集列表
        self.datasets: List[InformerDataset] = []

        # 根据数据类型加载不同的数据集
        if os.path.isdir(self.train_data_path):
            if self.data_type == 'min60':
                self._load_min60_datasets()
            elif self.data_type == 'daily':
                self._load_daily_datasets()
            elif self.data_type == 'weekly':
                self._load_weekly_datasets()

            # 合并所有数据集
            self.dataset = torch.utils.data.ConcatDataset(self.datasets)
            self.len = sum(len(dataset) for dataset in self.datasets)
        else:
            raise ValueError(f"训练数据路径 {self.train_data_path} 不存在")

    def _load_min60_datasets(self):
        """加载60分钟级别的数据集"""
        for i, file_name in enumerate(os.listdir(self.train_data_path)):
            if i >= self.limit:
                break
            if self.encrypted_data:
                min60_file = os.path.join(self.train_data_path, f'{file_name}/min60.csv.enc')
            else:
                min60_file = os.path.join(self.train_data_path, f'{file_name}/min60.csv')
            if os.path.exists(min60_file) and check_file(min60_file, min_lines=self.seq_len+self.forecast_horizon, encrypted_data=self.encrypted_data):
                dataset = InformerDataset(
                    seq_len=self.seq_len,
                    data_split=self.data_split,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    random_seed=self.random_seed,
                    forecast_horizon=self.forecast_horizon,
                    file_path=min60_file,
                    encrypted_data=self.encrypted_data,
                    data_stride_len=3,
                    columns_names=['open', 'high', 'low', 'close', 'volume', 'amount']
                )
                if len(dataset) > 0:
                    self.datasets.append(dataset)
                
    def _load_daily_datasets(self):
        """加载日线级别的数据集"""
        for i, file_name in enumerate(os.listdir(self.train_data_path)):
            if i >= self.limit:
                break
            if self.encrypted_data:
                daily_file = os.path.join(self.train_data_path, f'{file_name}/daily.csv.enc')
            else:
                daily_file = os.path.join(self.train_data_path, f'{file_name}/daily.csv')
            if os.path.exists(daily_file) and check_file(daily_file, min_lines=self.seq_len+self.forecast_horizon, encrypted_data=self.encrypted_data):
                dataset = InformerDataset(
                    seq_len=self.seq_len,
                    data_split=self.data_split,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    random_seed=self.random_seed,
                    forecast_horizon=self.forecast_horizon,
                    file_path=daily_file,
                    encrypted_data=self.encrypted_data,
                    data_stride_len=3,
                    columns_names=['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM','pcfNcfTTM']
                    # columns_names=['open', 'high', 'low', 'close', 'turn', 'pctChg', 'peTTM', 'pbMRQ', 'psTTM']
                )
                if len(dataset) > 0:
                    self.datasets.append(dataset)
                
    def _load_weekly_datasets(self):
        """加载周线级别的数据集"""
        for i, file_name in enumerate(os.listdir(self.train_data_path)):
            if i >= self.limit:
                break
            if self.encrypted_data:
                weekly_file = os.path.join(self.train_data_path, f'{file_name}/weekly.csv.enc')
            else:
                weekly_file = os.path.join(self.train_data_path, f'{file_name}/weekly.csv')
            if os.path.exists(weekly_file) and check_file(weekly_file, min_lines=self.seq_len+self.forecast_horizon, encrypted_data=self.encrypted_data):
                dataset = InformerDataset(
                    seq_len=self.seq_len,
                    data_split=self.data_split,
                    start_date=self.start_date,
                    end_date=self.end_date,
                    random_seed=self.random_seed,
                    forecast_horizon=self.forecast_horizon,
                    file_path=weekly_file,
                    encrypted_data=self.encrypted_data,
                    data_stride_len=1,
                    columns_names=['open', 'high', 'low', 'close', 'volume', 'amount', 'turn', 'pctChg']
                )
                if len(dataset) > 0:
                    self.datasets.append(dataset)

    @staticmethod
    def from_params2ds(data_split: str = 'train',
                       train_data_path: str = None,
                       data_type: str = '',
                       random_seed: int = 13,
                       seq_len: int = 512,
                       forecast_horizon: int = 64,
                       data_stride_len: int = 1,
                       encrypted_data: bool = False,
                       limit: int = 100000,
                       start_date="2022-01-01",
                       end_date="2024-11-01",
                       ) -> Dataset:
        return MomentDataset(data_split=data_split,
                             train_data_path=train_data_path,
                             data_type=data_type,
                             random_seed=random_seed,
                             seq_len=seq_len,
                             forecast_horizon=forecast_horizon,
                             data_stride_len=data_stride_len,
                             start_date=start_date,
                             end_date=end_date,
                             encrypted_data=encrypted_data,
                             limit=limit,
                             )

    def __len__(self):
        """返回数据集长度"""
        return self.len
        
    def __getitem__(self, idx):
        """获取数据项
        
        Args:
            idx: 索引
            
        Returns:
            数据项
        """
        # timeseries, forecast, input_mask, timeseries_orig, forecast_orig = self.dataset[idx]
        # return MomentDataItem(
        #     timeseries=torch.tensor(timeseries, dtype=torch.float32),
        #     forecast=torch.tensor(forecast, dtype=torch.float32),
        #     input_mask=torch.tensor(input_mask, dtype=torch.float32),
        #     timeseries_orig=torch.tensor(timeseries_orig, dtype=torch.float32),
        #     forecast_orig=torch.tensor(forecast_orig, dtype=torch.float32),
        # )
        return self.dataset[idx]


    @staticmethod
    def get_data_sets(data_args: DataArguments, seq_len: int, forecast_horizon: int) -> tuple[Dataset, Dataset]:
        """获取数据集

        Args:
            data_args: 数据参数
            model_args: 模型参数

        Returns:
            tuple[Dataset, Dataset]: 训练和测试数据集
        """
        # 训练数据
        train_dataset = MomentDataset(data_split="train",
                                      train_data_path=data_args.train_data_path,
                                      data_type=data_args.data_type,
                                      random_seed=data_args.random_seed,
                                      seq_len=seq_len,
                                      forecast_horizon=forecast_horizon,
                                      data_stride_len=data_args.train_data_stride_len,
                                      start_date=data_args.start_date,
                                      end_date=data_args.end_date,
                                      encrypted_data=data_args.encrypted_data,
                                      limit=data_args.limit,
                                      )

        # 测试数据
        test_dataset = MomentDataset(data_split="test",
                                     train_data_path=data_args.train_data_path,
                                     data_type=data_args.data_type,
                                     random_seed=data_args.random_seed,
                                     seq_len=seq_len,
                                     forecast_horizon=forecast_horizon,
                                     data_stride_len=data_args.test_data_stride_len,
                                     start_date=data_args.start_date,
                                     end_date=data_args.end_date,
                                     encrypted_data=data_args.encrypted_data,
                                     limit=data_args.limit,

                                     )

        return train_dataset, test_dataset

    @staticmethod
    def get_informer_datasets(data_args: DataArguments, seq_len: int, forecast_horizon: int) -> tuple[List[InformerDataset], List[InformerDataset]]:
        train_dataset, test_dataset = MomentDataset.get_data_sets(data_args, seq_len, forecast_horizon)
        train_dataset: MomentDataset
        test_dataset: MomentDataset
        return train_dataset.datasets, test_dataset.datasets


class MomentDataCollator(DataCollatorMixin):
    """MOMENT数据整理器"""

    def __init__(self, return_tensors: str = "pt"):
        """初始化数据整理器

        Args:
            return_tensors: 返回张量的类型，默认为"pt"（PyTorch）
        """
        super().__init__()
        self.return_tensors = return_tensors

    def torch_call(self, features):
        """整理数据批次

        Args:
            features: 数据特征列表

        Returns:
            整理后的数据批次
        """
        batch = {
            'timeseries': torch.stack([f.timeseries for f in features]),
            'forecast': torch.stack([f.forecast for f in features]),
            'input_mask': torch.stack([f.input_mask for f in features]),
            'timeseries_orig': torch.stack([f.timeseries_orig for f in features]),
            'forecast_orig': torch.stack([f.forecast_orig for f in features]),
        }
        return batch

