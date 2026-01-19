from typing import Union, List, Tuple, Dict, Text
from copy import copy, deepcopy
from inspect import getfullargspec
import pandas as pd
from qlib.data.dataset import DatasetH, DataHandler, DataHandlerLP

from sklearn.model_selection import train_test_split

class CustomDatasetH(DatasetH):
    """
    自定义 DatasetH，支持原有时间段切分，也支持按比例切分 train/valid

    dataset_config = {
        "class": "CustomDatasetH",
        "module_path": "__main__",  # 如果在脚本里定义
        "kwargs": {
            "handler": {
                "class": "DataHandlerLP",
                "module_path": "qlib.data.dataset.handler",
                "kwargs": {
                    "instruments": "csi300",
                    "start_time": "2015-01-01",
                    "end_time": "2022-12-31",
                    "fit_start_time": "2015-01-01",
                    "fit_end_time": "2020-12-31",
                },
            },
            "segments": {
                "train": ("2015-01-01", "2019-12-31"),
                "valid": ("2020-01-01", "2020-12-31"),  # ratio 模式下无效
                "test":  ("2021-01-01", "2022-12-31"),
            },
            "dataset_type": "ratio",
            "train_ratio": 0.8,
        },
    }
    """

    def __init__(
        self,
        handler: Union[Dict, DataHandler],
        segments: Dict[Text, Tuple],
        fetch_kwargs: Dict = {},
        dataset_type: str = "time",   # 'time' 或 'ratio'
        train_ratio: float = 0.8,     # 仅在 ratio 模式下有效
        **kwargs,
    ):
        self.dataset_type = dataset_type
        self.train_ratio = train_ratio
        super().__init__(handler=handler, segments=segments, fetch_kwargs=fetch_kwargs, **kwargs)
        self.train_data = None
        self.valid_data = None

    def prepare(self,
                segments: Union[List[Text], Tuple[Text], Text, slice, pd.Index],
                col_set=DataHandler.CS_ALL,
                data_key=DataHandlerLP.DK_I,
                **kwargs) -> Union[List[pd.DataFrame], pd.DataFrame]:
        """
        按比例切分训练集/验证集逻辑：
        - dataset_type='ratio' 时：
            - 'train' segment 内部分割 train/valid
            - 'valid' segment 配置无效
            - 'test' segment 不变
        """
        if self.dataset_type == "ratio" and (segments == "train"
                                             or segments == "valid"
                                             or (isinstance(segments, list) and "train" in segments)
                                             or (isinstance(segments, list) and "valid" in segments)
        ):
            if self.train_data is None or self.valid_data is None:
                # 1. 获取原始 train 数据
                raw_train = super().prepare("train", col_set=col_set, data_key=data_key, **kwargs)
                # 2. 按比例切分
                X_train, X_valid = train_test_split(raw_train, train_size=self.train_ratio, shuffle=True, random_state=42)
                self.train_data = X_train
                self.valid_data = X_valid
            if segments == "train":
                return self.train_data
            elif segments == "valid":
                return self.valid_data

        # 对其他 segment 依然使用原有逻辑
        return super().prepare(segments, col_set=col_set, data_key=data_key, **kwargs)
