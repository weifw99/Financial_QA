from qlib.data.dataset import Dataset
from qlib.utils import init_instance_by_config
from qlib.contrib.data.handler import DataHandlerLP

from qlib_.data.data_handler.alpha158 import Alpha158
import pandas as pd


def initialize_data_handler(data_handler_config: dict) -> DataHandlerLP:
    """
    :param data_handler_config:
        start_time: 2008-01-01
        end_time: 2020-08-01
        fit_start_time: 2008-01-01
        fit_end_time: 2014-12-31
        instruments: *market
    :return:
    """
    return Alpha158(**data_handler_config)


def load_dataset(config, handler) -> (Dataset, pd.DataFrame):
    dataset_conf = config["task"]["dataset"]
    dataset_conf['kwargs']['handler'] = handler
    dataset: Dataset = init_instance_by_config(dataset_conf)
    # Ensure fetching in the correct format
    fetched_data = handler.fetch()
    return dataset, fetched_data.reset_index()


def load_dataset_by_config(config:dict) -> (Dataset, pd.DataFrame):
    handler = initialize_data_handler(config["data_handler_config"])
    dataset_conf = config["task"]["dataset"]
    dataset_conf['kwargs']['handler'] = handler
    dataset = init_instance_by_config(dataset_conf)
    # Ensure fetching in the correct format
    fetched_data = handler.fetch()
    return dataset, fetched_data.reset_index()