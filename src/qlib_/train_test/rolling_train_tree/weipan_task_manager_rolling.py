# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This example shows how a TrainerRM works based on TaskManager with rolling tasks.
After training, how to collect the rolling results will be shown in task_collecting.
Based on the ability of TaskManager, `worker` method offer a simple way for multiprocessing.
"""
import json
import os
from pprint import pprint
from typing import List

import fire
import pandas as pd
import qlib
from qlib.backtest import backtest
from qlib.constant import REG_CN
from qlib.contrib.evaluate import risk_analysis, indicator_analysis
from qlib.contrib.model import LGBModel
from qlib.model.ens.ensemble import RollingEnsemble
from qlib.utils import fill_placeholder, get_date_by_shift
from qlib.utils.time import Freq
from qlib.workflow import R, Recorder
from qlib.workflow.record_temp import SigAnaRecord, PortAnaRecord
from qlib.workflow.task.gen import RollingGen, task_generator
from qlib.workflow.task.manage import TaskManager, run_task
from qlib.workflow.task.collect import RecorderCollector
from qlib.model.trainer import TrainerR, TrainerRM, task_train
from qlib.tests.config import CSI100_RECORD_LGB_TASK_CONFIG, CSI100_RECORD_XGBOOST_TASK_CONFIG

from qlib_.train_test.data_train import load_config_yaml

import gc

from qlib_.train_test.rolling_train_tree.export_feature_import import tree_frature_import_alpha1581_by_task
from qlib_.train_test.rolling_train_tree.qlib_html_report import generate_html_report
from qlib_.train_test.rolling_train_tree.task_manager_rolling import RollingTaskExample

if __name__ == "__main__":
    ## to see the whole process with your own parameters, use the command below
    # python task_manager_rolling.py main --experiment_name="your_exp_name"

    config_task_exps = [
        ("./config/zxzz399101_1/workflow_config_lgb_Alpha158_tree_import.yaml", 'wpgbk1158_tree_import_test', None),
        ("./config/zxzz399101_1/workflow_config_lgb_Alpha158_all.yaml", 'wpgbk1158_tree_all_test', None),
        ("./config/zxzz399101_1/workflow_config_lgb_Alpha158_rec_tree.yaml", 'wpgbk1158_rec_tree_test', None),
        ("./config/zxzz399101_1/workflow_config_lgb_Alpha158_rec_tree1.yaml", 'wpgbk1158_tree_select1_test', None),
    ]
    rolling_types = [RollingGen.ROLL_EX, RollingGen.ROLL_SD]
    # ROLL_EX 效果优于 ROLL_SD
    rolling_types = [RollingGen.ROLL_EX]
    # rolling_types = [RollingGen.ROLL_SD]

    for config_path, task_exp1, feature_task_config in config_task_exps:
        for rolling_type in rolling_types:
            print(f"========== {task_exp1}_{rolling_type} ==========")
            cfg = load_config_yaml(config_path=config_path)
            task_exp = f'{task_exp1}_{rolling_type}_wpgbk1158'

            cfg['market'] = 'wpgbk1158'
            cfg['data_handler_config']['instruments'] = 'wpgbk1158'


            RollingTaskExample(
                provider_uri=cfg['qlib_init']['provider_uri'],
                region=REG_CN,
                task_config=[cfg["task"] ],
                feature_task_config=feature_task_config,
                task_pool=task_exp,
                experiment_name=task_exp,
                rolling_step=21,
                # rolling_step=5,
                rolling_type=rolling_type,
                ouput_dir="data/wpgbk1158",
            ).main()




