# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This example shows how a TrainerRM works based on TaskManager with rolling tasks.
After training, how to collect the rolling results will be shown in task_collecting.
Based on the ability of TaskManager, `worker` method offer a simple way for multiprocessing.
"""
from qlib.workflow.task.gen import RollingGen, task_generator

from qlib_.train_test.data_train import load_config_yaml

from qlib_.train_test.rolling_train_tree.task_manager_rolling import RollingTaskExample

if __name__ == "__main__":
    ## to see the whole process with your own parameters, use the command below
    # python task_manager_rolling.py main --experiment_name="your_exp_name"


    config_task_exps = [
        ("./config/csi300/workflow_config_lgb_Alpha158_all.yaml", 'rolling_exp_tree_all', None),
        # ("./config/csi300/workflow_config_lgb_Alpha158_rec_tree.yaml", 'rolling_exp_rec_tree', None),
        # ("./config/csi300/workflow_config_lgb_Alpha158_rec_tree1.yaml", 'rolling_exp_rec_tree1', None),
        # ("./config/csi300/workflow_config_lgb_Alpha158_tree_import.yaml", 'rolling_exp_tree_import', None),
    ]
    # rolling_types = [RollingGen.ROLL_EX, RollingGen.ROLL_SD]
    # ROLL_EX 效果优于 ROLL_SD
    rolling_types = [RollingGen.ROLL_EX]

    for config_path, task_exp1, feature_task_config in config_task_exps:
        for rolling_type in rolling_types:
            print(f"========== {task_exp1}_{rolling_type} ==========")
            cfg = load_config_yaml(config_path=config_path)
            task_exp = f'{task_exp1}_{rolling_type}'

            RollingTaskExample(
                task_config=[cfg["task"] ],
                feature_task_config=feature_task_config,
                task_pool=task_exp,
                experiment_name=task_exp,
                task_db_name='class_rolling_db',
                rolling_step=21,
                rolling_type=rolling_type
            ).main()


    '''
    cfg = load_config_yaml(config_path="./config/workflow_config_lgb_Alpha158_tree_import.yaml")
    task_exp = 'rolling_exp_tree_import'

    cfg = load_config_yaml(config_path="./config/workflow_config_lgb_Alpha158_all.yaml")
    task_exp = 'rolling_exp_tree_all'

    RollingTaskExample(
        task_config=[cfg["task"] ],
        task_pool=task_exp,
        experiment_name=task_exp,
        rolling_step=21,
        rolling_type=RollingGen.ROLL_EX
        # rolling_type=RollingGen.ROLL_SD
    ).main()
    '''

