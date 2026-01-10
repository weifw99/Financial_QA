
import os
import pickle
from datetime import datetime, date
from pathlib import Path
import json
from typing import Tuple, Any

import yaml
import qlib
import qlib
from qlib.config import REG_CN, C
from qlib.utils import init_instance_by_config

from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord, SigAnaRecord


def init_qlib(config_path='') -> tuple[Any, Path, str]:
    """
    qlib init
    返回配置文件、工作目录、env_name
    :param config_path:
    :return:
    """
    # 1. 读取配置
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. 初始化 Qlib
    qlib_init = cfg.get("qlib_init", {})

    exp_name = qlib_init.get("exp_manager").get("kwargs").get('default_exp_name')
    base_dir = qlib_init.get("exp_manager").get("kwargs").get('uri').replace('file:', '')
    base_dir_path = Path(base_dir)
    work_dir = base_dir_path.parent
    print(f'工作目录：{work_dir}')
    # print(work_dir)
    if not work_dir.exists():
        os.makedirs(work_dir)

    if not C.registered:
        print("初始化 Qlib...")
        qlib.init(
            provider_uri=qlib_init.get("provider_uri", "~/.qlib/qlib_data/cn_data"),
            region=REG_CN if qlib_init.get("region") == "cn" else None,
            exp_manager=qlib_init.get("exp_manager", {
                "class": "MLflowExpManager",
                "module_path": "qlib.workflow.expm",
                "kwargs": {
                    "uri": "file:" + str(Path(os.getcwd()).resolve() / "mlruns"),
                    "default_exp_name": "Experiment",
                },
            })
        )
    return cfg, work_dir, exp_name


def tree_frature_import_alpha1581_by_task(config_path='', task_info=None):
    cfg, work_dir, exp_name = init_qlib(config_path)
    # 基于 task info 修改 配置
    # 修改 内容： segment

    cfg["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["start_time"] = task_info["dataset"]["kwargs"]["handler"]["kwargs"]["start_time"]
    cfg["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"] = task_info["dataset"]["kwargs"]["handler"]["kwargs"]["end_time"]
    cfg["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["fit_start_time"] = task_info["dataset"]["kwargs"]["handler"]["kwargs"]["fit_start_time"]
    cfg["task"]["dataset"]["kwargs"]["handler"]["kwargs"]["fit_end_time"] = task_info["dataset"]["kwargs"]["handler"]["kwargs"]["fit_end_time"]

    cfg["task"]['dataset']['kwargs']['segments'] = task_info["dataset"]["kwargs"]["segments"]
    import pandas as pd

    # 设置训练时间，为三年， 设置valid 默认是两个月，扩充到四个月
    cfg["task"]['dataset']['kwargs']['segments']['train'] = ( task_info['dataset']['kwargs']['segments']['train'][1] - pd.DateOffset(years=2, months=2), task_info['dataset']['kwargs']['segments']['train'][1] - pd.DateOffset(months=2) )
    cfg["task"]['dataset']['kwargs']['segments']['valid'] = ( task_info['dataset']['kwargs']['segments']['valid'][0] - pd.DateOffset(months=2), task_info['dataset']['kwargs']['segments']['valid'][1])
    cfg["task"]['dataset']['kwargs']['segments']['test'] = ( task_info['dataset']['kwargs']['segments']['test'][0], task_info['dataset']['kwargs']['segments']['test'][1])

    print(f"[EDIT] config feature import {cfg["task"]}")
    # task['record'][2]['kwargs']['config']['backtest']['start_time'] = task['dataset']['kwargs']['segments']['test'][0]
    # task['record'][2]['kwargs']['config']['backtest']['end_time'] = task['dataset']['kwargs']['segments']['test'][1]
    # task['dataset']['kwargs']['handler']['kwargs']['fit_start_time'] = task['dataset']['kwargs']['segments']['train'][0]
    # task['dataset']['kwargs']['handler']['kwargs']['fit_end_time'] = task['dataset']['kwargs']['segments']['train'][1]

    return tree_frature_import_alpha158(cfg, work_dir, exp_name)


def tree_frature_import_alpha158_by_config(config_path=''):
    cfg, work_dir, exp_name = init_qlib(config_path)
    return tree_frature_import_alpha158(cfg, work_dir, exp_name)


def tree_frature_import_alpha158(cfg, work_dir, exp_name):
    # cfg, work_dir, exp_name = init_qlib(config_path)

    # 3. 初始化 MTSDatasetH
    dataset_cfg = cfg["task"]["dataset"]
    dataset = init_instance_by_config(dataset_cfg)

    port_analysis_config = cfg['port_analysis_config']
    # 初始化 model
    model_cfg = cfg["task"]["model"]
    print(f"初始化 model: {model_cfg}")
    model = init_instance_by_config(model_cfg)

    ''' '''
    # experiment_name = "workflow"
    recorder_info_file = f"{work_dir}/{exp_name}_recorder_info_{datetime.now().strftime('%Y-%m-%d')}.json"
    # start exp
    with R.start(experiment_name=exp_name) as rec:
        print("当前 record_id:", rec.id)  # ✅ record_id 就在这里
        # 当前的工作 record
        active_recorder = rec.active_recorder
        # 保存 info 到 JSON 文件
        with open(recorder_info_file, "w", encoding="utf-8") as f:
            json.dump(active_recorder.info, f, indent=4, ensure_ascii=False)
        print(f"active_recorder_info: {json.dumps(active_recorder.info, indent=4)}")

        # train model
        R.log_params(**flatten_dict(cfg["task"]))
        model.fit(dataset)
        print("model:", model)

        recorder = R.get_recorder(recorder_id=active_recorder.id, experiment_id=active_recorder.experiment_id)
        # recorder1 = R.get_recorder()
        # ✅ 显式保存模型
        recorder.save_objects(model=model)

        # 预测 + 评估
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # 2️⃣ 保存 IC / ICIR 等指标（关键）
        SigAnaRecord(
            recorder=recorder,
            ana_long_short=False,
            ann_scaler=252,
        ).generate()

    if not isinstance(model, LGBModel):
        return None

    ds_feature_names = dataset.handler.get_cols()
    print("dataset feature_names:", ds_feature_names)
    model: LGBModel = model
    # 获取特征名称
    feature_names = model.model.feature_name()

    print("feature_names len:", len(feature_names))
    print("ds_feature_names len:", len(ds_feature_names))
    # 提取 Column_x 中的数字并映射到真实特征名
    real_feature_names = []
    for col_name in feature_names:
        index = int(col_name.replace('Column_', ''))
        real_feature_names.append(ds_feature_names[index])
    # 获取基于增益的重要性
    importance = model.model.feature_importance(importance_type='gain')

    # 创建特征重要性 DataFrame
    import pandas as pd
    importance_df = pd.DataFrame({
        'feature': real_feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    print(importance_df)
    # importance_df.to_csv(f"{work_dir}/exp_name_feature_importance.csv")
    return importance_df


def train_model_alpha158(config_path=''):

    cfg, work_dir, exp_name = init_qlib(config_path)

    # 3. 初始化 MTSDatasetH
    dataset_cfg = cfg["task"]["dataset"]
    dataset = init_instance_by_config(dataset_cfg)

    port_analysis_config = cfg['port_analysis_config']
    # 初始化 model
    model_cfg = cfg["task"]["model"]
    print(f"初始化 model: {model_cfg}")
    model = init_instance_by_config(model_cfg)

    ''' '''
    # experiment_name = "workflow"
    recorder_info_file = f"{work_dir}/{exp_name}_recorder_info_{datetime.now().strftime('%Y-%m-%d')}.json"
    # start exp
    train_model = None
    with R.start(experiment_name=exp_name) as rec:
        print("当前 record_id:", rec.id)  # ✅ record_id 就在这里
        # 当前的工作 record
        active_recorder = rec.active_recorder
        # 保存 info 到 JSON 文件
        with open(recorder_info_file, "w", encoding="utf-8") as f:
            json.dump(active_recorder.info, f, indent=4, ensure_ascii=False)
        print(f"active_recorder_info: {json.dumps(active_recorder.info, indent=4)}" )

        # train model
        R.log_params(**flatten_dict(cfg["task"]))
        model.fit(dataset)
        print("model:", model)
        train_model = model

        recorder = R.get_recorder(recorder_id=active_recorder.id, experiment_id=active_recorder.experiment_id)
        # recorder1 = R.get_recorder()
        # ✅ 显式保存模型
        recorder.save_objects(model=model)

        # 预测 + 评估
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # 2️⃣ 保存 IC / ICIR 等指标（关键）
        SigAnaRecord(
            recorder=recorder,
            ana_long_short=False,
            ann_scaler=252,
        ).generate()

        # 测试
        '''
        PortAnaRecord(
            recorder=recorder,
            config=port_analysis_config,
        ).generate()
        '''


def predict_data_model(config_path='', recorder_file=None):
    cfg, work_dir, exp_name = init_qlib(config_path)
    if recorder_file:
        recorder_info_file = f"{work_dir}/{recorder_file}"
    else:
        recorder_info_file = f"{work_dir}/{exp_name}_recorder_info_{datetime.now().strftime('%Y-%m-%d')}.json"
    # 加载阶段
    try:
        with open(recorder_info_file, "r", encoding="utf-8") as f:
            loaded_info_text = json.load(f)
        print("从文件加载的 recorder_info 内容:")
    except FileNotFoundError:
        print("文件不存在，请先运行保存部分的代码")

    # 加载 recorder_info
    recorder_info_str = json.dumps(loaded_info_text, indent=4)
    recorder_info = json.loads(recorder_info_str)
    print(type(recorder_info), recorder_info)

    # 获取 recorder, 加载模型
    recorder = R.get_recorder(recorder_id=recorder_info['id'], experiment_id=recorder_info['experiment_id'])
    model = recorder.load_object("model")
    model._writer = None  # 🔥 关键, 避免_writer需要在 train 阶段初始化，直接 predict 失败
    print("model:", model)

    # 创建 dataset 初始化 MTSDatasetH
    dataset_cfg = cfg["task"]["dataset"]
    print(f"初始化 MTSDatasetH, 配置: {dataset_cfg}")

    # dataset_cfg['kwargs']['handler']['kwargs']['start_time'] = date(2021, 1, 1)
    # dataset_cfg['kwargs']['handler']['kwargs']['end_time'] = date(2025, 5, 15)
    #
    # dataset_cfg['kwargs']['segments'] = {
    #     'test': [date(2025, 1, 1), date(2025, 5, 15)]
    # }

    print(f"修改后 MTSDatasetH, 配置: {dataset_cfg}")
    # 对 dataset_cfg 进行修改
    # 1. 修改 end_time  'end_time': datetime.date(2025, 5, 15)
    #  保持不变 fit_start_time: 2021-01-01  fit_end_time:   2024-12-31
    # 2.
    dataset = init_instance_by_config(dataset_cfg)

    print("dataset:", dataset)
    ds_feature_names = dataset.handler.get_cols()
    print("dataset feature_names:", ds_feature_names)

    SignalRecord(
        model=model,
        dataset=dataset,
        recorder=recorder
    ).generate()

    pred = recorder.load_object("pred.pkl")

    # 重置索引，将 datetime 和 instrument 变为普通列
    df_normal = pred.reset_index()
    print("pred type:", type(df_normal))
    print("pred:", df_normal)


    if isinstance(model, LGBModel):
        model: LGBModel = model
        # 获取特征名称
        feature_names = model.model.feature_name()

        print("feature_names len:", len(feature_names))
        print("ds_feature_names len:", len(ds_feature_names))
        # 提取 Column_x 中的数字并映射到真实特征名
        real_feature_names = []
        for col_name in feature_names:
            index = int(col_name.replace('Column_', ''))
            real_feature_names.append(ds_feature_names[index])
        # 获取基于增益的重要性
        importance = model.model.feature_importance(importance_type='gain')

        # 创建特征重要性 DataFrame
        import pandas as pd
        importance_df = pd.DataFrame({
            'feature': real_feature_names,
            'importance': importance
        }).sort_values('importance', ascending=False)

        print(importance_df)

        importance_df.to_csv(f"{work_dir}/exp_name_feature_importance.csv")








if __name__ == "__main__":
    config_path = "config/import_c/csi300_lgb_Alpha158_all.yaml"
    config_path = "config/import_c/zxzz399101_lgb_Alpha158_all.yaml"
    train_model_alpha158(config_path=config_path)
    predict_data_model(config_path=config_path)

    # predict_data_model(config_path=config_path, recorder_file='recorder_info_2025-12-25.json')
    # predict_data_model(config_path=config_path, recorder_file='lgb_exp_all_recorder_info_2025-12-26.json')




