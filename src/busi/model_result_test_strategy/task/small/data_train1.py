
import os
import pickle
from datetime import datetime, date, timedelta
from pathlib import Path
import json
from typing import Tuple, Any

import pandas as pd
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


def train_model_alpha158(config_path='', data_date=None):

    cfg, work_dir, exp_name = init_qlib(config_path)

    # 3. 初始化 MTSDatasetH
    dataset_cfg = cfg["task"]["dataset"]

    # 更新 dataset_cfg
    # 根据数据截止时间，设置数据集的 start_time 和 end_time
    # 修改 train、valid、test 时间
    if data_date:
        segments1 = {
            'train': [dataset_cfg['kwargs']['segments']['train'][0], now - timedelta(days=61)],
            'valid': [now - timedelta(days=60), now],
            'test': [now - timedelta(days=15), now],
        }
        dataset_cfg['kwargs']['handler']['kwargs']['end_time'] = now
        dataset_cfg['kwargs']['handler']['kwargs']['fit_start_time'] = segments1['train'][0]
        dataset_cfg['kwargs']['handler']['kwargs']['fit_end_time'] = segments1['train'][1]

        dataset_cfg['kwargs']['segments'] = segments1

    dataset = init_instance_by_config(dataset_cfg)

    port_analysis_config = cfg['port_analysis_config']
    # 初始化 model
    model_cfg = cfg["task"]["model"]
    print(f"初始化 model: {model_cfg}")
    model = init_instance_by_config(model_cfg)

    ''' '''
    # experiment_name = "workflow"
    # recorder_info_file = f"{work_dir}/{exp_name}_recorder_info_{datetime.now().strftime('%Y-%m-%d')}.json"
    # start exp
    train_model = None
    active_recorder = None
    with R.start(experiment_name=exp_name) as rec:
        print("当前 record_id:", rec.id)  # ✅ record_id 就在这里
        # 当前的工作 record
        active_recorder = rec.active_recorder
        # 保存 info 到 JSON 文件
        # with open(recorder_info_file, "w", encoding="utf-8") as f:
        #     json.dump(active_recorder.info, f, indent=4, ensure_ascii=False)
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

        '''
        # 2️⃣ 保存 IC / ICIR 等指标（关键）
        SigAnaRecord(
            recorder=recorder,
            ana_long_short=False,
            ann_scaler=252,
        ).generate()

        # 测试
        
        PortAnaRecord(
            recorder=recorder,
            config=port_analysis_config,
        ).generate()
        '''
    return active_recorder


def predict_data_model(config_path='', recorder_info=None):
    cfg, work_dir, exp_name = init_qlib(config_path)
    # 加载阶段
    print(type(recorder_info), recorder_info)

    # 获取 recorder, 加载模型
    recorder = R.get_recorder(recorder_id=recorder_info['id'], experiment_id=recorder_info['experiment_id'])
    model = recorder.load_object("model")
    model._writer = None  # 🔥 关键, 避免_writer需要在 train 阶段初始化，直接 predict 失败
    print("model:", model)

    # 创建 dataset 初始化 MTSDatasetH
    dataset_cfg = cfg["task"]["dataset"]
    print(f"初始化 MTSDatasetH, 配置: {dataset_cfg}")
    # 获取模型训练时候的参数，修改数据集的 time
    params = recorder.list_params()

    now = datetime.now().date()

    dataset_cfg['kwargs']['handler']['kwargs']['start_time'] = params['dataset.kwargs.handler.kwargs.start_time']
    dataset_cfg['kwargs']['handler']['kwargs']['end_time'] = now
    dataset_cfg['kwargs']['handler']['kwargs']['fit_start_time'] = params['dataset.kwargs.handler.kwargs.fit_start_time']
    dataset_cfg['kwargs']['handler']['kwargs']['fit_end_time'] = params['dataset.kwargs.handler.kwargs.fit_end_time']

    dataset_cfg['kwargs']['segments'] = {
        'test': [now - timedelta(days=15), now],
        'train': params['dataset.kwargs.segments.train'],
        'valid': params['dataset.kwargs.segments.valid'],
    }

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

    return df_normal










if __name__ == "__main__":

    config_paths = [
        "config/zxzz399101/workflow_config_lgb_Alpha158_all.yaml",
        "config/zxzz399101/workflow_config_lgb_Alpha158_rec_tree.yaml",
        # "config/zxzz399101/workflow_config_lgb_Alpha158_rec_tree1.yaml",
        # "config/zxzz399101/workflow_config_lgb_Alpha158_tree_import.yaml",
    ]

    # 获取当前时间
    now = datetime.now().date()
    weekday = now.weekday()
    work_dir = "/Users/dabai/liepin/study/llm/Financial_QA/data/qlib_exp"
    busi_name = "small"

    base_dir = f"{work_dir}/{busi_name}"

    if not os.path.exists(base_dir):
        os.makedirs(base_dir, exist_ok=True)

    recorder_info_file = f"{base_dir}/small_recorder_info.json"

    '''
    segments = {
        'train':[date(2021, 1, 1), now-timedelta(days=61)],
        'valid':[now-timedelta(days=60), now],
        'test':[now-timedelta(days=15), now],
    }


    cfg, work_dir, exp_name = init_qlib(config_paths[0])

    dataset_cfg = cfg["task"]["dataset"]
    # start_time 和 end_time
    dataset_cfg['kwargs']['handler']['kwargs']['start_time']
    dataset_cfg['kwargs']['handler']['kwargs']['end_time'] =  now
    dataset_cfg['kwargs']['handler']['kwargs']['fit_start_time'] = segments['train'][0]
    dataset_cfg['kwargs']['handler']['kwargs']['fit_end_time'] = segments['train'][1]
    segments1 = {
        'train': [dataset_cfg['kwargs']['segments']['train'][0], now - timedelta(days=61)],
        'valid': [now - timedelta(days=60), now],
        'test': [now - timedelta(days=15), now],
    }

    dataset_cfg['kwargs']['segments'] = segments1
    '''


    if weekday == 1: # 周二
        # 训练模型，更新模型
        recorder_list = []

        for config_path in config_paths:
            recorder = train_model_alpha158(config_path=config_path, data_date=now)

            recorder_list.append(json.dumps(recorder.info, ensure_ascii=False))

        if len(recorder_list) > 0:
            with open(recorder_info_file, "w", encoding="utf-8") as f:
                json.dump(recorder_list, f, ensure_ascii=False)
                print("保存 recorder_info 成功")

    # 加载阶段
    try:
        with open(recorder_info_file, "r", encoding="utf-8") as f:
            loaded_info_text = json.load(f)
        print("从文件加载的 recorder_info 内容:")
    except FileNotFoundError:
        print("文件不存在，请先运行保存部分的代码")

    # 加载 recorder_info
    recorder_info_str = json.dumps(loaded_info_text)
    recorder_infos = json.loads(recorder_info_str)
    print(type(recorder_infos), recorder_infos)

    pre_dfs =  []
    for config_path, recorder_info in zip(config_paths, recorder_infos):
        print("config_path:", config_path)
        print("recorder_info:", recorder_info)
        pre_df = predict_data_model(config_path=config_path, recorder_info=json.loads(recorder_info))
        pre_dfs.append(pre_df)

    result = pd.concat(pre_dfs, axis=0)
    result.to_csv(f"{base_dir}/small_result.csv", index=False)


    # predict_data_model(config_path=config_path, recorder_file='recorder_info_2025-12-25.json')
    # predict_data_model(config_path=config_path)
    # predict_data_model(config_path=config_path, recorder_file='lgb_exp_all_recorder_info_2025-12-26.json')




