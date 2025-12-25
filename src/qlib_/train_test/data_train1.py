
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


def init_qlib(config_path='') -> tuple[Any, Path]:
    # 1. 读取配置
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. 初始化 Qlib
    qlib_init = cfg.get("qlib_init", {})

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
    return cfg, work_dir


def train_model_alpha158(config_path=''):

    cfg, work_dir = init_qlib(config_path)

    # 3. 初始化 MTSDatasetH
    dataset_cfg = cfg["task"]["dataset"]
    dataset = init_instance_by_config(dataset_cfg)

    '''

    # 直接拿 handler
    handler = dataset.handler
    print(f"✅ Handler 初始化完成: {type(handler)}")

    # 4. 导出 feature 和 label
    feature_df = handler.fetch(col_set="feature")
    label_df = handler.fetch(col_set="label")

    print(f"Feature shape: {feature_df.shape}, Label shape: {label_df.shape}")

    # 5. 保存为 StaticDataLoader 可用的文件
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "feature.pkl"), "wb") as f:
        pickle.dump(feature_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(output_dir, "label.pkl"), "wb") as f:
        pickle.dump(label_df, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ 导出完成：{output_dir}/feature.pkl, {output_dir}/label.pkl")

    # get features and labels
    # from qlib.data.dataset import DataHandlerLP
    # df_train, df_valid = dataset.prepare(["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    # x_train, y_train = df_train["feature"], df_train["label"]
    # x_valid, y_valid = df_valid["feature"], df_valid["label"]

    '''

    port_analysis_config = cfg['port_analysis_config']
    # 初始化 model
    model_cfg = cfg["task"]["model"]
    print(f"初始化 model: {model_cfg}")
    model = init_instance_by_config(model_cfg)

    ''' '''
    experiment_name = "workflow"
    recorder_info_file = f"{work_dir}/recorder_info_{datetime.now().strftime('%Y-%m-%d')}.json"
    # start exp
    train_model = None
    with R.start(experiment_name=experiment_name) as rec:
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

        recorder = R.get_recorder(recorder_id=active_recorder.id, experiment_name=experiment_name)
        recorder1 = R.get_recorder()
        # ✅ 显式保存模型
        recorder.save_objects(model=model)

        # prediction
        # recorder = R.get_recorder()
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

        # record = R.get_recorder(recorder_id="<record_id>")
        # model = record.load_object("model")

        # 保存 info 到 JSON 文件
        with open(recorder_info_file, "w", encoding="utf-8") as f:
            json.dump(active_recorder.info, f, indent=4, ensure_ascii=False)
        print(f"active_recorder_info: {json.dumps(active_recorder.info, indent=4)}")

    # 加载阶段
    try:
        with open(recorder_info_file, "r", encoding="utf-8") as f:
            loaded_info_text = json.load(f)
        print("从文件加载的 recorder_info 内容:")
        recorder_info_str = json.dumps(loaded_info_text, indent=4)
        recorder_info = json.loads(recorder_info_str)
        print( type(recorder_info) , recorder_info )
    except FileNotFoundError:
        print("文件不存在，请先运行保存部分的代码")

    # 获取 recorder,
    recorder = R.get_recorder(recorder_id=recorder_info['id'], experiment_id=recorder_info['experiment_id'])
    # record = get_recorder("workflow")
    model = recorder.load_object("model")
    print("model:", model)

def predict_data_model(config_path='', recorder_file=None):
    cfg, work_dir = init_qlib(config_path)
    if recorder_file:
        recorder_info_file = recorder_file
    else:
        recorder_info_file = f"{work_dir}/recorder_info_{datetime.now().strftime('%Y-%m-%d')}.json"
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

    dataset_cfg['kwargs']['handler']['kwargs']['start_time'] = date(2021, 1, 1)
    dataset_cfg['kwargs']['handler']['kwargs']['end_time'] = date(2025, 5, 15)

    dataset_cfg['kwargs']['segments'] = {
        'test': [date(2025, 1, 1), date(2025, 5, 15)]
    }

    print(f"修改后 MTSDatasetH, 配置: {dataset_cfg}")
    # 对 dataset_cfg 进行修改
    # 1. 修改 end_time  'end_time': datetime.date(2025, 5, 15)
    #  保持不变 fit_start_time: 2021-01-01  fit_end_time:   2024-12-31
    # 2.
    dataset = init_instance_by_config(dataset_cfg)

    print("dataset:", dataset)

    SignalRecord(
        model=model,
        dataset=dataset,
        recorder=recorder
    ).generate()

    pred = recorder.load_object("pred.pkl")

    print("pred type:", type(pred))
    print("pred:", pred)






if __name__ == "__main__":
    config_path = "./workflow_config_tra_Alpha158.yaml"
    # train_model_alpha158(config_path=config_path)

    predict_data_model(config_path=config_path)




