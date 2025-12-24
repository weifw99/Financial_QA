
import os
import pickle
import yaml
import qlib
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config

from qlib.contrib.model.gbdt import LGBModel
from qlib.contrib.data.handler import Alpha158
from qlib.utils import init_instance_by_config, flatten_dict
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord, PortAnaRecord



def train_model_alpha158(config_path, output_dir):
    # 1. 读取配置
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)

    # 2. 初始化 Qlib
    qlib_init = cfg.get("qlib_init", {})
    qlib.init(
        provider_uri=qlib_init.get("provider_uri", "~/.qlib/qlib_data/cn_data"),
        region=REG_CN if qlib_init.get("region") == "cn" else None,
    )



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

    # 初始化 record
    record_cfg = cfg["task"]["record"]
    record = init_instance_by_config(record_cfg)

    # get features and labels
    # from qlib.data.dataset import DataHandlerLP
    # df_train, df_valid = dataset.prepare(["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    # x_train, y_train = df_train["feature"], df_train["label"]
    # x_valid, y_valid = df_valid["feature"], df_valid["label"]

    '''

    # 初始化 model
    model_cfg = cfg["task"]["model"]
    print(model_cfg)
    model = init_instance_by_config(model_cfg)

    # start exp
    with R.start(experiment_name="workflow") as rec:
        print("当前 record_id:", rec.id)  # ✅ record_id 就在这里
        # train
        R.log_params(**flatten_dict(cfg["task"]))
        model.fit(dataset)

        recorder = R.get_recorder()
        # ✅ 显式保存模型
        recorder.save_objects(model=model)

        # prediction
        recorder = R.get_recorder()
        sr = SignalRecord(model, dataset, recorder)
        sr.generate()

        # record = R.get_recorder(recorder_id="<record_id>")
        # model = record.load_object("model")


if __name__ == "__main__":

    train_model_alpha158("./workflow_config_tra_Alpha158.yaml", "data")


    import pickle
    import pandas as pd

    with open("../data/feature.pkl", "rb") as f:
        features = pickle.load(f)

    print("Feature shape:", features.shape)
    print("Feature columns:", features.columns)

    print(features.head())  # 查看前几行特征

    columns = [col for col in features.columns]
    for c in columns:
        print(c)


