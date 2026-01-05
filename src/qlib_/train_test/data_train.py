import os
import pickle
import yaml
import qlib
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config


def load_config_yaml(config_path):
    with open(config_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg


def dump_dataset_to_static(config_path, output_dir):
    # 1. 读取配置
    cfg = load_config_yaml(config_path)

    # 2. 初始化 Qlib
    qlib_init = cfg.get("qlib_init", {})
    qlib.init(
        provider_uri=qlib_init.get("provider_uri", "~/.qlib/qlib_data/cn_data"),
        region=REG_CN if qlib_init.get("region") == "cn" else None,
    )



    # 3. 初始化 MTSDatasetH
    dataset_cfg = cfg["task"]["dataset"]
    dataset = init_instance_by_config(dataset_cfg)

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
    from qlib.data.dataset import DataHandlerLP
    df_train, df_valid = dataset.prepare(["train", "valid"], col_set=["feature", "label"], data_key=DataHandlerLP.DK_L)
    x_train, y_train = df_train["feature"], df_train["label"]
    x_valid, y_valid = df_valid["feature"], df_valid["label"]


    # 初始化 model
    model_cfg = cfg["task"]["model"]
    model = init_instance_by_config(model_cfg)



if __name__ == "__main__":

    dump_dataset_to_static("../workflow_config_tra_Alpha158.yaml", "data")


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