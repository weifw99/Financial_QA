
import yaml
import pickle
import os
import qlib
from qlib.config import REG_CN
from qlib.utils import init_instance_by_config

def main():
    # 1. 读取配置文件
    with open("workflow_config_tra_Alpha360.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 2. 初始化 Qlib
    qlib_init_cfg = config.get("qlib_init", {})
    qlib.init(provider_uri=qlib_init_cfg.get("provider_uri", "~/.qlib/qlib_data/cn_data"),
              region=REG_CN if qlib_init_cfg.get("region") == "cn" else None)

    # 3. 构建 dataset
    dataset_config = config["task"]["dataset"]
    dataset = init_instance_by_config(dataset_config)

    # 直接拿 handler
    handler = dataset.handler
    print(f"✅ Handler 初始化完成: {type(handler)}")

    # 4. 导出 feature 和 label
    feature_df = handler.fetch(col_set="feature")
    label_df = handler.fetch(col_set="label")

    print(f"Feature shape: {feature_df.shape}, Label shape: {label_df.shape}")

    # 5. 保存为 StaticDataLoader 可用的文件
    output_dir = f"data"
    os.makedirs(output_dir, exist_ok=True)
    with open(os.path.join(output_dir, "feature.pkl"), "wb") as f:
        pickle.dump(feature_df, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(os.path.join(output_dir, "label.pkl"), "wb") as f:
        pickle.dump(label_df, f, protocol=pickle.HIGHEST_PROTOCOL)

    print(f"✅ 导出完成：{output_dir}/feature.pkl, {output_dir}/label.pkl")

    # # 4. 准备数据
    # for seg in ["train", "valid", "test"]:
    #     df_features = dataset.prepare(seg, col_set="feature")
    #
    #     df_labels = dataset.prepare(seg, col_set="label")
    #
    #     os.makedirs("data", exist_ok=True)
    #     with open(f"data/feature_{seg}.pkl", "wb") as f:
    #         pickle.dump(df_features, f)
    #     with open(f"data/label_{seg}.pkl", "wb") as f:
    #         pickle.dump(df_labels, f)
    #     print(f"Feature shape: {len(df_features)}, Label shape: {len(df_labels)}")
    #     print(f"✅ {seg} 数据已保存")

# 注意！必须加这一行
if __name__ == "__main__":
    main()

    print("✅ 数据已保存到 data/feature.pkl 和 data/label.pkl")

    import pickle
    import pandas as pd

    with open("data/feature.pkl", "rb") as f:
        features = pickle.load(f)

    print("Feature shape:", features.shape)
    print("Feature columns:", features.columns)

    columns = [col for col in features.columns]
    for c in columns:
        print(c)

    print(features.head())  # 查看前几行特征


    # path_ = '/Users/dabai/Downloads/feature.pkl'
    #
    # with open(path_, "rb") as f:
    #     features = pickle.load(f)
    #
    # print("Feature shape:", features.shape)
    # print("Feature columns:", features.columns)
    #
    # print(features.head())  # 查看前几行特征
