
import os
import qlib
from qlib.workflow import R
from qlib.config import REG_CN
from qlib.data.dataset import DatasetH
from qlib.contrib.model.pytorch_general_nn import GeneralPTNN
from qlib.contrib.data.handler import Alpha158

def init_qlib():
    # 初始化 Qlib 数据和实验系统，指定 provider_uri 与 mlruns 目录
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)
    # 指定 mlruns 存储目录（可以修改为你期望的路径，如 "./mlruns"）
    R.set_uri("/Users/dabai/Downloads/rd_agent_test/result_ck/7b18b8c5cd83478db6fd1f8ad755522d/mlruns")


def get_latest_recorder(experiment_name):
    # 获取指定 experiment 下的所有 recorder，并选取最后一个
    exp = R.get_exp(experiment_name)
    recorders = exp.list_recorders()
    # 这里简单选择最新的 recorder（也可以依据评估指标进一步排序）
    if not recorders:
        raise ValueError(f"No recorder found for experiment {experiment_name}")
    latest_recorder = recorders[-1]
    return latest_recorder


def load_dataset():
    # 构造数据集，保持配置与训练时一致
    handler = Alpha158(
        start_time="2010-01-01",
        end_time="2025-05-15",
        fit_start_time="2010-01-01",
        fit_end_time="2018-12-31",
        instruments="csi300"
    )
    dataset = DatasetH(
        handler=handler,
        segments={
            "train": ("2010-01-01", "2018-12-31"),
            "valid": ("2019-01-01", "2020-12-31"),
            "test": ("2021-01-01", "2025-05-15")
        }
    )
    return dataset


def init_model():
    # 初始化模型结构，参数必须与训练时保持一致
    model = GeneralPTNN(
        n_epochs=6,
        lr=1e-3,
        batch_size=2000,
        loss="mse",
        GPU=0,
        pt_model_uri="model.model_cls",
        pt_model_kwargs={"num_features": 20}
    )
    return model


def run_prediction(experiment_name="your_experiment_name"):
    init_qlib()
    dataset = load_dataset()
    model = init_model()

    recorder = get_latest_recorder(experiment_name)
    model_path = recorder.get_local_path("model.pkl")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"模型文件不存在：{model_path}")

    model.load(model_path)
    print(f"✅ 成功加载模型: {model_path}")

    # 对 test 集数据进行预测
    pred = model.predict(dataset.get_subset("test"))
    print("📈 预测结果样例：")
    print(pred.head())


if __name__ == "__main__":
    run_prediction(experiment_name="my_exp")
