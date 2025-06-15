
import os
import importlib
import qlib
from qlib.workflow import R, Experiment

from qlib_.test.qlib_help.util_config import load_config


def import_object(obj_path, module_path):
    """支持模块路径字符串导入类或函数"""
    if module_path:
        module = importlib.import_module(module_path)
        return getattr(module, obj_path)
    else:
        module_path, class_name = obj_path.rsplit(".", 1)
        module = importlib.import_module(module_path)
        return getattr(module, class_name)


def init_qlib(config):
    qlib.init(
        provider_uri=os.path.expanduser(config["qlib_init"]["provider_uri"]),
        region=config["qlib_init"].get("region", "cn")
    )
    # R.set_uri(uri=config["mlflow"]["uri"])
    R.set_uri("/Users/dabai/Downloads/rd_agent_test/result_ck/7b18b8c5cd83478db6fd1f8ad755522d/mlruns")


def get_latest_recorder(experiment_name):
    exp = R.get_exp(experiment_name=experiment_name)
    recorders = exp.list_recorders(rtype=Experiment.RT_L)
    if not recorders:
        raise ValueError(f"No recorders found in experiment '{experiment_name}'")
    return recorders[-1]


def build_dataset(config):
    handler_cfg = config['kwargs']["handler"]
    handler_cls = import_object(handler_cfg["class"], handler_cfg["module_path"])
    handler = handler_cls(**handler_cfg["kwargs"])

    dataset_cls = import_object(config["class"],  config["module_path"])
    return dataset_cls(
        handler=handler,
        segments=config['kwargs']["segments"]
    )


def build_model(config):
    model_cls = import_object(config["class"], config["module_path"])
    return model_cls(**config["kwargs"])


def predict_from_yaml(config_path="config.yaml"):
    config = load_config(config_path)
    init_qlib(config)

    dataset = build_dataset(config['task']["dataset"])
    # model = build_model(config["task"]['model'])

    recorder = get_latest_recorder(config["experiment_name"])
    print(f"✅ 加载模型: params.pkl")
    model = recorder.load_object("params.pkl")
    # trainset, validset, testset = dataset.prepare(["train", "valid", "test"])
    preds = model.predict(dataset, segment="test")
    print("📊 预测结果（前几行）：")
    print(preds.head())
