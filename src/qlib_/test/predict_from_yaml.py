
import os
import yaml
import importlib
import qlib
from jinja2 import Template, meta
from qlib.workflow import R, Experiment
from ruamel.yaml import YAML


def render_template(config_path: str,  render_env: dict={}) -> str:
    """
    render the template based on the environment or render_env

    Parameters
    ----------
    config_path : str
        configuration path

    Returns
    -------
    str
        the rendered content
    """
    with open(config_path, "r") as f:
        config = f.read()
    # Set up the Jinja2 environment
    template = Template(config)

    # Parse the template to find undeclared variables
    env = template.environment
    parsed_content = env.parse(config)
    variables = meta.find_undeclared_variables(parsed_content)

    # Get context from os.environ according to the variables
    context = {var: os.getenv(var, "") for var in variables if var in os.environ}
    context.update(render_env)
    print(f"Render the template with the context: {context}")

    # Render the template with the context
    rendered_content = template.render(context)
    return rendered_content


def load_config(config_path="config.yaml"):
    """
    加载配置文件
    :param config_path:
    :return:
    """
    # dataset_cls=TSDatasetH step_len=20 num_timesteps=20 PYTHONWARNINGS=ignore TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1 qrun conf.yaml --experiment_name my_exp
    render_env = {"dataset_cls": 'TSDatasetH',
                  "num_timesteps": 20,
                  "step_len": 20}
    rendered_yaml = render_template(config_path, render_env=render_env)
    yaml = YAML(typ="safe", pure=True)
    config = yaml.load(rendered_yaml)
    return config


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
    # git_ignore_folder/RD-Agent_workspace/7b18b8c5cd83478db6fd1f8ad755522d/mlruns/153096223065276677/da7cb29b508149cb942e71c711556df5/artifacts/params.pkl
    # R.set_uri("/Users/dabai/Downloads/rd_agent_test/result_ck/7b18b8c5cd83478db6fd1f8ad755522d/mlruns")
    R.set_uri("/Users/dabai/Downloads/rd_agent_test/git_ignore_folder/RD-Agent_workspace/7b18b8c5cd83478db6fd1f8ad755522d/mlruns")


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

    if "experiment_name" in config:
        experiment_name = config['experiment_name']
    else:
        experiment_name = 'workflow'
    recorder = get_latest_recorder(experiment_name)
    print(f"✅ 加载模型: params.pkl")
    model = recorder.load_object("params.pkl")
    # trainset, validset, testset = dataset.prepare(["train", "valid", "test"])
    preds = model.predict(dataset, segment="test")
    print("📊 预测结果（前几行）：")
    print(preds.head())


if __name__ == "__main__":

    # OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONFAULTHANDLER=1
    os.environ["OMP_NUM_THREADS"] = '1'
    os.environ["MKL_NUM_THREADS"] = '1'
    os.environ["PYTHONFAULTHANDLER"] = '1'
    # PYTHONWARNINGS=ignore TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1
    os.environ["PYTHONWARNINGS"] = 'ignore'
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
    os.environ["PYTHONUNBUFFERED"] = '1'

    config_path = '/Users/dabai/Downloads/rd_agent_test/git_ignore_folder/RD-Agent_workspace/7b18b8c5cd83478db6fd1f8ad755522d/conf.yaml'
    config_path = 'config.yaml'

    # predict_from_yaml(config_path)

    import joblib

    params_path = '/Users/dabai/Downloads/rd_agent_test/git_ignore_folder/RD-Agent_workspace/7b18b8c5cd83478db6fd1f8ad755522d/mlruns/153096223065276677/da7cb29b508149cb942e71c711556df5/artifacts/params.pkl'
    params_path = '/Users/dabai/Downloads/rd_agent_test/git_ignore_folder/RD-Agent_workspace/7b18b8c5cd83478db6fd1f8ad755522d/ret.pkl'
    params = joblib.load(params_path)
    print(params)


# model: Model = init_instance_by_config(task_config["model"], accept_types=Model)
# dataset: Dataset = init_instance_by_config(task_config["dataset"], accept_types=Dataset)