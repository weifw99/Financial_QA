import datetime
import os
from pathlib import Path

import pandas as pd
import torch
import yaml
import importlib
import qlib
from jinja2 import Template, meta
from qlib.contrib.data.handler import Alpha158
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
                  "step_len": 1}
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


def init_qlib(config, uri=None):
    mlflow_tracking_uri = f"{uri}/mlruns"
    os.environ["MLFLOW_TRACKING_URI"] = mlflow_tracking_uri
    os.environ["MLFLOW_DISABLE_ARTIFACT_CACHE"] = 'true'
    qlib.init(
        provider_uri=os.path.expanduser(config["qlib_init"]["provider_uri"]),
        region=config["qlib_init"].get("region", "cn")
    )
    # R.set_uri(uri=config["mlflow"]["uri"])
    # git_ignore_folder/RD-Agent_workspace/7b18b8c5cd83478db6fd1f8ad755522d/mlruns/153096223065276677/da7cb29b508149cb942e71c711556df5/artifacts/params.pkl
    # R.set_uri("/Users/dabai/Downloads/rd_agent_test/result_ck/7b18b8c5cd83478db6fd1f8ad755522d/mlruns")
    R.set_uri(mlflow_tracking_uri)


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


def predict_from_yaml(config_path="config.yaml", uri=None):
    config = load_config(config_path)

    # 修改时间
    end_time = datetime.date(2025, 6, 20)
    config["task"]["dataset"]["kwargs"]["segments"]["test"] = [ datetime.date(2025, 6, 1),  end_time]
    config['data_handler_config']['end_time'] = end_time

    # config['data_handler_config']['label'] = ['Ref($close, -1) / $close - 1'] # 修改为了预测明天的涨跌数据，不可用于回测，只为得到最后一天的预测结果

    init_qlib(config, uri= uri)

    '''
    data_handler_conf = config["data_handler_config"]
    # 3. 构建 handler 实例
    handler = Alpha158(**data_handler_conf)
    # 4. 获取推理特征数据
    features = handler.fetch(col_set="feature")
    print("特征数据维度:", features.shape)
    # 5. 查看最后几天的特征是否有 NaN（每一行代表一个股票/时间组合）
    print("\n最后10行缺失统计：")
    print(features.tail(10).isna().sum(axis=1))
    # 6. 进一步筛查 2025-06-20 是否存在（确认推理范围）
    last_date = pd.to_datetime("2025-06-20")
    if last_date in features.index.get_level_values("datetime"):
        print(f"\n✅ 存在 {last_date.date()} 的特征数据")
    else:
        print(f"\n❌ 不存在 {last_date.date()} 的特征数据")
    '''



    dataset = build_dataset(config['task']["dataset"])
    # model = build_model(config["task"]['model'])

    if "experiment_name" in config:
        experiment_name = config['experiment_name']
    else:
        experiment_name = 'workflow'
    recorder = get_latest_recorder(experiment_name)
    # 旧路径
    # print("原始 artifact_uri:", recorder.artifact_uri)
    # 设置你希望的新路径（注意加 file:// 前缀）
    # ✅ Monkey patch
    # recorder._artifact_uri = f"file://{recorder.uri}/{recorder.experiment_id}/{recorder.id}/artifacts"
    # print("新的 artifact_uri:", recorder.artifact_uri)

    print(f"✅ 加载模型: params.pkl")
    model = recorder.load_object("params.pkl")

    model_save_dir = os.path.join(uri, "model_ckpt")
    if not os.path.exists(model_save_dir):
        os.makedirs(model_save_dir)
    best_model_path = os.path.join(model_save_dir, f"base_model_params.pt")
    torch.save(model.dnn_model, best_model_path)

    # trainset, validset, testset = dataset.prepare(["train", "valid", "test"])
    preds = model.predict(dataset, segment="test")
    print("📊 预测结果（前几行）：")
    print(preds.head())

    # 1. 将索引中的日期提取出来作为单独的列
    preds1 = preds.reset_index()

    # 2. 筛选指定日期，例如：2025-06-20
    date_str = '2025-06-20'
    specified_date = pd.to_datetime(date_str)  # 确保与索引的日期类型一致
    filtered_preds = preds1[preds1['datetime'] == specified_date]

    filtered_preds.columns = ['datetime', 'instrument', 'score']
    # 筛选 score 大于 0 的数据
    filtered_preds = filtered_preds[filtered_preds['score'] > 0]

    # 按照 'score' 列降序排列
    sorted_preds = filtered_preds.sort_values(by='score', ascending=False)

    # 保存为 CSV 文件
    result_path = os.path.join(Path(uri).parent, "result")
    if not os.path.exists(result_path):
        os.makedirs(result_path)
    output_file = f"{result_path}/{date_str}_{Path(uri).name}_sorted_preds.csv"
    sorted_preds.to_csv(output_file, index=False)

    # 4. 取出前20条数据
    top_preds = sorted_preds.head(30)

    print(f"📊 筛选并排序后的前30条数据（{specified_date}）:")
    print(top_preds)

    print('\n\n\n\n\n')


if __name__ == "__main__":

    # OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 PYTHONFAULTHANDLER=1
    os.environ["OMP_NUM_THREADS"] = '1'
    os.environ["MKL_NUM_THREADS"] = '1'
    os.environ["PYTHONFAULTHANDLER"] = '1'
    # PYTHONWARNINGS=ignore TF_CPP_MIN_LOG_LEVEL=2 PYTHONUNBUFFERED=1
    os.environ["PYTHONWARNINGS"] = 'ignore'
    os.environ["TF_CPP_MIN_LOG_LEVEL"] = '2'
    os.environ["PYTHONUNBUFFERED"] = '1'

    base_path = '/Users/dabai/work/data/agent_qlib/online_check/fin_model_orig/2025-06-07_07-14-29-272373/csi300_qwen30b_loop36' # mlruns/419691347954910967/514f5ce587374c03a6f1b83f524f9fae

    path_list = [
        '/Users/dabai/work/data/agent_qlib/online_check/fin_model_orig/2025-06-07_07-14-29-272373/csi300_qwen30b_loop36',
        '/Users/dabai/work/data/agent_qlib/online_check/fin_model_orig/2025-06-07_07-14-29-272373/csi300_qwen30b_loop19',
        '/Users/dabai/work/data/agent_qlib/online_check/fin_model_orig/2025-06-07_07-14-29-272373/csi300_qwen30b_loop21',
        '/Users/dabai/work/data/agent_qlib/online_check/fin_model_orig/2025-06-07_07-14-29-272373/csi300_qwen30b_loop44',
        '/Users/dabai/work/data/agent_qlib/online_check/fin_model_orig/2025-06-07_07-14-29-272373/csi300_qwen30b_loop61',

    ]

    # for i, base_path in enumerate(path_list):
    #
    #     import sys
    #     print(i, sys.path)
    #     if i>0:
    #         sys.path.remove(path_list[i-1])
    #     sys.path.append(base_path)
    #     print(i, sys.path)
    #
    #     # 如果模块已经导入，需要删除缓存，确保下次重新加载
    #     module_name = "model"  # 替换为你要重新加载的模块名
    #     if module_name in sys.modules:
    #         del sys.modules[module_name]
    #
    #     config_path = f'{base_path}/conf.yaml'
    #     # config_path = 'config.yaml'
    #
    #     predict_from_yaml(config_path, uri=base_path)


    df_list = []
    for base_path in path_list:
        print(base_path)
        date_str = '2025-06-20'
        result_path = os.path.join(Path(base_path).parent, "result")
        output_file = f"{result_path}/{date_str}_{Path(base_path).name}_sorted_preds.csv"

        df = pd.read_csv(output_file)
        df_list.append(df)

    # base_p = '/Users/dabai/liepin/飞书_bak/搜索算法/搜索召回/result/'
    # df_list = []
    # for p_ in os.listdir(base_p):
    #     if p_.endswith('csv'):
    #         df = pd.read_csv(base_p + p_)
    #         df_list.append(df)

    # 步骤 1：合并所有数据
    all_df = pd.concat(df_list, ignore_index=True)

    # 步骤 2：按 instrument 聚合，统计次数和 score 平均
    agg_df = all_df.groupby(["datetime", "instrument"]).agg(
        count=("score", "count"),
        avg_score=("score", "mean"),
        sum_score=("score", "sum")
    ).reset_index()

    # 步骤 3：排序 - 先按 count 降序，再按 avg_score 降序
    agg_df = agg_df.sort_values(by=["avg_score", "count" ], ascending=[False, False])

    # 输出前几行结果
    print(agg_df.head())

    config_path = f'{base_path}/conf.yaml'
    config = load_config(config_path)

    output_file = f"{result_path}/merge_sorted_preds.csv"
    agg_df.to_csv(output_file, index=False)

    init_qlib(config, uri=base_path)
    # dataset = build_dataset(config['task']["dataset"])

    # 获取指定市场数据，使用 instruments 指定市场（如 csi300）
    market = 'csi300'  # 可以换成其他市场名，比如 'szse', 'hs300' 等
    from qlib.data import D

    start_time = '2025-06-20'
    end_time = '2025-06-20'
    # 获取某个日期的指定市场股票数据
    instruments = D.instruments(market=market)
    instrument_list = D.list_instruments(instruments=instruments, start_time=start_time, end_time=end_time, as_list=True)
    required_fields = ["$open", "$close", "$low", "$high", "$volume", "$factor", "Ref($close, -1) / $close - 1"]

    df = D.features(instrument_list, required_fields,  start_time=start_time,  end_time=end_time)
    df.rename(
        columns={
            "$open": "open",
            "$close": "close",
            "$low": "low",
            "$high": "high",
            "$volume": "volume",
            "$factor": "factor",
            "Ref($close, -1) / $close - 1": "lab",
        },
        inplace=True,
    )
    df = df.reset_index()
    print(df)
    df['datetime'] = pd.to_datetime(df['datetime']).dt.strftime("%Y-%m-%d")
    result = pd.merge(df, agg_df, on=['datetime', 'instrument'],  how='inner')
    output_file = f"{result_path}/merge_lab_preds.csv"
    result.to_csv(output_file, index=False)
    print('vgafdg')



    # import joblib
    #
    # params_path = '/Users/dabai/Downloads/rd_agent_test/git_ignore_folder/RD-Agent_workspace/7b18b8c5cd83478db6fd1f8ad755522d/mlruns/153096223065276677/da7cb29b508149cb942e71c711556df5/artifacts/params.pkl'
    # params_path = '/Users/dabai/Downloads/rd_agent_test/git_ignore_folder/RD-Agent_workspace/7b18b8c5cd83478db6fd1f8ad755522d/ret.pkl'
    # params = joblib.load(params_path)
    # print(params)


# model: Model = init_instance_by_config(task_config["model"], accept_types=Model)
# dataset: Dataset = init_instance_by_config(task_config["dataset"], accept_types=Dataset)