# Copyright (c) Microsoft Corporation.
# Licensed under the MIT License.

"""
This example shows how a TrainerRM works based on TaskManager with rolling tasks.
After training, how to collect the rolling results will be shown in task_collecting.
Based on the ability of TaskManager, `worker` method offer a simple way for multiprocessing.
"""
import json
import os
from pprint import pprint
from typing import List

import fire
import pandas as pd
import qlib
from qlib.backtest import backtest
from qlib.constant import REG_CN
from qlib.contrib.evaluate import risk_analysis, indicator_analysis
from qlib.contrib.model import LGBModel
from qlib.model.ens.ensemble import RollingEnsemble
from qlib.utils import fill_placeholder, get_date_by_shift
from qlib.utils.time import Freq
from qlib.workflow import R, Recorder
from qlib.workflow.record_temp import SigAnaRecord, PortAnaRecord
from qlib.workflow.task.gen import RollingGen, task_generator
from qlib.workflow.task.manage import TaskManager, run_task
from qlib.workflow.task.collect import RecorderCollector
from qlib.model.trainer import TrainerR, TrainerRM, task_train
from qlib.tests.config import CSI100_RECORD_LGB_TASK_CONFIG, CSI100_RECORD_XGBOOST_TASK_CONFIG

from qlib_.train_test.data_train import load_config_yaml

import gc

from qlib_.train_test.rolling_train_tree.export_feature_import import tree_frature_import_alpha1581_by_task
from qlib_.train_test.rolling_train_tree.qlib_html_report import generate_html_report


class RollingTaskExample:
    def __init__(
        self,
        provider_uri="~/.qlib/qlib_data/cn_data",
        region=REG_CN,
        task_url="mongodb://127.0.0.1:27017/",
        task_db_name="rolling_db",
        experiment_name="rolling_exp",
        task_pool=None,  # if user want to  "rolling_task"
        task_config=None,
        feature_task_config=None,
        rolling_step=550,
        rolling_type=RollingGen.ROLL_SD,
    ):
        self.feature_task_config = feature_task_config
        # TaskManager config
        if task_config is None:
            task_config = [CSI100_RECORD_XGBOOST_TASK_CONFIG, CSI100_RECORD_LGB_TASK_CONFIG]
        mongo_conf = {
            "task_url": task_url,
            "task_db_name": task_db_name,
        }
        qlib.init(provider_uri=provider_uri, region=region, mongo=mongo_conf)
        self.experiment_name = experiment_name
        if task_pool is None:
            self.trainer = TrainerR(experiment_name=self.experiment_name)
        else:
            self.task_pool = task_pool
            self.trainer = TrainerRM(self.experiment_name, self.task_pool)
            self.task_manager = TaskManager(task_pool=self.task_pool)

        self.task_config = task_config
        self.task_record_config = [ task['record'] for task in task_config]
        self.rolling_gen = RollingGen(step=rolling_step, rtype=rolling_type)

    # Reset all things to the first status, be careful to save important data
    def reset(self):
        print("========== reset ==========")
        if isinstance(self.trainer, TrainerRM):
            self.task_manager.remove()
        exp = R.get_exp(experiment_name=self.experiment_name)
        for rid in exp.list_recorders():
            exp.delete_recorder(rid)

    def task_generating(self):
        print("========== task_generating ==========")
        tasks = task_generator(
            tasks=self.task_config,
            generators=self.rolling_gen,  # generate different date segments
        )
        pprint(tasks)
        print(f"========== task_generating {len(tasks)} ==========")
        return tasks

    def check_update_task_backtest(self, tasks):
        print("========== update_task_backtest ==========")

        from qlib.data import D
        calendar = D.calendar()
        safe_end = calendar[-2]  # 永远留一个 buffer
        print("safe_end:", safe_end)
        temp_tasks = []
        for task in tasks:
            # 获取测试时间
            test_start = task['dataset']['kwargs']['segments']['test'][0]
            test_end = task['dataset']['kwargs']['segments']['test'][1]

            if test_end is None:
                task['dataset']['kwargs']['segments']['test'] = (test_start, safe_end)

            task['record'][2]['kwargs']['config']['backtest']['start_time'] = task['dataset']['kwargs']['segments']['test'][0]
            task['record'][2]['kwargs']['config']['backtest']['end_time'] = task['dataset']['kwargs']['segments']['test'][1]
            task['dataset']['kwargs']['handler']['kwargs']['fit_start_time'] = task['dataset']['kwargs']['segments']['train'][0]
            task['dataset']['kwargs']['handler']['kwargs']['fit_end_time'] = task['dataset']['kwargs']['segments']['train'][1]

            if test_start <= safe_end and test_end is not None:
                temp_tasks.append(task)

        pprint(temp_tasks)
        print(f"========== update_task_backtest {len(temp_tasks)} ==========")
        return temp_tasks

    def task_training(self, tasks):
        print("========== task_training ==========")
        running_res = []
        while( True):
            running_task = self.task_manager.fetch_task(status=TaskManager.STATUS_RUNNING)

            if running_task is None:
                break
            # 通过底层 collection 安全地 update_one
            _id = running_task["_id"]
            running_res.append(running_task['res'])
            res = self.task_manager.task_pool.update_one({"_id": _id, "status": TaskManager.STATUS_RUNNING},
                                          {"$set": {"status": TaskManager.STATUS_WAITING}})
            print(f"[RECOVER] reset task {_id} result: matched={res.matched_count}, modified={res.modified_count}")

        if len(running_res) > 0:
            exp = R.get_exp(experiment_name=self.experiment_name)
            for res in running_res:
                exp.delete_recorder(res.id)

        for task in tasks:
            print(f"[START] task  train {task}")
            print(f"task config {task}")

            if self.feature_task_config is not None:
                # 特征筛选  tree train
                importance_df: pd.DataFrame = tree_frature_import_alpha1581_by_task(config_path=self.feature_task_config, task_info= task)

                infer_processors = task['dataset']['kwargs']['handler']['kwargs']['infer_processors']
                for processor in infer_processors:
                    if processor['class'] == 'FilterCol':
                        # 筛选一半的特征
                        processor['kwargs']['col_list'] = importance_df['feature'].tolist()[:74]
                        if 'TRAModel' == task['model']['class']:
                            # 修改模型输入大小
                            # processor['kwargs']['col_list'] = importance_df['feature'].tolist()[:74]
                            pass

                print(f"[EDIT]task config {task}")

            self.trainer.train([task])
            print(f"[END] task train")

        # self.trainer.train(tasks)

    def query_task(self):
        print("========== query_task ==========")

        _id_list = [r["_id"] for r in self.task_manager.query()]

        list_metrics = []
        list_result = []
        report_normals = []
        port_analysis = []
        feature_importances = []
        for _id in _id_list:
            task_r: dict = self.task_manager.re_query(_id)
            status = task_r['status']
            if not status == TaskManager.STATUS_DONE:
                print(f"[SKIP] task {_id} is not done")
                continue

            rec = task_r["res"]
            rec: Recorder

            '''
            metrics1 = rec.list_metrics()  # 获取指标
            # 2️⃣ 保存 IC / ICIR 等指标（关键）
            SigAnaRecord(
                recorder=rec,
                ana_long_short=False,
                ann_scaler=252,
            ).generate()

            # 测试
            port_analysis_config = None
            for record_ in self.task_record_config[0]:
                if record_['class'] == 'PortAnaRecord':
                    port_analysis_config = record_['kwargs']['config']
            if port_analysis_config is not None:
                PortAnaRecord(
                    recorder=rec,
                    config=port_analysis_config,
                ).generate()
            '''

            metrics = rec.list_metrics()  # 获取指标

            params = rec.list_params()  # 获取参数
            seg_test = params['dataset.kwargs.segments.test'].replace('(', '').replace(')', '').replace("'",
                                                                                                        '').replace(
                'Timestamp', '').replace('00:00:00', '').strip()
            # '2017-03-02 , 2017-03-30'
            metrics['seg_test'] = seg_test

            list_metrics.append(metrics)

            rec.load_object('dataset')
            rec.load_object('sig_analysis/ic.pkl')
            rec.load_object('sig_analysis/ric.pkl')

            positions_normals = rec.load_object('portfolio_analysis/positions_normal_1day.pkl')
            position_list = []
            for k, v in positions_normals.items():
                # print(k)
                # print(v)
                position_list.append([k, v.init_cash, v.position['now_account_value'], str(v.position)])

            position_pd = pd.DataFrame(position_list)
            position_pd.columns = ['date', 'init_cash', 'now_account_value', 'position']

            report_normal_1day = rec.load_object('portfolio_analysis/report_normal_1day.pkl').reset_index()  # 获取回测结果
            report_normals.append(report_normal_1day)

            port_analysis_1day = rec.load_object('portfolio_analysis/port_analysis_1day.pkl').reset_index()
            port_analysis_1day['seg_test'] = seg_test
            port_analysis.append(port_analysis_1day)

            pred_df = rec.load_object('pred.pkl')  # 获取预测结果
            lab_df = rec.load_object('label.pkl')  # 获取标签
            if 'label' in pred_df.columns:
                lab_df.columns = ['orig_label']
            else:
                lab_df.columns = ['label']
            result = pred_df.join(lab_df, how='inner').reset_index()
            list_result.append(result)

            if (params['model.class'] == 'LGBModel' and 'params.pkl' in rec.list_artifacts()
                    and 'dataset.kwargs.handler.kwargs.infer_processors' in params):
                model_params = rec.load_object('params.pkl')  # 获取标签
                model_params: LGBModel
                infer_processors = json.loads(
                    params['dataset.kwargs.handler.kwargs.infer_processors'].replace("'", '"'))
                ds_feature_names = infer_processors[0]['kwargs']['col_list']
                # 获取特征名称
                feature_names = model_params.model.feature_name()

                print("feature_names len:", len(feature_names))
                print("ds_feature_names len:", len(ds_feature_names))
                # 提取 Column_x 中的数字并映射到真实特征名
                real_feature_names = []
                for col_name in feature_names:
                    index = int(col_name.replace('Column_', ''))
                    real_feature_names.append(ds_feature_names[index])
                # 获取基于增益的重要性
                importance = model_params.model.feature_importance(importance_type='gain')

                # 创建特征重要性 DataFrame
                importance_df = pd.DataFrame({
                    'feature': real_feature_names,
                    'importance': importance
                }).sort_values('importance', ascending=False)
                importance_df['seg_test'] = seg_test
                importance_df['rank'] = range(1, len(importance_df) + 1)

                feature_importances.append(importance_df)

            print(rec)

        if not os.path.exists(f"data/{self.experiment_name}"):
            os.makedirs(f"data/{self.experiment_name}")

        metrics_pd = pd.DataFrame(list_metrics)
        metrics_pd.to_csv(f"data/{self.experiment_name}/metrics.csv", index=False)

        if len(list_result) > 0:
            df_all = pd.concat(list_result, ignore_index=True)
            df_all.to_csv(f"data/{self.experiment_name}/pre_result.csv", index=False)
            print(df_all)

        if len(report_normals) > 0:
            report_all = pd.concat(report_normals, ignore_index=True)
            report_all.to_csv(f"data/{self.experiment_name}/report_normal.csv", index=False)
            print(report_all)

        if len(port_analysis) > 0:
            post_all = pd.concat(port_analysis, ignore_index=True)
            post_all.to_csv(f"data/{self.experiment_name}/port_analysis.csv", index=False)
            print(post_all)

        if len(feature_importances) > 0:
            feature_all = pd.concat(feature_importances, ignore_index=True)
            feature_all.to_csv(f"data/{self.experiment_name}/feature_importance.csv", index=False)
            print(feature_all)

        # 基于 rolling 结果进行整体回测
        config = self.task_record_config[0][2]['kwargs']['config']

        print("config:", config)

        strategy_config = config["strategy"]
        _default_executor_config = {
            "class": "SimulatorExecutor",
            "module_path": "qlib.backtest.executor",
            "kwargs": {
                "time_per_step": "day",
                "generate_portfolio_metrics": True,
            },
        }
        executor_config = config.get("executor", _default_executor_config)
        backtest_config = config["backtest"]
        pred_df = df_all.reset_index()[['datetime', 'instrument', 'score']]
        pred_df = pred_df.set_index(['datetime', 'instrument'])
        # replace the "<PRED>" with prediction saved before
        placeholder_value = {"<PRED>": pred_df}
        fill_placeholder(strategy_config, placeholder_value)
        fill_placeholder(executor_config, placeholder_value)

        # if the backtesting time range is not set, it will automatically extract time range from the prediction file
        dt_values = pred_df.index.get_level_values("datetime")
        if backtest_config["start_time"] is None:
            backtest_config["start_time"] = dt_values.min()
        if backtest_config["end_time"] is None:
            backtest_config["end_time"] = get_date_by_shift(dt_values.max(), 1)

        artifact_objects = {}
        # custom strategy and get backtest
        portfolio_metric_dict, indicator_dict = backtest(
            executor=executor_config, strategy=strategy_config, **backtest_config
        )
        for _freq, (report_normal, positions_normal) in portfolio_metric_dict.items():
            artifact_objects.update({f"report_normal_{_freq}.pkl": report_normal})
            artifact_objects.update({f"positions_normal_{_freq}.pkl": positions_normal})

        for _freq, indicators_normal in indicator_dict.items():
            artifact_objects.update({f"indicators_normal_{_freq}.pkl": indicators_normal[0]})
            artifact_objects.update({f"indicators_normal_{_freq}_obj.pkl": indicators_normal[1]})

        ret_freq = []
        if executor_config["kwargs"].get("generate_portfolio_metrics", False):
            _count, _freq = Freq.parse(executor_config["kwargs"]["time_per_step"])
            ret_freq.append(f"{_count}{_freq}")

        risk_analysis_freq = [ret_freq[0]]
        indicator_analysis_freq = [ret_freq[0]]

        for _analysis_freq in risk_analysis_freq:
            if _analysis_freq in portfolio_metric_dict:
                report_normal, _ = portfolio_metric_dict.get(_analysis_freq)
                analysis = dict()
                analysis["excess_return_without_cost"] = risk_analysis(
                    report_normal["return"] - report_normal["bench"], freq=_analysis_freq
                )
                analysis["excess_return_with_cost"] = risk_analysis(
                    report_normal["return"] - report_normal["bench"] - report_normal["cost"], freq=_analysis_freq
                )
                analysis_df = pd.concat(analysis)  # type: pd.DataFrame
                # log metrics
                # save results
                artifact_objects.update({f"port_analysis_{_analysis_freq}.pkl": analysis_df})
                # print out results
                pprint(f"The following are analysis results of benchmark return({_analysis_freq}).")
                pprint(risk_analysis(report_normal["bench"], freq=_analysis_freq))
                pprint(f"The following are analysis results of the excess return without cost({_analysis_freq}).")
                pprint(analysis["excess_return_without_cost"])
                pprint(f"The following are analysis results of the excess return with cost({_analysis_freq}).")
                pprint(analysis["excess_return_with_cost"])

        for _analysis_freq in indicator_analysis_freq:
            if _analysis_freq in indicator_dict:
                indicators_normal = indicator_dict.get(_analysis_freq)[0]
                analysis_df = indicator_analysis(indicators_normal)
                # save results
                artifact_objects.update({f"indicator_analysis_{_analysis_freq}.pkl": analysis_df})
                pprint(f"The following are analysis results of indicators({_analysis_freq}).")
                pprint(analysis_df)

        report_normal_1day = artifact_objects['report_normal_1day.pkl'].reset_index()
        port_analysis_1day = artifact_objects['port_analysis_1day.pkl'].reset_index()

        report_normal_1day.to_csv(f"data/{self.experiment_name}/report_normal_1day_all.csv", index=False)
        port_analysis_1day.to_csv(f"data/{self.experiment_name}/port_analysis_1day_all.csv", index=False)
        print(report_normal_1day)

        positions_normals = artifact_objects['positions_normal_1day.pkl']
        position_list = []
        for k, v in positions_normals.items():
            print(k)
            print(v)
            position_list.append([k, v.init_cash, v.position['now_account_value'], str(v.position)])

        position_pd = pd.DataFrame(position_list)
        position_pd.columns = ['date', 'init_cash', 'now_account_value', 'position']
        position_pd.to_csv(f"data/{self.experiment_name}/positions_normal_1day_all.csv", index=False)

        # 生成 回测报表 HTML 报表
        pred_label_df = df_all.reset_index()
        pred_label_df = pred_label_df.set_index(['datetime', 'instrument'])
        generate_html_report(artifact_objects['port_analysis_1day.pkl'],
                             artifact_objects['report_normal_1day.pkl'],
                             pred_label_df,
                             output_dir=f"data/{self.experiment_name}/report_output",
                             )



    def worker(self):
        # NOTE: this is only used for TrainerRM
        # train tasks by other progress or machines for multiprocessing. It is same as TrainerRM.worker.
        print("========== worker ==========")
        run_task(task_train, self.task_pool, experiment_name=self.experiment_name)

    def task_collecting(self):
        print("========== task_collecting ==========")

        def rec_key(recorder):
            task_config = recorder.load_object("task")
            model_key = task_config["model"]["class"]
            rolling_key = task_config["dataset"]["kwargs"]["segments"]["test"]
            return model_key, rolling_key

        def my_filter(recorder):
            # only choose the results of "LGBModel"
            model_key, rolling_key = rec_key(recorder)
            if model_key == "LGBModel":
                return True
            return False

        collector = RecorderCollector(
            experiment=self.experiment_name,
            # process_list=RollingGroup(),
            rec_key_func=rec_key,
            artifacts_key=["pred", "label"],
            process_list=[RollingEnsemble()],
            artifacts_path={"pred": "pred.pkl", "label": "label.pkl"},
            # rec_filter_func=my_filter,
        )
        print(collector())
        res = collector()
        print( res.keys())
        print(len( res))

    def main(self):
        # self.reset()

        tasks_org = self.task_generating()
        tasks = self.check_update_task_backtest(tasks_org)
        self.task_training(tasks)

        self.query_task()
        # self.task_collecting()


if __name__ == "__main__":
    ## to see the whole process with your own parameters, use the command below
    # python task_manager_rolling.py main --experiment_name="your_exp_name"

    csi300_feature_task_config = './config/import_c/csi300_lgb_Alpha158_all.yaml'
    zxzz399101_feature_task_config = './config/import_c/zxzz399101_lgb_Alpha158_all.yaml'

    config_task_exps = [
        # ("./config/csi300/workflow_config_lgb_Alpha158_tree_import.yaml", 'rolling_exp_tree_import', None),
        # ("./config/csi300/workflow_config_lgb_Alpha158_all.yaml", 'rolling_exp_tree_all', None),
        # ("./config/csi300/workflow_config_lgb_Alpha158_rec_tree.yaml", 'rolling_exp_rec_tree', None),
        # ("./config/csi300/workflow_config_lgb_Alpha158_rec_tree.yaml", 'rolling_exp_tree_select', csi300_feature_task_config),
        ("./config/csi300/workflow_config_lgb_Alpha158_rec_tree1.yaml", 'rolling_exp_tree_select1', None),
        # ("./config/csi300/workflow_config_tra_Alpha158_rec.yaml", 'rolling_exp_rec_tra', csi300_feature_task_config),
        # ("./config/csi300/workflow_config_tra_Alpha158_rec_tree.yaml", 'rolling_exp_rec_tree_tra', csi300_feature_task_config),
        # ("./config/csi300/workflow_config_tra_Alpha158_tree_import.yaml", 'rolling_exp_tree_import_tra11', csi300_feature_task_config),
    ]

    config_task_exps = [
        ("./config/zxzz399101/workflow_config_lgb_Alpha158_tree_import.yaml", 'zxzz399101_tree_import', None),
        ("./config/zxzz399101/workflow_config_lgb_Alpha158_all.yaml", 'zxzz399101_tree_all', None),
        ("./config/zxzz399101/workflow_config_lgb_Alpha158_rec_tree.yaml", 'zxzz399101_rec_tree', None),
        ("./config/zxzz399101/workflow_config_lgb_Alpha158_rec_tree1.yaml", 'zxzz399101_tree_select1', None),
    ]
    # rolling_types = [RollingGen.ROLL_EX, RollingGen.ROLL_SD]
    # ROLL_EX 效果优于 ROLL_SD
    rolling_types = [RollingGen.ROLL_EX]
    rolling_types = [RollingGen.ROLL_SD]
    config_task_exps = [
        ("./config/zxzz399101/workflow_config_lgb_Alpha158_all.yaml", 'zxzz399101_tree_all11', None),
    ]

    for config_path, task_exp1, feature_task_config in config_task_exps:
        for rolling_type in rolling_types:
            print(f"========== {task_exp1}_{rolling_type} ==========")
            cfg = load_config_yaml(config_path=config_path)
            task_exp = f'{task_exp1}_{rolling_type}'

            RollingTaskExample(
                task_config=[cfg["task"] ],
                feature_task_config=feature_task_config,
                task_pool=task_exp,
                experiment_name=task_exp,
                rolling_step=21,
                rolling_type=rolling_type
            ).main()


    '''
    cfg = load_config_yaml(config_path="./config/workflow_config_lgb_Alpha158_tree_import.yaml")
    task_exp = 'rolling_exp_tree_import'

    cfg = load_config_yaml(config_path="./config/workflow_config_lgb_Alpha158_all.yaml")
    task_exp = 'rolling_exp_tree_all'

    RollingTaskExample(
        task_config=[cfg["task"] ],
        task_pool=task_exp,
        experiment_name=task_exp,
        rolling_step=21,
        rolling_type=RollingGen.ROLL_EX
        # rolling_type=RollingGen.ROLL_SD
    ).main()
    '''

