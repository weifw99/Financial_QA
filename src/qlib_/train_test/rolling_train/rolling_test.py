from qlib_.train_test.rolling_train.rolling_utils import run_rolling_experiment

res = run_rolling_experiment(
    config_path="config/workflow_config_lgb_Alpha158_tree_import.yaml",
    provider_uri="/Users/dabai/.qlib/qlib_data/cn_data",
    rolling_step=21,
    rolling_type="expanding",
)

# 查看结果
print(res["all_predictions"].head())
print(res["rolling_metrics"].groupby("rolling_id")["IC"].mean())
print(res["stability"])
