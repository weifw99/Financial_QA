import itertools
import yaml
import os
import pandas as pd
from run_backtest import main as run_backtest_main
from shutil import copyfile

# 参数网格
param_grid = {
    "weight_premium": [0.5, 0.6, 0.7],
    "weight_price": [0.2, 0.3, 0.4],
    "weight_slope": [0.05],
    "weight_r2": [0.05],
}

def run_all_configs():
    keys, values = zip(*param_grid.items())
    combinations = list(itertools.product(*values))
    results = []

    for i, combo in enumerate(combinations):
        combo_dict = dict(zip(keys, combo))
        print(f"\n🔁 Running combo {i + 1}/{len(combinations)}: {combo_dict}")

        # 修改 config.yaml
        with open("config.yaml", "r") as f:
            config = yaml.safe_load(f)
        for k in combo_dict:
            config["selector"]["params"][k] = combo_dict[k]

        config["report"]["output_dir"] = f"./report/exp_{i}"
        with open("config.yaml", "w") as f:
            yaml.dump(config, f)

        # 运行回测
        run_backtest_main()

        # 读取收益数据
        try:
            df = pd.read_csv(os.path.join(config["report"]["output_dir"], "portfolios", "report_normal_1day.csv"))
            returns = df["daily_return"].dropna()
            ann_return = returns.mean() * 252
            ann_std = returns.std() * (252 ** 0.5)
            sharpe = ann_return / ann_std if ann_std > 0 else 0
        except Exception as e:
            print(f"❌ Failed for combo {i}: {e}")
            ann_return = 0
            sharpe = 0

        results.append({**combo_dict, "annual_return": ann_return, "sharpe": sharpe, "exp_id": i})

    df_result = pd.DataFrame(results)
    df_result.to_csv("grid_search_results.csv", index=False)
    print("\n✅ All grid search finished. Results saved to `grid_search_results.csv`.")

if __name__ == "__main__":
    run_all_configs()