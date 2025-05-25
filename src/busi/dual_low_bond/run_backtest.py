
import qlib
from qlib.workflow.cli import run
from stop_loss_executor import StopLossExecutor
from momentum_factor import calc_slope_r2
import yaml
import pandas as pd
import os

def main():
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region="cn")

    with open("config.yaml", "r") as f:
        config = yaml.safe_load(f)

    # 运行回测（qlib workflow根据 config.yaml）
    run("config.yaml")

    report_dir = config['report']['output_dir']
    pos_path = os.path.join(report_dir, "weights.csv")
    price_path = os.path.join(report_dir, "price.csv")

    positions = pd.read_csv(pos_path, index_col=0, parse_dates=True)
    prices = pd.read_csv(price_path, index_col=0, parse_dates=True)

    # 计算动量因子
    slopes, r2s = calc_slope_r2(prices, window=10)
    # 注意：通常需要在回测策略内部使用 slopes, r2s 与价格、溢价率合并形成score_df
    # 此处为简化示例，实际策略文件中应直接使用

    # 止损处理
    if config.get("stop_loss", {}).get("enable", False):
        drop_pct = config["stop_loss"].get("drop_pct", 0.08)
        max_dd = config["stop_loss"].get("max_drawdown", 0.15)
        sl = StopLossExecutor(drop_pct=drop_pct, max_drawdown=max_dd)
        adjusted_positions = sl.apply(positions, prices)
        adjusted_positions.to_csv(os.path.join(report_dir, "adjusted_weights.csv"))
        print(f"止损后持仓权重已保存: {os.path.join(report_dir, 'adjusted_weights.csv')}")

if __name__ == "__main__":
    main()