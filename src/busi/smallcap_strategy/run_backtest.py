# run_backtest.py - 主回测运行脚本，加载配置、数据，执行策略，记录结果
import backtrader as bt
import yaml
import mlflow
import os
from utils.data_loader import load_stock_data
from strategies.smallcap_strategy import SmallCapStrategy

def load_config(path='config/config.yaml'):
    """从 YAML 配置文件加载策略参数"""
    with open(path, 'r') as f:
        return yaml.safe_load(f)

def run():
    config = load_config()
    cerebro = bt.Cerebro()

    # 添加策略及其参数
    cerebro.addstrategy(SmallCapStrategy, **config['strategy'])

    # 加载所有股票与指数数据
    datafeeds = load_stock_data('data')
    for feed in datafeeds:
        cerebro.adddata(feed)

    # 设置滑点和佣金
    cerebro.broker.set_slippage_perc(perc=0.00015)  # 买卖滑点各 0.015%
    cerebro.broker.setcommission(commission=0.00025)  # 万 2.5 的佣金
    cerebro.broker.set_cash(1000000)  # 初始资金

    # 设置 MLflow 实验记录
    '''
    mlflow.set_tracking_uri("outputs/mlruns")
    with mlflow.start_run(run_name="smallcap_strategy"):
        result = cerebro.run()
        port_value = cerebro.broker.getvalue()
        mlflow.log_metric("FinalValue", port_value)

        # 绘制净值图并记录
        import matplotlib.pyplot as plt
        cerebro.plot(style='candlestick')
        plt.savefig("outputs/plots/net_value.png")
        mlflow.log_artifact("outputs/plots/net_value.png")
    '''
    result = cerebro.run()
    port_value = cerebro.broker.getvalue()
    print("Final Portfolio Value: %.2f" % port_value)

    # 绘制净值图并记录
    import matplotlib.pyplot as plt
    cerebro.plot(style='candlestick')
    plt.savefig("outputs/plots/net_value.png")

if __name__ == '__main__':
    run()