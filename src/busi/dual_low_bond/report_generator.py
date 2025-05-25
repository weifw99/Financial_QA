import pandas as pd
import matplotlib.pyplot as plt
from jinja2 import Template
import os

def generate_html_report(report_dir="report"):
    pos = pd.read_csv(os.path.join(report_dir, "adjusted_weights.csv"), index_col=0, parse_dates=True)
    price = pd.read_csv(os.path.join(report_dir, "price.csv"), index_col=0, parse_dates=True)

    # 简单收益曲线计算
    returns = (pos.shift(1) * price.pct_change()).sum(axis=1)
    cum_returns = (1 + returns).cumprod()

    plt.figure(figsize=(10, 4))
    plt.plot(cum_returns, label="策略净值")
    plt.title("策略净值曲线")
    plt.grid(True)
    plt.savefig(os.path.join(report_dir, "cum_return.png"))

    with open("template.html") as f:
        html_template = Template(f.read())
    with open(os.path.join(report_dir, "report.html"), "w") as f:
        f.write(html_template.render(title="策略回测报告", image_path="cum_return.png"))