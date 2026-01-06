# qlib_html_report.py

import base64
import pickle
from typing import Iterable

import matplotlib.pyplot as plt
import pandas as pd
from io import BytesIO

from qlib.contrib.report import analysis_model

from qlib_.test.qlib_help.util_config import load_config
from qlib_.test.qlib_help.util_qlib import init_qlib, get_latest_recorder

import os
from qlib.contrib.report import analysis_position
import plotly.io as pio
from plotly.subplots import make_subplots
from plotly.graph_objs import Figure
from datetime import datetime


def generate_html_report(port_analysis,
                         report_normal_df,
                         pred_label_df,
                         output_dir = "report_output",
                         report_filename = f"strategy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html",
                         ):
    """

    :param port_analysis:  recorder.load_object("portfolio_analysis/port_analysis_1day.pkl")
    :param report_normal_df: recorder.load_object("portfolio_analysis/report_normal_1day.pkl")
    :param pred_label_df: index is **pd.MultiIndex**, index name is **[instrument, datetime]**; columns names is **[score, label]**.
    :return:
    """

    # 假设你已经有了这几个输入文件或字典
    # position_df: 持仓数据 DataFrame
    # report_normal_df: 报告数据（如收益、IC）DataFrame
    # pred_label_df: 预测值和真实标签 DataFrame

    # 定义保存路径
    os.makedirs(output_dir, exist_ok=True)
    # html_path = os.path.join(output_dir, f"strategy_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html")
    html_path = os.path.join(output_dir, report_filename)

    # 图表组装容器
    figures = []

    # 1️⃣ 收益类图表
    figures.append(("收益指标", analysis_position.report_graph(report_normal_df, show_notebook=False)[0]))

    # 2️⃣ 风险类图表
    # analysis_position.risk_analysis_graph(port_analysis, report_normal_df)
    figures.append(("风险分析", analysis_position.risk_analysis_graph(port_analysis, report_normal_df, show_notebook=False)))

    # 3️⃣ 因子有效性分析
    figures.append(("IC 时间序列", analysis_position.score_ic_graph(pred_label_df, show_notebook=False)[0]))

    # ---- HTML 报告拼接 ----
    html_parts = []
    html_parts.append(f"<html><head><meta charset='utf-8'><title>策略分析报告</title></head><body>")
    html_parts.append(f"<h1>策略分析报告（{datetime.now().strftime('%Y-%m-%d')}）</h1>")

    # 按模块写入图表
    for title, fig in figures:
        html_parts.append(f"<h2>{title}</h2>")
        print(f"✅ 生成图表：{title}")
        if isinstance(fig, list):
            for sub_fig in fig:
                html_parts.append(pio.to_html(sub_fig, include_plotlyjs='cdn', full_html=False))
        elif isinstance(fig, Figure):
            html_parts.append(pio.to_html(fig, include_plotlyjs='cdn', full_html=False))
        elif isinstance(fig, Iterable):
            # raise ValueError("Invalid figure type")
            for i, sub_fig in enumerate(fig):
                html_parts.append(pio.to_html(sub_fig, include_plotlyjs=False, full_html=False))
        else:
            continue
        # html_parts.append(pio.to_html(fig, include_plotlyjs='cdn', full_html=False))

    html_parts.append("</body></html>")

    # 写入 HTML 文件
    with open(html_path, "w", encoding="utf-8") as f:
        f.write("\n".join(html_parts))

    print(f"✅ 报告已生成：{html_path}")
