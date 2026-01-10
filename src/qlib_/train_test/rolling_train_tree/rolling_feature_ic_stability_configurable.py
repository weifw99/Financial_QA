# rolling_feature_ic_stability_alpha158_dataset.py

import pandas as pd
import numpy as np
from typing import Dict, List
import qlib
from qlib.constant import REG_CN
from qlib.data.dataset import DatasetH
from qlib.contrib.data.handler import Alpha158
from scipy.stats import spearmanr
import json
import os

# ----------------------
# Rolling segments
# ----------------------
def gen_rolling_segments(train_start, test_start, test_end, step=21):
    test_dates = pd.date_range(test_start, test_end, freq="B")
    for i in range(0, len(test_dates), step):
        cur_test_start = test_dates[i]
        cur_test_end = test_dates[min(i + step - 1, len(test_dates) - 1)]
        yield {
            "train": (pd.Timestamp(train_start), cur_test_start - pd.Timedelta(days=1)),
            "test": (cur_test_start, cur_test_end),
        }

# ----------------------
# IC / Rank IC
# ----------------------
def calc_ic(df: pd.DataFrame, feature: str, label: str):
    ic_list = []
    ric_list = []
    # print(df.columns)
    feat_col = ("feature", feature)
    label_col = ("label", "LABEL0")  # Alpha158 默认 label 名
    for dt, g in df.groupby(level="datetime"):
        # print(dt)
        # print(g.columns)
        x = g[feat_col]
        y = g[label_col]
        if x.isna().all() or y.isna().all():
            continue
        ic_list.append(x.corr(y))
        ric, _ = spearmanr(x, y)
        ric_list.append(ric)
    return np.nanmean(ic_list), np.nanmean(ric_list)

# ----------------------
# Analyzer
# ----------------------
class RollingFeatureICAnalyzer:
    def __init__(self, dataset: DatasetH, features: List[str], label: str):
        self.dataset = dataset
        self.features = features
        self.label = label

    def run(self, rolling_segments: List[Dict]):
        records = []
        for seg_id, seg in enumerate(rolling_segments):
            print(f"[Rolling {seg_id}] test={seg['test']}")

            df = self.dataset.prepare(
                col_set=["feature", "label"],
                segments=(seg["test"][0], seg["test"][1]),
            )

            # 计算每个特征 IC
            for feat in self.features:
                if 'LABEL0' == feat:
                    continue
                ic, ric = calc_ic(df, feat, self.label)
                records.append({
                    "segment": seg_id,
                    "feature": feat,
                    "ic": ic,
                    "rank_ic": ric,
                })

        return pd.DataFrame(records)

# ----------------------
# Stable summary
# ----------------------
def summarize_stability(ic_df: pd.DataFrame):
    summary = (
        ic_df.groupby("feature")
        .agg(
            ic_mean=("ic", "mean"),
            ic_std=("ic", "std"),
            ic_win_rate=("ic", lambda x: (x > 0).mean()),
            ric_mean=("rank_ic", "mean"),
            ric_std=("rank_ic", "std"),
            ric_win_rate=("rank_ic", lambda x: (x > 0).mean()),
        )
        .sort_values("ric_mean", ascending=False)
    )
    return summary

# ----------------------
# Main
# ----------------------
def main(
    # instruments_pool="csi300",# zxzz399101
    instruments_pool="zxzz399101",# zxzz399101
    label="Ref($close, -6) / Ref($close, -1) - 1",
    train_start="2015-01-01",
    test_start="2017-01-01",
    test_end="2025-12-31",
    rolling_step=21,
):
    # Qlib init
    qlib.init(provider_uri="~/.qlib/qlib_data/cn_data", region=REG_CN)

    # DatasetH + Alpha158
    handler = Alpha158(instruments=instruments_pool)
    dataset = DatasetH(handler=handler, segments={"train": (train_start, test_start), "test": (test_start, test_end)})

    # 所有特征名
    features = handler.get_cols()
    print(f"[INFO] total Alpha158 features: {len(features)}")

    # Rolling segments
    rolling_segments = list(
        gen_rolling_segments(
            train_start=pd.Timestamp(train_start),
            test_start=pd.Timestamp(test_start),
            test_end=pd.Timestamp(test_end),
            step=rolling_step,
        )
    )

    analyzer = RollingFeatureICAnalyzer(
        dataset=dataset,
        features=features,
        label=label
    )

    ic_df = analyzer.run(rolling_segments)
    print(ic_df.head())

    summary = summarize_stability(ic_df)
    print(summary)

    os.makedirs(f"feature_selection/{instruments_pool}", exist_ok=True)
    summary.to_csv(f"feature_selection/{instruments_pool}/feature_ic_stability.csv")
    ic_df.to_csv(f"feature_selection/{instruments_pool}/feature_ic_rolling_detail.csv", index=False)

    # Stable features
    stable = summary[(summary["ric_mean"] > 0) & (summary["ric_win_rate"] > 0.5)]
    stable_features = stable.index.tolist()
    with open(f"feature_selection/{instruments_pool}/stable_features.json", "w") as f:
        json.dump(stable_features, f, indent=2)
    print(f"[DONE] stable features: {len(stable_features)}")

if __name__ == "__main__":
    main()
