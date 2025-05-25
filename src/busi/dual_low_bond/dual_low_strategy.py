from qlib.contrib.strategy.rule_strategy import TopkSelector
import pandas as pd

class DualLowSelector(TopkSelector):
    def __init__(self, weight_premium=0.7, weight_price=0.3, topk=10):
        super().__init__(topk=topk)
        self.weight_premium = weight_premium
        self.weight_price = weight_price

    def __call__(self, score_df: pd.DataFrame) -> pd.Series:
        df = score_df.copy()
        df = df[(df['close'] >= 90) & (df['close'] <= 110)]  # 价格过滤示例
        df["score"] = -self.weight_premium * df["convert_premium"] - self.weight_price * df["close"]
        df = df.sort_values("score")
        selected = df.head(self.topk)
        weights = pd.Series(0.0, index=df.index)
        weights.loc[selected.index] = 1.0 / self.topk
        return weights