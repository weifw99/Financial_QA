import pandas as pd

class StopLossExecutor:
    def __init__(self, drop_pct=0.08, max_drawdown=0.15):
        self.drop_pct = drop_pct
        self.max_drawdown = max_drawdown

    def apply(self, positions: pd.DataFrame, prices: pd.DataFrame) -> pd.DataFrame:
        new_positions = positions.copy()
        for col in positions.columns:
            price_series = prices[col]
            pos_series = positions[col]
            entry_price = None
            peak_price = None
            for i in range(len(pos_series)):
                if pos_series.iloc[i] > 0:
                    if entry_price is None:
                        entry_price = price_series.iloc[i]
                        peak_price = entry_price
                    else:
                        peak_price = max(peak_price, price_series.iloc[i])
                    drop = (price_series.iloc[i] - entry_price) / entry_price
                    drawdown = (price_series.iloc[i] - peak_price) / peak_price
                    if drop < -self.drop_pct or drawdown < -self.max_drawdown:
                        new_positions.iloc[i:, col] = 0
                        break
        return new_positions