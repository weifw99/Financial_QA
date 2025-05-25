import pandas as pd
import numpy as np
from numpy.polynomial.polynomial import polyfit

def calc_slope_r2(price_df: pd.DataFrame, window: int = 10):
    slopes = pd.DataFrame(index=price_df.index, columns=price_df.columns)
    r2s = pd.DataFrame(index=price_df.index, columns=price_df.columns)

    for col in price_df.columns:
        for i in range(window, len(price_df)):
            y = price_df[col].iloc[i-window:i]
            x = np.arange(window)
            if y.isnull().any():
                slopes.iloc[i, slopes.columns.get_loc(col)] = np.nan
                r2s.iloc[i, r2s.columns.get_loc(col)] = np.nan
                continue
            coefs = polyfit(x, y.values, 1)
            slopes.iloc[i, slopes.columns.get_loc(col)] = coefs[1]
            y_fit = coefs[0] + coefs[1] * x
            ss_res = np.sum((y.values - y_fit) ** 2)
            ss_tot = np.sum((y.values - np.mean(y.values)) ** 2)
            r2 = 1 - ss_res / ss_tot if ss_tot != 0 else np.nan
            r2s.iloc[i, r2s.columns.get_loc(col)] = r2

    return slopes, r2s