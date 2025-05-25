import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import os

def plot_sharpe_heatmap(df_result, param1="weight_premium", param2="weight_price"):
    df_pivot = df_result.pivot(index=param1, columns=param2, values="sharpe")
    plt.figure(figsize=(8, 6))
    sns.heatmap(df_pivot, annot=True, fmt=".2f", cmap="coolwarm")
    plt.title("Sharpe Ratio Heatmap")
    plt.savefig("sharpe_heatmap.png")
    plt.show()

def plot_best_equity_curve():
    df = pd.read_csv("grid_search_results.csv")
    best_exp_id = df.sort_values("sharpe", ascending=False).iloc[0]["exp_id"]
    print(f"📈 Plotting best equity curve from exp_{int(best_exp_id)}")
    report_path = f"./report/exp_{int(best_exp_id)}/portfolios/report_normal_1day.csv"
    df_portfolio = pd.read_csv(report_path, index_col=0, parse_dates=True)
    df_portfolio["cum_return"] = (1 + df_portfolio["daily_return"]).cumprod()

    plt.figure(figsize=(10, 5))
    df_portfolio["cum_return"].plot()
    plt.title("Best Cumulative Return")
    plt.ylabel("Net Value")
    plt.grid(True)
    plt.savefig("best_equity_curve.png")
    plt.show()

if __name__ == "__main__":
    df_result = pd.read_csv("grid_search_results.csv")
    plot_sharpe_heatmap(df_result)
    plot_best_equity_curve()