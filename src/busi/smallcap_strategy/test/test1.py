from busi.smallcap_strategy.test.industry_factor_research import IndustryFactorResearch

import numpy as np
import matplotlib.pyplot as plt


def optimize_combo_score(research, rps_list=[20, 40, 55, 80, 120],
                         future_days=[5, 10, 15, 20],
                         weight_list=[0.1, 0.3, 0.5, 0.7, 0.9]):
    """
    自动优化 combo_score 参数：
    - rps_list: RPS 周期候选
    - future_days: 持有期候选
    - weight_list: hot_score 权重候选（RPS 权重 = 1-hot）

    返回：
    - 最优组合参数
    - 对应 IC_mean
    - 分层收益（Q1-Q5）
    """
    best_ic = -np.inf
    best_params = None
    best_layer = None

    for rps_N in rps_list:
        research.build_rps(rps_N)
        for future_day in future_days:
            research.compute_future_ret(future_day)
            for w_hot in weight_list:
                w_rps = 1 - w_hot
                research.build_combo_score(w_hot, w_rps)

                ic_res = research.calc_ic()
                ic_mean = ic_res['IC_mean']

                if ic_mean > best_ic:
                    best_ic = ic_mean
                    best_params = {
                        'RPS周期': rps_N,
                        'future_day': future_day,
                        'w_hot': w_hot,
                        'w_rps': w_rps,
                        'IC_mean': ic_mean
                    }
                    best_layer = research.layer_analysis()

    print("🔥 最优参数组合：", best_params)
    print("📈 对应分层收益：")
    print(best_layer)

    # 绘制 Q1-Q5 平均收益柱状图
    plt.figure(figsize=(8, 5))
    best_layer['平均收益'].plot(kind='bar')
    plt.title(
        f"Q1-Q5 分层收益（RPS={best_params['RPS周期']}, future_day={best_params['future_day']}, IC={best_params['IC_mean']:.4f})")
    plt.ylabel("未来收益")
    plt.show()

    return best_params, best_layer

def optimize_combo_score_advanced(research,
                                  rps_list=[20, 40, 55, 80, 120],
                                  future_days=[5, 10, 15, 20],
                                  weight_list=[0.1, 0.3, 0.5, 0.7, 0.9],
                                  window_trends=[3, 5, 7, 10, 15, 20, 40, 60, ]):
    """
    自动优化增强版 combo_score_advanced 参数：
    - rps_list: RPS 周期候选
    - future_days: 持有期候选
    - weight_list: hot_score 原因子权重候选（增强版热度权重 = 1 - hot）
    - window_trend: 资金流衍生指标滚动窗口

    返回：
    - 最优组合参数
    - 对应分层收益（Q1-Q5）
    """
    best_ic = -np.inf
    best_params = None
    best_layer = None


    for window_trend in window_trends:
        # 构建资金流衍生特征
        research.build_advanced_flow_features(window_trend=window_trend)

        for rps_N in rps_list:
            research.build_rps(rps_N)
            for future_day in future_days:
                research.compute_future_ret(future_day)
                for w_hot in weight_list:
                    w_advanced = 1 - w_hot
                    research.build_combo_score_advanced(w_hot=w_hot, w_advanced=w_advanced)

                    ic_res = research.calc_ic()
                    ic_mean = ic_res['IC_mean']

                    if ic_mean > best_ic:
                        best_ic = ic_mean
                        best_params = {
                            'RPS周期': rps_N,
                            'future_day': future_day,
                            'w_hot': w_hot,
                            'w_advanced': w_advanced,
                            'IC_mean': ic_mean,
                            'window_trend': window_trend
                        }
                        best_layer = research.layer_analysis()
                    # 分层收益
                    # best_layer, _, _ = research.plot_layers_advanced(combo_col='combo_score_advanced')

    print("🔥 最优参数组合：", best_params)
    print("📈 对应分层收益：")
    print(best_layer)

    # 绘制 Q1-Q5 平均收益柱状图
    plt.figure(figsize=(8, 5))
    best_layer['平均收益'].plot(kind='bar')
    plt.title(
        f"Q1-Q5 分层收益（RPS={best_params['RPS周期']}, future_day={best_params['future_day']}, IC={best_params['IC_mean']:.4f})")
    plt.ylabel("未来收益")
    plt.show()

    return best_params, best_layer


from busi.smallcap_strategy.utils.selected_industries_util import load_industry_price, load_industry_fundflow

base_price_path = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/industry/industry_price"
base_path = "/Users/dabai/liepin/study/llm/Financial_QA/data/zh_data/industry"
# 加载数据
df_price = load_industry_price(base_price_path)
df_flow = load_industry_fundflow(f'{base_path}/industry_flow.csv')



# research = IndustryFactorResearch(df_price, df_flow)
#
#
# # 假设 research 是你已经初始化好的 IndustryFactorResearch 对象
# research.build_hot_factors()  # 先构建热度因子
#
# # 自动优化并可视化
# best_params, best_layer = optimize_combo_score(research)
#

# 1. 初始化
research = IndustryFactorResearch(df_price, df_flow)

# 3. 构建多周期 RPS（可选择 20/40/55/80/120）
# research.build_rps(40)
# research.build_rps(55)
research.build_rps(20)
research.build_hot_factors()
# research.build_combo_score(0.9, 0.1)
research.compute_future_ret(future_day=10)


# 构建资金流衍生指标
research.build_advanced_flow_features(window_trend=3)
# 构建增强版 combo_score
research.build_combo_score_advanced(w_hot=0.95, w_advanced=0.05)
# best_layer = research.layer_analysis()
# print(best_layer)
#
# 4. 拆分因子 IC 分析，剔除负 IC 因子
ic_df = research.factor_ic_analysis(future_day=10)
print(ic_df)

# 5. 自动搜索最优组合权重 & future_day
best_params, best_layer = optimize_combo_score_advanced(research,
                                               rps_list=[5, 10, 20, 40, 55, 80, 120],
                                               future_days=[5,10,15,20],
                                               weight_list=[0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.95]
                                                )

# 6. 输出分层收益 & 可视化
print(best_params)
print(best_layer)


# {'RPS周期': 5, 'future_day': 20, 'w_hot': 0.05, 'w_advanced': 0.95, 'IC_mean': 0.13631835749094978, 'window_trend': 7}
research.build_hot_factors()
research.build_advanced_flow_features(window_trend=7)

research.build_rps(5)
research.compute_future_ret(20)
research.build_combo_score_advanced(w_hot=0.05, w_advanced=0.95)

print(research.get_daily_quantile_details())

# research.build_hot_factors()
# research.build_rps(40)
# research.build_combo_score(0.6, 0.4)
# research.compute_future_ret(10)
#
# print(research.calc_ic())
# print(research.layer_analysis())
#
# research.plot_layers()
