以下是整个项目的整体流程梳理。这个流程从ETF候选池的筛选开始，到历史数据下载、回测策略的实现，再到结果的分析和可视化，涵盖了从数据获取到模型评估的完整步骤。

⸻

✅ 整体流程概述

1. ETF候选池筛选（etf_filter.py）
	•	根据财务指标或动量策略筛选合适的ETF。
	•	产生两个CSV文件：top_etf_simple.csv 和 top_etf_r2.csv。

筛选逻辑：
	•	top_etf_simple.csv：基于简单的财务指标（如净资产、收益率等）筛选出的候选ETF。
	•	top_etf_r2.csv：基于动量得分（收益率 * R²，R²为过去半年价格的拟合度）筛选出的候选ETF。

2. 下载历史数据（download_etf_history.py）
	•	从AkShare拉取选定ETF的历史数据（包含收盘价、开盘价、成交量等信息）。
	•	数据保存到data/文件夹，每个ETF的数据保存为单独的CSV文件。

主要功能：
	•	从top_etf_simple.csv 和 top_etf_r2.csv中读取ETF代码。
	•	下载每个ETF的历史数据，并保存为<ETF代码>.csv文件。

3. 回测策略（etf_momentum_bt.py）
	•	根据下载的ETF数据，使用Backtrader框架进行回测。
	•	使用动量策略，按每月调仓选出Top K ETF。
	•	引入止损功能，止损比例为用户设定。
	•	与沪深300指数（000300.SH）进行净值对比。

策略概述：
	•	动量策略：选择过去表现最好的ETF，按动量策略（收益率 * R²）调整投资组合。
	•	每月调仓：根据用户设定的调仓日进行ETF的持仓更新。
	•	止损：设定止损比例，一旦亏损达到止损比例即卖出ETF。

回测过程：
	1.	读取历史数据和基准（沪深300指数）。
	2.	每月根据动量评分选取Top K ETF。
	3.	对比策略净值和基准净值。

4. 回测结果分析与可视化
	•	回测结束后，使用Backtrader的pyfolio分析器输出回报分析。
	•	绘制策略净值与沪深300指数的对比图。

可视化输出：
	•	策略和基准（沪深300指数）的净值曲线。
	•	回测结果的回报分析（如年度收益率、最大回撤等）。

5. 导出结果（可选）
	•	导出回测结果和回报分析，以便进一步的分析或报告使用。

⸻

✅ 步骤详解

1. 筛选ETF候选池

运行etf_filter.py脚本，选择筛选方式（如：基于财务指标或动量评分）来生成ETF候选池。

python etf_filter.py

输出文件：
	•	top_etf_simple.csv：基于简单筛选条件得到的ETF列表。
	•	top_etf_r2.csv：基于动量得分（收益率 * R²）得到的ETF列表。

2. 下载ETF历史数据

运行download_etf_history.py脚本，批量下载候选池中ETF的历史数据（数据保存在data/文件夹中）。

python download_etf_history.py

3. 运行回测策略

运行etf_momentum_bt.py脚本，加载候选池数据和ETF历史数据，运行回测。

回测参数：
	•	--pool：选择候选池，simple 或 r2。
	•	--topk：持仓ETF数量（默认5只ETF）。
	•	--rebalance_day：每月调仓的日期（默认调仓日为1号）。
	•	--stoploss：止损比例（默认5%）。

python etf_momentum_bt.py --pool simple --topk 5 --rebalance_day 1 --stoploss 0.05

4. 查看回测结果

回测完成后，Backtrader会绘制策略净值与沪深300指数的净值对比图，同时输出回报分析（如夏普比率、最大回撤等）。

⸻

✅ 项目文件结构

etf_strategy_project/
│
├── data/                          # 存放ETF历史数据的文件夹
│   └── 159915.csv                 # 各ETF的历史数据文件（CSV）
│
├── top_etf_simple.csv             # 财务指标筛选的ETF候选池
├── top_etf_r2.csv                 # 动量得分筛选的ETF候选池
│
├── etf_filter.py                  # 筛选ETF候选池的脚本
├── download_etf_history.py        # 下载ETF历史数据的脚本
└── etf_momentum_bt.py             # 回测策略脚本



⸻

✅ 可选扩展
	1.	数据增强：增加更多的数据来源或加入基本面数据（如PE、PB等），优化筛选策略。
	2.	回测优化：可加入风险控制、动态调整仓位、更多的止损策略等。
	3.	多策略组合：根据不同的策略（如均值回归、动量策略等）进行组合回测，比较不同策略的效果。
