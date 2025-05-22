"""量化分析模块

主要功能：
1. 数据清理和预处理
2. 因子计算（技术指标、量价因子、基本面因子、宏观因子）
3. 统计分析（相关性分析、平稳性检验、Hurst指数）

技术栈：
- 计算框架：Pandas / NumPy / Scipy / scikit-learn
- 可视化：Matplotlib / Plotly / Seaborn
- 特征工程：Featuretools
"""

from pathlib import Path

# 定义分析结果存储路径
BASE_DIR = Path(__file__).parent.parent.parent.parent
ANALYSIS_DIR = BASE_DIR / 'data' / 'zh_data' / 'analysis'
ANALYSIS_DIR.mkdir(exist_ok=True)