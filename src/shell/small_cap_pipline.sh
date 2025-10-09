#!/bin/bash

# 日志输出函数
log() {
    echo "[$(date '+%Y-%m-%d %H:%M:%S')] $1"
}

# 配置路径
PROJECT_DIR="/Users/dabai/liepin/study/llm/Financial_QA"
QLIB_DATA_DIR="$PROJECT_DIR/data/qlib_data"
SOURCE_DATA_DIR="$QLIB_DATA_DIR/cn_data"

TARGET_OUTPUT_DIR="/Users/dabai/.qlib/qlib_data/cn_data"  # 替换为实际的目标目录
CONDA_ENV_NAME="env_py3_12"                # 替换为你的 Conda 环境名

# 检查目标目录是否存在，不存在则创建
if [ ! -d "$TARGET_OUTPUT_DIR" ]; then
    log "目标目录不存在，正在创建: $TARGET_OUTPUT_DIR"
    mkdir -p "$TARGET_OUTPUT_DIR"
fi

# 初始化 Conda 并激活环境
log "初始化 Conda 并激活环境: $CONDA_ENV_NAME"
eval "$(conda shell.bash hook)"
conda activate "$CONDA_ENV_NAME"

if [ $? -ne 0 ]; then
    log "Conda 环境激活失败，请确认环境名称是否正确或 Conda 是否已安装。"
    exit 1
fi

# 切换到项目根目录并执行 python 命令
cd "$PROJECT_DIR" || { log "切换目录到 $PROJECT_DIR 失败"; exit 1; }
log "当前目录: $(pwd)"

log "开始执行 所有 A 股数据任务 src.data.zh_data.zh_run_sync ..."
python -m src.data.zh_data.zh_run_sync

log "开始执行 指数成分股拉取任务 src.data.zh_data.index.component_stock_pull ..."
python -m src.data.zh_data.index.component_stock_pull

log "开始执行 etf数据同步任务 src.busi.etf_.etf_data..."
python -m src.busi.etf_.etf_data_code

log "开始执行 重要指数数据任务 src.data.zh_data.index.important_indices_sh_sz ..."
python -m src.data.zh_data.index.important_indices_sh_sz

# log "开始执行 行业数据任务 src.data.zh_data.industry.industry_fundflow_task ..."
# python -m src.data.zh_data.industry.industry_fundflow_task

# log "开始执行 行业数据任务 src.data.zh_data.industry.stock_industry_task ..."
# python -m src.data.zh_data.industry.stock_industry_task

# log "开始执行 行业数据任务 src.data.zh_data.industry.stock_industry_hist_task ..."
# python -m src.data.zh_data.industry.stock_industry_hist_task

log "开始执行 小市值信号生成任务 src.busi.smallcap_strategy.task.signal_task ..."
python -m src.busi.smallcap_strategy.task.signal_task

log "脚本执行完成。"
