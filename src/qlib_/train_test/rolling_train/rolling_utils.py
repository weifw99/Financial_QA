import qlib
import pandas as pd
import qlib
import pandas as pd
from qlib.workflow.task.gen import RollingGen
from qlib.workflow.task.gen import RollingGen
from qlib.workflow.record_temp import SignalRecord, SigAnaRecord
from qlib.workflow.recorder import Recorder
from qlib.config import C


def run_rolling_experiment(
    config_path: str,
    provider_uri: str,
    rolling_step: int = 252,
    rolling_type: str = "expanding",
):
    """
    使用 RollingGen 运行 rolling 实验，并汇总结果

    参数:
    - config_path: str
        指向 workflow yaml 配置文件的路径（例如 workflow_config_lightgbm_Alpha158.yaml）。
        该 yaml 描述“一次 workflow 如何跑”（dataset/model/strategy/record 等）。
        注意：rolling 行为由 RollingGen 控制，而不是写入 yaml。

    - provider_uri: str
        Qlib 数据目录（provider location），例如 "~/.qlib/qlib_data/cn_data" 的绝对或相对路径。
        在调用 qlib.init(...) 时将被用来查找历史行情、因子等数据。

    - rolling_step: int (默认 252)
        RollingGen 的 step 参数。表示每次 rolling 窗口向前推进的天数（交易日），
        252 约等于一整年（按交易日计）。你可以改成 21（1 个月）或 63（1 季度）等。

    - rolling_type: str (默认 "expanding")
        Rolling 窗口类型，可选 "expanding" 或 "sliding"：
          - "expanding": 训练窗口随时间扩大（常用于希望利用越多历史越好情形）
          - "sliding": 训练窗口长度固定（常用于非平稳市场或想限制历史长度）
    """

    # ========== 1. 初始化 Qlib ==========
    # qlib.init 会初始化数据 provider、注册 region 等。务必保证 provider_uri 路径存在并包含 qlib 数据。
    qlib.init(
        provider_uri=provider_uri,
        region="cn",
    )

    # ========== 2. 构造 RollingGen ==========
    # RollingGen 是官方推荐的“任务级别 rolling 生成器”，会根据 rolling_step/rolling_type 生成多个子任务（每个子任务对应一轮 rolling）
    # step: 每轮向前移动多少交易日； rolling_type: "expanding" 或 "sliding"
    rolling_gen = RollingGen(
        step=rolling_step,
        rtype=rolling_type,
    )

    # ========== 3. 运行 Rolling ==========
    # R.run 接受 config_path（yaml）并且可以接受 rolling_gen 来按轮次执行 workflow。
    # 返回值 recs 是一个 Recorder 对象列表（List[Recorder]），每个 Recorder 对应一轮 rolling 完成后的记录器（含预测、指标、回测结果等）
    rolling_gen.run()
    recs = None
    # run(
    #     config_path,
    #     rolling_gen=rolling_gen,
    # )

    # recs: List[Recorder]
    print(f"Total rolling rounds: {len(recs)}")

    # ============================================================
    # 4️⃣ 汇总所有 rolling 的 prediction（用于回测）
    # ============================================================
    # 说明：
    # - Recorder 在运行过程中会把预测存为一个对象，通常 key 名称会是 "pred.pkl"（取决于 record 配置）。
    # - pred 的常见结构：DataFrame，索引通常是 MultiIndex (datetime, instrument)，列里有 score 或 pred（模型输出分数）
    # - 我们给每条记录加上 rolling_id，方便后续回测时按轮过滤或合并时去重/权重分配
    all_preds = []

    for i, rec in enumerate(recs):
        try:
            # load_object("pred.pkl")：从 Recorder 中加载预测结果对象
            pred = rec.load_object("pred.pkl")
            # 增加一列以标注属于第几轮 rolling，便于后续分组/过滤
            pred["rolling_id"] = i
            all_preds.append(pred)
        except Exception:
            # 如果某轮没有预测文件，会跳过并输出 warning
            # 常见原因：record 配置没有保存 pred，或该轮执行异常导致未写入
            print(f"[WARN] rolling {i} has no prediction")

    # 将每轮预测合并为一个大表并按时间排序
    # 注意：如果 all_preds 为空会报错，这里假设至少有一次成功预测
    if all_preds:
        all_preds_df = pd.concat(all_preds).sort_index()
    else:
        all_preds_df = pd.DataFrame()
        print("[WARN] no predictions were collected from any rolling rounds")

    # ============================================================
    # 5️⃣ 汇总每一轮 rolling 的指标（IC / Rank IC）
    # ============================================================
    # 说明：
    # - Recorder 可能会保存 ic.pkl / metrics 文件，里面包含按日期的 IC/RankIC 或汇总值
    # - 具体 key 名称取决于你在 yaml record 中的配置；这里使用 "ic.pkl" 作为常见示例
    metrics = []

    for i, rec in enumerate(recs):
        try:
            # 期望 ic 是一个 DataFrame 或 Series，记录每个日期的 IC 指标
            ic = rec.load_object("ic.pkl")
            # 标注是哪一轮 rolling
            # 如果 ic 是 Series 或 DataFrame，下面一行会给它增加一列/字段
            try:
                ic["rolling_id"] = i
            except Exception:
                # 如果 ic 不是 DataFrame（极少见），我们包成 DataFrame
                ic = pd.DataFrame({"value": ic, "rolling_id": i})
            metrics.append(ic)
        except Exception:
            print(f"[WARN] rolling {i} has no ic record")

    if metrics:
        metrics_df = pd.concat(metrics)
    else:
        metrics_df = pd.DataFrame()
        print("[WARN] no rolling ic metrics were collected")

    # ============================================================
    # 6️⃣ 对比 Train / Test 指标（稳定性分析）
    # ============================================================
    # 说明：
    # - 我们希望把每一轮的 train 与 test 指标读出并放到一张表里。
    # - 常见 Recorder 中用于保存指标的 key 可能是 "train_metrics.pkl" / "test_metrics.pkl" 或类似名称（这取决于 yaml 的 record 配置）
    # - 这里假设每个文件是一个 dict 或 Series，包含 "IC"、"Rank IC" 等汇总指标
    stability = []

    for i, rec in enumerate(recs):
        try:
            train_metrics = rec.load_object("train_metrics.pkl")
            test_metrics = rec.load_object("test_metrics.pkl")

            # 下面取出常见的指标：IC / Rank IC
            stability.append(
                {
                    "rolling_id": i,
                    # 如果 train_metrics 是 dict 或 Series，则这种索引方式可用；否则需按实际结构调整
                    "train_ic": train_metrics.get("IC") if isinstance(train_metrics, dict) else train_metrics["IC"],
                    "test_ic": test_metrics.get("IC") if isinstance(test_metrics, dict) else test_metrics["IC"],
                    "train_rank_ic": train_metrics.get("Rank IC") if isinstance(train_metrics, dict) else train_metrics["Rank IC"],
                    "test_rank_ic": test_metrics.get("Rank IC") if isinstance(test_metrics, dict) else test_metrics["Rank IC"],
                }
            )
        except Exception:
            # 有些 recorder 可能没有分别记录 train/test 指标，或命名不同；跳过并 warn
            print(f"[WARN] rolling {i} has no train/test metrics or file names differ")

    stability_df = pd.DataFrame(stability)

    # 最终返回一个字典，包含合并的预测、按轮指标、稳定性对比表以及原始的 Recorder 列表（方便用户后续自行深入分析）
    return {
        "all_predictions": all_preds_df,   # DataFrame: 合并后的所有轮次预测（可直接用于回测）
        "rolling_metrics": metrics_df,     # DataFrame: 每轮的按日 IC/RankIC 等（用于绘图分析）
        "stability": stability_df,        # DataFrame: 每轮 train/test 汇总指标对比（用于稳定性判断）
        "recorders": recs,                # List[Recorder]: 每一轮的 Recorder，便于手动加载其它对象
    }
