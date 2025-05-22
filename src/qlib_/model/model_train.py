import logging

from qlib.model import Model
from qlib.workflow import R
from qlib.workflow.record_temp import SignalRecord
from qlib.utils import init_instance_by_config
from data.dataset import get_dataset

# 配置日志，忽略低于 ERROR 级别的日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)



def train_model(model_config):
    """
    训练模型
    :param model_config: 模型配置
    :return: 记录 ID
    """
    logger.info("准备数据集...")
    dataset = get_dataset()
    model: Model = init_instance_by_config(model_config)

    logger.info("开始训练模型...")
    with R.start(experiment_name="tutorial_exp"):
        logger.info("数据准备...")
        data = dataset.prepare("train", col_set=["feature", "label"])
        if data.isnull().values.any():
            logger.warning("训练数据中包含 NaN 值。{data}")
        
        model.fit(dataset)
        R.save_objects(trained_model=model)

        rec = R.get_recorder()
        
        logger.info("生成信号记录...")
        sr = SignalRecord(model, dataset, rec)
        sr.generate()

    logger.info("模型训练完成。")
    return rec.id