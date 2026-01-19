from datetime import datetime, timedelta

from busi.model_result_test_strategy.utils.data_loader import load_stock_data
from src.busi.model_result_test_strategy.task.seed_message import format_signal_message, send_email, send_wechat_smsg
from src.busi.model_result_test_strategy.task.signal_generator import SmallCapSignalGenerator

config = dict(
    min_mv=10e8,
    min_profit=0,
    min_revenue=1e8,
    hight_price=50,
    hold_count_high=15,
)

def main():
    # 1. 加载最近30日的数据（指数 + 个股）
    today = datetime.today()

    rank_model_result_path = [
        '/Users/dabai/liepin/study/llm/Financial_QA/data/qlib_exp/small/small_rank_result.csv',
    ]
    class_model_result_path = [
        '/Users/dabai/liepin/study/llm/Financial_QA/data/qlib_exp/small/small_class_result.csv',
    ]
    extend_datas = {
        1000: (rank_model_result_path, class_model_result_path)
    }
    to_idx = datetime.now()
    from_idx = to_idx - timedelta(days=30)

    # 加载所有股票与指数数据
    _, data_dfs = load_stock_data(from_idx, to_idx, extend_datas)


    for i in range(25):
        data_date = today - timedelta(days=i)
        print(f"数据日期: {data_date.date()}")

        # data_date = today - timedelta(days=3)
    data_date = today
    # data_date = today - timedelta(days=1)
    # 2. 初始化生成器
    generator = SmallCapSignalGenerator(config)
    generator.load_data(data_dfs, data_date)

    # 4. 生成信号
    signal = generator.generate_signals()

    execute_date = datetime.today()
    signal['execute_date'] = execute_date.date().strftime('%Y-%m-%d')
    signal['date_date'] = generator.stock_data_date.date().strftime('%Y-%m-%d')

    print(f"📅 执行日期: {execute_date.date()}")
    print(f"📅 数据截止日期: {generator.stock_data_date.date()}")
    print(f"📥 建议买入: {signal['buy']}")

    # 假设你已有 signal = {...}
    content = format_signal_message(signal, execute_date, generator.stock_data_date.date())

    print(content)

    print(signal)

    # 发送
    send_email("小市值策略信号", str(signal), "18910770963@163.com")
    send_email("小狮子明细", content, "837602401@qq.com", is_md= True)
    # send_email("小狮子明细", content, "77946997@qq.com", is_md= True)
    send_wechat_smsg("小市值策略信号", content)


if __name__ == '__main__':
    main() #