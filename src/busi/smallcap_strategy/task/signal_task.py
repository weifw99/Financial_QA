from datetime import datetime, timedelta

from src.busi.smallcap_strategy.task.seed_message import format_signal_message, send_email, send_wechat_smsg
from src.busi.smallcap_strategy.task.signal_generator import SmallCapSignalGenerator
from src.busi.smallcap_strategy.task.data_loader import load_recent_data

config = dict(
    # smallcap_index=['csi932000', 'sz399101', 'BK1158'],
    smallcap_index=['csi932000', 'BK1158'],
    large_indices=['sh.000300', 'etf_SH159919', 'sh.000016', 'etf_SZ510050', 'sh000905'],
    min_mv=10e8,
    min_profit=0,
    min_revenue=1e8,
    hight_price=100,
    momentum_days=16,
    hold_count_high=15,
)

def main():
    # 1. 加载最近30日的数据（指数 + 个股）
    today = datetime.today()
    stock_data_dict, data_date = load_recent_data()

    for i in range(25):
        data_date = today - timedelta(days=i)
        print(f"数据日期: {data_date.date()}")

        # data_date = today - timedelta(days=3)
    data_date = today
    # data_date = today - timedelta(days=1)
    # 2. 初始化生成器
    generator = SmallCapSignalGenerator(config)
    generator.load_data(stock_data_dict, data_date)

    # 3. 当前持仓（如无自动记录可手动传入）
    current_hold = ["stock_A", "stock_B"]  # 示例

    # 4. 生成信号
    signal = generator.generate_signals(current_hold=current_hold)

    execute_date = datetime.today()
    print(f"📅 执行日期: {execute_date.date()}")
    print(f"📅 数据截止日期: {data_date.date()}")
    print(f"🚨 趋势熔断: {signal['trend_crash']}")
    print(f"🚨 趋势动量: {signal['recovery_scores']}")
    print(f"📊 动量领先: {signal['momentum_ok']}")
    print(f"🔁 动量排名: {signal['momentum_rank']}")
    print(f"🔁 动量排名1: {signal['ranks_comp']}")
    print(f"📥 建议买入: {signal['buy']}")
    print(f"💸 持仓: {signal['current_hold']}")

    # 假设你已有 signal = {...}
    content = format_signal_message(signal, execute_date, data_date)

    print(content)

    # 发送
    # send_email("【小市值策略信号】", content, "your_friend@example.com")
    send_wechat_smsg("小市值策略信号", content)


if __name__ == '__main__':
    main() #