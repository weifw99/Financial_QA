import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

import requests


def send_email(subject, body, to_email):
    from_email = "your_email@example.com"
    from_name = "策略助手"
    password = "your_email_app_password"  # 注意是邮箱的"应用专用密码"

    msg = MIMEText(body, 'plain', 'utf-8')
    msg['From'] = formataddr((from_name, from_email))
    msg['To'] = to_email
    msg['Subject'] = subject

    try:
        server = smtplib.SMTP_SSL("smtp.example.com", 465)  # 根据邮箱服务提供商填写
        server.login(from_email, password)
        server.sendmail(from_email, [to_email], msg.as_string())
        server.quit()
        print("📧 邮件发送成功")
    except Exception as e:
        print("❌ 邮件发送失败:", e)


def send_wechat_smsg(title, content):
    sendkey = "SCT286201TIKloVj1iTFFpYQ6z8kS4nKyc"
    url = f"https://sctapi.ftqq.com/{sendkey}.send"
    data = {
        'title': title,
        'desp': content
    }
    try:
        res = requests.post(url, data=data)
        print("📲 微信通知成功:", res.json())
    except Exception as e:
        print("❌ 微信通知失败:", e)


def format_signal_message(signal, exe_date, data_date):
    header = f"""\
### 📈 小市值策略信号

- **执行日期**：{exe_date.strftime('%Y-%m-%d')}
- **数据截止**：{data_date.strftime('%Y-%m-%d')}
- **趋势熔断**：{'🚨 是' if signal['trend_crash'] else '✅ 否'} （最近 3 天至少 2 天跌超 2.5%，或者平均跌超 3%。且波动率较低。）
- **动量领先**：{'🚀 是' if signal['momentum_ok'] else '📉 否'} （领先：小市值组合动量排名在top2，或则跌出top2，但是最近三天的动量趋势向上）

- **动量趋势（顺序-今天/昨天/前天）**：{signal['recovery_scores']}
"""

    # 动量排名表格
    momentum_md = "| 指数名称 | 动量收益 |\n| :-- | --: |\n"
    for name, val in signal['momentum_rank']:
        momentum_md += f"| {name} | {val:.2%} |\n"

    # 建议买入表格（包含是否已持有 + 收盘价）
    if signal['buy']:
        i = 1
        buy_md = "| 编号 | 股票代码 | 市值 (亿) | 当前已持仓 | 最新收盘价 | 无负面消息（最近的Q） |\n| :-- | :-- | --: | :--: | --: | --: |\n"
        for stock, mv, held, close_price, filter_flag in signal['buy']:
            held_str = "✅ 是" if held else "❌ 否"
            filter_str = "✅ 是" if filter_flag else "❌ 否"
            price_str = f"{close_price:.2f}" if close_price is not None else "N/A"
            buy_md += f"| {i} | {stock} | {mv / 1e8:.2f} | {held_str} | {price_str} | {filter_str} |\n"
            i += 1
    else:
        buy_md = "无"

    # 当前持仓列表
    if signal['current_hold']:
        hold_md = "\n".join([f"- {stock}" for stock in signal['current_hold']])
    else:
        hold_md = "无"

    action_md = f"""\
**📥 建议买入：**\n
{buy_md}

**💼 当前持仓：**\n
{hold_md}
"""

    return header + "\n" + momentum_md + "\n" + action_md
