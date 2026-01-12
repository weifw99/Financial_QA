import smtplib
from email.mime.text import MIMEText
from email.utils import formataddr

import requests


import imaplib
import email
from email.header import decode_header
import re

def strip_html_tags(html):
    """简单的 HTML 转纯文本"""
    return re.sub(r'<[^>]+>', '', html)

def receive_latest_email(sender_filter=None, subject_filter=None):
    """
    获取邮箱最新邮件（可按发件人或主题过滤）
    sender_filter: 精确过滤发件人邮箱
    subject_filter: 主题包含关键字（支持中文）
    """

    # ==============================
    # 邮箱配置（请改这里）
    # ==============================
    imap_server = "imap.163.com"  # QQ邮箱 IMAP 服务器
    email_user = "18910770963@163.com"  # 你的邮箱
    email_password = "RWfmJRKShxvqnRBu"  # ⚠ 必须使用授权码，不是登录密码
    mailbox = "INBOX"  # 默认收件箱
    # ==============================

    try:
        # 1) 连接 IMAP
        # mail = imaplib.IMAP4_SSL(imap_server, 993)
        mail = imaplib.IMAP4_SSL(imap_server)
        mail.login(email_user, email_password)

        # 解决网易邮箱报错：Unsafe Login. Please contact kefu@188.com for help
        imaplib.Commands["ID"] = ('AUTH',)
        args = ("name", email_user, "contact", email_user, "version", "1.0.0", "vendor", "myclient")
        mail._simple_command("ID", str(args).replace(",", "").replace("\'", "\""))
        mail_dir = mail.list()
        print(mail_dir)

        status, _ = mail.select(mailbox)
        # status, _ = mail.select('inbox')
        print(f"邮箱状态：{status}")
        if status != "OK":
            raise Exception(f"无法选择邮箱：{mailbox}")
        # status, messages = mail.search(None, "ALL")
        # email_ids = messages[0].split()
        # latest_email = None

        # 获取所有邮件 UID
        status, data = mail.uid('search', None, "ALL")
        if status != "OK":
            raise Exception("无法获取邮件列表")

        all_uids = data[0].split()
        if not all_uids:
            print("⚠️ 邮箱为空")
            return None

        print(len(all_uids))

        # 取最近 latest_n 封邮件
        latest_n = 20
        recent_uids = all_uids[-latest_n:]

        for uid in reversed(recent_uids):  # 从最新往前遍历
            status, msg_data = mail.uid('fetch', uid, "(RFC822)")
            if status != "OK":
                continue

            msg = email.message_from_bytes(msg_data[0][1])

            # 解析主题
            subject, enc = decode_header(msg["Subject"])[0]
            if isinstance(subject, bytes):
                subject = subject.decode(enc or "utf-8", errors="ignore")

            # 解析发件人
            from_addr = msg.get("From", "")

            # 过滤
            if sender_filter and sender_filter not in from_addr:
                continue
            if subject_filter and subject_filter not in subject:
                continue
            print(f"发件人: {from_addr}")

            # 解析正文
            charset = msg.get_content_charset()
            body = msg.get_payload(decode=True).decode(charset or "utf-8", errors="ignore")
            print(f"body: {type(body)}", body)

            mail.logout()
            import ast
            obj = ast.literal_eval(body.strip())
            return obj
            # return {
            #     "subject": subject.strip(),
            #     "from": from_addr.strip(),
            #     "body": obj
            # }

        mail.logout()
        print("⚠️ 最近邮件中没有匹配的邮件")
        return None

    except Exception as e:
        print("❌ 收取邮件失败:", e)
    return None



def send_email(subject, body, to_email, is_md= False):
    from_email = "837602401@qq.com"
    from_name = "837602401"
    password = "uuwfzpsylmcqbgac"  # 注意是邮箱的"应用专用密码"

    if is_md:
        # 将Markdown转换为HTML
        import markdown
        html_content = markdown.markdown(body)
        import re
        html_content = re.sub(r'\n+', '<br>', html_content)
        msg = MIMEText(html_content, 'html', 'utf-8')
        msg['From'] = formataddr((from_name, from_email))
        msg['To'] = to_email
        msg['Subject'] = subject

    else:
        msg = MIMEText(body, 'plain', 'utf-8')
        msg['From'] = formataddr((from_name, from_email))
        msg['To'] = to_email
        msg['Subject'] = subject

    try:
        server = smtplib.SMTP_SSL("smtp.qq.com", 465)  # 根据邮箱服务提供商填写
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
- **趋势熔断**：{'🚨 是' if signal['small_pct_1'] <= -0.045 or signal['small_pct_2'] <= -0.06 else '✅ 否'} （最近 1 天跌超 4.5% 或者最近 2 天跌超 6%）
- **动量排名**：{signal['top_n']}
- **动量是否top1**：{'🚀 是' if signal['momentum_ok'] else '📉 否'} （领先：小市值组合动量排名在top1）
- **动量是否top2**：{'🚀 是' if signal['momentum_ok2'] else '📉 否'} （领先：小市值组合动量排名在top2）
- **动量是否top2[short]**：{'🚀 是' if signal['momentum_ok2_short'] else '📉 否'} （领先：short 小市值组合动量排名在top2）
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
    # if signal['current_hold']:
    #     hold_md = "\n".join([f"- {stock}" for stock in signal['current_hold']])
    # else:
    #     hold_md = "无"

    action_md = f"""\
**📥 建议买入：**\n
{buy_md}
**💼 附加信息：**\n
- **回测逻辑**：
    调仓/开仓逻辑：(每周三或者 持仓为 0) AND 小市值动量排名第一。 
    每日全局止损逻辑： 短期动量跌出 top2  AND （长期动量迭出 top2 OR 长期动量迭出 top1 并且股票的最小持仓天数大于 2）  （**每日止损逻辑比调仓逻辑要松一点**）全部清仓。
    个股每天止损：跌破 6% 卖出（止损时检测下，避免被刷下去）
    
    slope：{signal['slope']}
"""

    return header + "\n" + momentum_md + "\n" + action_md


if __name__ == '__main__':
    result = receive_latest_email(sender_filter="837602401@qq.com")
    print( result)