

def convert_code(code: str) -> str:
    """将 'sh.000005' 或 'sz.002415' 转换为 '000005.SH' 或 '002415.SZ'"""
    parts = code.split('.')
    if len(parts) == 2:
        exchange = parts[0].upper()
        symbol = parts[1]
        return f"{symbol}.{exchange}"
    else:
        return code  # 如果格式不对，原样返回
