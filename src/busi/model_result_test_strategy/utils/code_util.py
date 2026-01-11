import re

def normalize_code(code):
    """
    SN000001 -> sn.000001
    SH600519 -> sh.600519
    sn.000001 -> sn.000001（已规范的不再动）
    """
    if code is None:
        return code

    code = str(code).strip()

    # 已经是 sn.000001 这种格式
    if "." in code:
        return code.lower()

    m = re.match(r"([A-Za-z]+)(\d+)", code)
    if not m:
        # 非标准格式，兜底处理
        return code.lower()

    prefix, num = m.groups()
    return f"{prefix.lower()}.{num}"