"""
隐私脱敏（对应问题 P2-5：DEBUG 日志落盘包含用户输入原文等个人信息）。

只做「输出侧脱敏」，不改变业务数据：
- 手机号、身份证号、邮箱、长数字串一律打码
- 用于日志与埋点，不用于展示给用户
"""
import re

_PATTERNS = (
    # 用「前后不是数字」而非 \b：中文紧邻数字时 \b 不成立，会导致漏脱敏
    (re.compile(r"(?<!\d)1[3-9]\d{9}(?!\d)"), lambda m: m.group(0)[:3] + "****" + m.group(0)[-4:]),
    (re.compile(r"(?<![\dXx])\d{17}[\dXx](?![\dXx])"), lambda m: m.group(0)[:6] + "********" + m.group(0)[-4:]),
    (re.compile(r"[\w.+-]+@[\w-]+\.[\w.]+"), lambda m: m.group(0)[:2] + "***@" + m.group(0).split("@")[-1]),
    (re.compile(r"(?<!\d)\d{12,}(?!\d)"), lambda m: m.group(0)[:4] + "****"),
    (re.compile(r"(?i)(sk-[A-Za-z0-9]{6})[A-Za-z0-9]+"), lambda m: m.group(1) + "***"),
)

MAX_LOG_TEXT = 60


def mask_text(text: str) -> str:
    """对自由文本做脱敏，返回可直接写日志的字符串。"""
    if not text:
        return ""
    masked = str(text)
    for pattern, repl in _PATTERNS:
        masked = pattern.sub(repl, masked)
    return masked


def brief(text: str, limit: int = MAX_LOG_TEXT) -> str:
    """脱敏 + 截断，用于日志摘要。"""
    masked = mask_text(text).replace("\n", " ").strip()
    if len(masked) <= limit:
        return masked
    return masked[:limit] + "…"
