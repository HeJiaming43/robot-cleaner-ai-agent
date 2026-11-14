"""
统一对外错误文案（对应问题 P0-6：异常原文回显用户）。

原则：用户看到的是「可操作的信息 + 错误码」，技术细节只进日志。
错误码用于支持侧关联日志，不含任何内部实现线索。
"""
import uuid

ERROR_MESSAGE_TEMPLATE = (
    "抱歉，本轮对话未能完成（错误码 {code}）。"
    "请稍后重试；若问题持续，可提供该错误码联系人工客服。"
)

EMPTY_ANSWER_MESSAGE = (
    "抱歉，我暂时无法回答这个问题。"
    "您可以补充具体机型、故障现象或使用场景，也可以联系人工客服进一步处理。"
)


def new_error_code() -> str:
    """生成短错误码，仅供支持侧关联日志。"""
    return uuid.uuid4().hex[:8].upper()


def friendly_error_message(code: str) -> str:
    return ERROR_MESSAGE_TEMPLATE.format(code=code)
