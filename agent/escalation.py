"""
分层兜底与转人工判定（对应问题 P2-3：无人工兜底与失效出口）。

原则：判定逻辑做成纯函数，不依赖 Streamlit 与模型，便于离线回归；
触发条件取「可观测的确定性信号」，不引入额外的分类模型调用：
1. 用户主动要求人工 —— 最高优先级，任何情况都应转；
2. 本轮出现系统异常；
3. 回答为空；
4. 回答中出现「未查到 / 无法查询」类表述 —— 说明知识库未覆盖。
"""
from dataclasses import dataclass

HUMAN_KEYWORDS = ("转人工", "人工客服", "找人工", "真人", "投诉", "联系客服")

INSUFFICIENT_MARKERS = (
    "暂未查到",
    "未查到",
    "没有查到",
    "无法查询",
    "没有找到相关",
    "知识库中暂未",
    "我不知道",
)

ESCALATION_HINT = (
    "这个问题我暂时无法给出可靠答复。"
    "可以点击下方按钮转接人工客服，我们会记录你的问题并跟进。"
)


@dataclass(frozen=True)
class EscalationDecision:
    should_escalate: bool
    reason: str = ""
    message: str = ""


def assess(
    answer: str,
    user_query: str = "",
    had_error: bool = False,
) -> EscalationDecision:
    """返回是否需要转人工及原因。reason 用于工单与埋点，message 用于界面提示。"""
    query = user_query or ""
    if any(keyword in query for keyword in HUMAN_KEYWORDS):
        return EscalationDecision(True, "用户主动要求人工", ESCALATION_HINT)

    if had_error:
        return EscalationDecision(True, "本轮回答出现系统异常", ESCALATION_HINT)

    text = (answer or "").strip()
    if not text:
        return EscalationDecision(True, "回答为空", ESCALATION_HINT)

    for marker in INSUFFICIENT_MARKERS:
        if marker in text:
            return EscalationDecision(True, "知识库未覆盖该问题", ESCALATION_HINT)

    return EscalationDecision(False)
