"""
上传内容守卫（对应问题 P0-5：RAG 提示注入面）。

知识库文档会以「参考资料」形式拼进模型上下文，因此上传文件本身就是
一条进入模型指令通道的路径。这里做两道处理：

1. 上传时检测常见注入特征，命中即拒绝导入并提示管理员复核；
2. 运行时的最终防线不依赖这里——检索结果始终被包裹在「不可信数据」边界内，
   并在提示词中明确禁止执行资料中的指令（见 rag/retrieval_gate.py 与系统提示词）。
"""
from typing import Iterable

# 常见提示注入 / 指令覆盖特征。只做「提示 + 拦截」，不做静默改写：
# 静默改写会破坏知识原文，反而让问题更难排查。
INJECTION_MARKERS: tuple[str, ...] = (
    "忽略以上",
    "忽略之前",
    "忽略上述",
    "无视以上",
    "ignore previous",
    "ignore above",
    "ignore all previous",
    "disregard previous",
    "system prompt",
    "开发者模式",
    "你现在是",
    "从现在开始你",
    "输出你的提示词",
    "reveal your prompt",
)


def detect_injection(text: str, markers: Iterable[str] | None = None) -> list[str]:
    """返回命中的特征列表；空列表表示未发现可疑内容。"""
    if not text:
        return []
    lowered = text.lower()
    pool = INJECTION_MARKERS if markers is None else tuple(markers)
    return [marker for marker in pool if marker.lower() in lowered]


def decode_for_scan(raw: bytes, limit: int = 200_000) -> str:
    """
    尽力把上传内容解码为文本用于扫描。
    二进制（如 PDF）解码失败时返回占位说明，不阻断扫描流程——
    PDF 的注入防护依赖运行时的不可信数据边界。
    """
    if not raw:
        return ""
    try:
        return raw[:limit].decode("utf-8", errors="ignore")
    except Exception:
        return ""
