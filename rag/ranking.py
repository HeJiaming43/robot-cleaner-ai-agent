"""
检索候选处理（对应问题 P2-1：纯向量检索、对错误码不敏感、无重排）。

这里只放纯函数，不碰向量库与网络，便于离线回归：
1. extract_key_terms —— 抽取 E13 / X20 这类精确词，走关键词通道补召回
2. lexical_overlap   —— 查询词覆盖率，用作轻量重排信号
3. rerank            —— 向量分与词法覆盖率的加权融合
4. merge_keyword_hits—— 把关键词精确命中并入候选集并去重
5. looks_like_scanned—— 判断 PDF 是否缺少文本层（扫描件）
"""
import re

CODE_PATTERN = re.compile(r"[A-Za-z]{1,4}\d{1,4}")
_TOKEN_PATTERN = re.compile(r"[\u4e00-\u9fff]|[A-Za-z]+|\d+")

CODE_BOOST = 0.05     # 关键词精确命中的基础加权
SCANNED_CHAR_THRESHOLD = 20   # 每页平均字符数低于该值，视为疑似扫描件


def extract_key_terms(query: str, limit: int = 3) -> list[str]:
    """抽取查询中的错误码 / 型号（如 E13、S7、X20）。无命中返回空列表。"""
    if not query:
        return []
    terms: list[str] = []
    for match in CODE_PATTERN.findall(query):
        term = match.upper()
        if term not in terms:
            terms.append(term)
    return terms[:limit]


def _tokens(text: str) -> set[str]:
    return set(_TOKEN_PATTERN.findall((text or "").lower()))


def lexical_overlap(query: str, text: str) -> float:
    """查询词在文本中的覆盖率，取值 0~1。CJK 按字切分，英文按词切分。"""
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0.0
    return len(query_tokens & _tokens(text)) / len(query_tokens)


def rerank(query: str, pairs: list[tuple], alpha: float = 0.7) -> list[tuple]:
    """
    向量分与词法覆盖率加权融合，按降序返回。
    向量分先按本轮最大值归一化，避免不同相似度度量下量纲不一致。
    """
    if not pairs:
        return []

    scores = [float(score) for _, score in pairs]
    top = max(scores) if scores else 0.0
    denominator = top if top > 0 else 1.0

    ranked: list[tuple] = []
    for doc, score in pairs:
        vector_part = max(0.0, float(score)) / denominator
        lexical_part = lexical_overlap(query, getattr(doc, "page_content", "") or "")
        ranked.append((doc, float(alpha) * vector_part + (1 - float(alpha)) * lexical_part))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return ranked


def merge_keyword_hits(pairs: list[tuple], hits: list, k: int) -> list[tuple]:
    """
    把关键词通道命中的文档并入候选集。
    关键词命中给「当前最高分 + 小加权」，保证其进入上下文而不喧宾夺主；
    按正文去重，避免与向量通道重复占位。
    """
    if not hits:
        return pairs[:k]

    seen = {(getattr(doc, "page_content", "") or "").strip() for doc, _ in pairs}
    top_score = max([float(score) for _, score in pairs], default=0.0)
    merged = list(pairs)

    for doc in hits:
        content = (getattr(doc, "page_content", "") or "").strip()
        if not content or content in seen:
            continue
        seen.add(content)
        merged.append((doc, top_score + CODE_BOOST if top_score > 0 else CODE_BOOST))

    return merged[:k]


def looks_like_scanned(documents: list, min_chars_per_page: int = SCANNED_CHAR_THRESHOLD) -> bool:
    """
    判断解析结果是否「几乎没有文本」。
    扫描版 PDF 用 PyPDFLoader 读出来是空白页，原实现会静默跳过，
    这里用于在上传时给出明确提示，而不是假装导入成功。
    """
    if not documents:
        return False
    total_chars = sum(len((getattr(doc, "page_content", "") or "").strip()) for doc in documents)
    pages = max(len(documents), 1)
    return total_chars / pages < min_chars_per_page
