"""
结构化埋点（对应问题 P2-5：无 request_id、无延迟与命中指标、无法归因）。

设计取舍：
- 落 JSONL 而非引入 Prometheus / OpenTelemetry，保持零新增依赖；
  每行一个事件，可直接用 jq / pandas / Excel 读取，便于运营侧做看板。
- 事件只记录「可聚合的字段」，原文经过脱敏与截断，避免日志成为 PII 载体。
- 写失败绝不影响主流程（埋点不能成为新的故障点）。
"""
import json
import os
import threading
import uuid
from datetime import datetime

from utils.path_tool import get_abs_path
from utils.privacy import brief

_events_dir = get_abs_path("logs")
_write_lock = threading.Lock()


def new_request_id() -> str:
    """一次「用户提问 → 回答」的链路 ID，贯穿检索、生成与反馈。"""
    return uuid.uuid4().hex[:12]


def _events_path() -> str:
    os.makedirs(_events_dir, exist_ok=True)
    return os.path.join(_events_dir, f"events_{datetime.now().strftime('%Y%m%d')}.jsonl")


def record(event: str, **fields) -> None:
    """
    追加一条事件。字段为 None 的会被丢弃，字符串字段做脱敏截断。
    任何异常都在此吞掉并静默——埋点不可用不应拖垮对话。
    """
    payload = {
        "ts": datetime.now().isoformat(timespec="seconds"),
        "event": event,
    }
    for key, value in fields.items():
        if value is None:
            continue
        payload[key] = brief(value) if isinstance(value, str) else value

    try:
        line = json.dumps(payload, ensure_ascii=False)
        with _write_lock:
            with open(_events_path(), "a", encoding="utf-8") as f:
                f.write(line + "\n")
    except Exception:
        pass


def percentile(values: list[float], ratio: float) -> float:
    """朴素分位数实现（线性插值），用于 p50/p95 这类延迟统计。"""
    if not values:
        return 0.0
    ordered = sorted(values)
    if len(ordered) == 1:
        return float(ordered[0])
    position = (len(ordered) - 1) * ratio
    lower = int(position)
    upper = min(lower + 1, len(ordered) - 1)
    weight = position - lower
    return float(ordered[lower] * (1 - weight) + ordered[upper] * weight)
