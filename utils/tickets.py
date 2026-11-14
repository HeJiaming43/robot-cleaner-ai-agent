"""
工单落库（对应问题 P2-3：转人工没有承接载体）。

当前实现是把工单以 JSONL 追加到 data/tickets/ 下，作为「可被人工接续的最小闭环」：
运营侧可直接读取，也能被后续的工单系统对接替换。
生产环境应改为投递到工单服务（如企业微信/工单平台）并带回外部单号。
"""
import json
import os
import uuid
from datetime import datetime

from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.privacy import mask_text


def _tickets_dir() -> str:
    path = get_abs_path("data/tickets")
    os.makedirs(path, exist_ok=True)
    return path


def create_ticket(session_id: str, question: str, answer: str, reason: str) -> str:
    """
    创建工单并返回单号。失败时仍返回单号，避免前端因写盘异常而不可用。
    """
    ticket_id = f"T{datetime.now().strftime('%Y%m%d')}-{uuid.uuid4().hex[:6].upper()}"
    payload = {
        "ticket_id": ticket_id,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "session_id": session_id,
        "reason": reason,
        "question": mask_text(question),
        "answer_digest": mask_text((answer or "")[:200]),
        "status": "open",
    }

    try:
        path = os.path.join(_tickets_dir(), f"tickets_{datetime.now().strftime('%Y%m%d')}.jsonl")
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")
        logger.info(f"[工单] 已创建 {ticket_id}，原因：{reason}")
    except Exception as e:
        logger.error(f"[工单] 写入失败 {ticket_id}: {e}", exc_info=True)

    return ticket_id
