"""
知识库管理鉴权（对应问题 P0-5：知识库写入无鉴权）。

设计取舍：
- 未配置 KB_ADMIN_PASSWORD 时，写入能力默认关闭（只读），而不是「默认放行」；
  安全默认值优先，避免部署时忘记配置就直接对外开放写入口。
- 校验使用 hmac.compare_digest，避免口令比较的时序侧信道。
"""
import hmac
import os

ENV_KEY = "KB_ADMIN_PASSWORD"


def configured_password() -> str:
    """读取环境变量中的管理口令；未配置返回空串。"""
    return (os.environ.get(ENV_KEY) or "").strip()


def admin_configured() -> bool:
    """是否已配置管理口令。未配置时知识库写入应整体禁用。"""
    return bool(configured_password())


def verify_admin_password(submitted: str, expected: str | None = None) -> bool:
    """
    校验管理口令。
    expected 显式传入时用于测试或外部注入；否则读取环境变量。
    未配置口令一律返回 False——即「没有口令就不可能通过」。
    """
    target = configured_password() if expected is None else (expected or "").strip()
    if not target:
        return False
    return hmac.compare_digest((submitted or "").strip(), target)
