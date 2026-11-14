"""
会话身份（对应问题 P0-3：用户身份为随机数且进程级共享）。

原实现 get_user_id 从 1001~1010 中随机取值并存入模块级全局变量：
同一进程内所有访客共用一个随机身份，可能把另一位用户的面积、耗材、
故障记录当成本人数据返回；模型还可以随意传别人的 user_id 取数。

现在改为：
1. 身份由会话 ID 稳定解析（同一会话恒定，不同会话不同）；
2. 通过 ContextVar 在一次对话内传递，不污染进程级状态；
3. 工具从上下文取身份，模型无法指定他人 user_id。

生产环境应把 resolve_session_user_id 替换为登录态解析（SSO / JWT sub），
并在取数前做权限校验。
"""
import hashlib
from contextlib import contextmanager
from contextvars import ContextVar
from dataclasses import dataclass

# 与 data/external/records.csv 中的模拟用户保持一致
DEMO_USER_IDS = tuple(str(1001 + i) for i in range(10))


class IdentityNotSetError(RuntimeError):
    """当前上下文未绑定会话身份时访问用户数据，直接拒绝。"""


@dataclass(frozen=True)
class SessionIdentity:
    session_id: str
    user_id: str
    is_demo: bool = False


_current_identity: ContextVar["SessionIdentity | None"] = ContextVar(
    "current_identity", default=None
)


def resolve_session_user_id(session_id: str) -> str:
    """
    演示环境的身份解析：按会话 ID 稳定映射到一个模拟用户。
    注意这不是权限机制本身——它只保证「同一会话始终是同一个人，
    不同会话不会共用同一个人」，避免随机共享造成的越权。
    """
    seed = (session_id or "anonymous").encode("utf-8")
    index = int(hashlib.sha256(seed).hexdigest()[:8], 16) % len(DEMO_USER_IDS)
    return DEMO_USER_IDS[index]


def build_identity(session_id: str) -> SessionIdentity:
    return SessionIdentity(
        session_id=session_id or "anonymous",
        user_id=resolve_session_user_id(session_id),
        is_demo=True,
    )


def current_identity() -> "SessionIdentity | None":
    return _current_identity.get()


def current_user_id() -> str:
    identity = _current_identity.get()
    if identity is None:
        raise IdentityNotSetError("当前上下文未绑定会话身份，拒绝访问用户数据")
    return identity.user_id


@contextmanager
def session_scope(session_id: str):
    """在一次对话内绑定身份，退出时恢复原值，避免跨请求泄漏。"""
    identity = build_identity(session_id)
    token = _current_identity.set(identity)
    try:
        yield identity
    finally:
        _current_identity.reset(token)
