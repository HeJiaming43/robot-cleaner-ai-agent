"""
后台预热（对应问题 P1-3：预热状态定义在 Streamlit 脚本顶层）。

原实现的 _preloaded / _preload_done 定义在 app.py 顶层，每次 rerun
都会重新执行脚本、重建这些对象，而 session_state 又阻止线程重启，
于是新一次 rerun 拿到的 Event 永远不会被 set，get_agent 里的
wait(timeout=95) 可能空等到超时（日志中出现两次完整预热即为征兆）。

修复思路：把预热状态放进「被导入的模块」。Streamlit 只重跑主脚本，
已导入模块的模块级状态在进程内保持不变，因此这里天然是进程级单例。
"""
import threading

from utils.logger_handler import logger

_lock = threading.Lock()
_started = False
_done = threading.Event()
_state: dict[str, object] = {"vector_service": None, "agent": None}


def _load_vector_service() -> None:
    if _state["vector_service"] is not None:
        return
    logger.info("[预热] 开始加载向量库服务")
    from rag.vector_store import get_shared_vector_store
    _state["vector_service"] = get_shared_vector_store()
    logger.info("[预热] 向量库服务加载完成")


def _load_agent() -> None:
    if _state["agent"] is not None:
        return
    logger.info("[预热] 开始加载智能客服引擎")
    from agent.react_agent import ReactAgent
    _state["agent"] = ReactAgent()
    logger.info("[预热] 智能客服引擎加载完成")


def _run() -> None:
    threads = [
        threading.Thread(target=_load_vector_service, daemon=True),
        threading.Thread(target=_load_agent, daemon=True),
    ]
    try:
        for thread in threads:
            thread.start()
        # join 设上限，避免依赖异常导致预热线程永久卡住
        for thread in threads:
            thread.join(timeout=90)
    except Exception as e:
        logger.error(f"[预热] 后台初始化失败: {e}", exc_info=True)
    finally:
        _done.set()
        logger.info("[预热] 后台初始化流程结束")


def start_preload() -> bool:
    """
    启动预热。进程内只启动一次，重复调用是幂等的（返回 False）。
    这个幂等性由模块级 _started 保证，不依赖 session_state。
    """
    global _started
    with _lock:
        if _started:
            return False
        _started = True

    threading.Thread(target=_run, daemon=True).start()
    return True


def is_started() -> bool:
    return _started


def is_ready() -> bool:
    return _done.is_set()


def wait(timeout: float) -> bool:
    """等待预热完成；返回是否已完成（超时返回 False，由调用方决定是否同步初始化）。"""
    return _done.wait(timeout)


def preloaded_agent():
    return _state["agent"]


def preloaded_vector_service():
    return _state["vector_service"]
