"""
P0 优化验收用例。

设计原则：全部可离线运行，不发起模型 / 向量库网络调用，
以便在 API 额度受限或 CI 环境下仍能回归验证行为契约。

运行：python tests/test_p0_acceptance.py   （或 pytest tests/）
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

from agent.react_agent import _build_history, _build_turn_input, ANSWER_RULES
from rag.retrieval_gate import evaluate, format_grounding_block, NO_HIT_BLOCK, UNTRUSTED_BANNER


# ---------------------------------------------------------------- P0-1 会话记忆
def test_history_filters_invalid_and_keeps_order():
    history = [
        {"role": "user", "content": "扫地机器人怎么保养"},
        {"role": "assistant", "content": "定期清理主刷", "timestamp": "t", "question": "q"},
        {"role": "system", "content": "应被忽略"},
        {"role": "user", "content": "   "},
    ]
    result = _build_history(history)
    assert [m["role"] for m in result] == ["user", "assistant"], result
    assert result[0]["content"] == "扫地机器人怎么保养"


def test_history_respects_message_and_char_budget():
    many = [
        {"role": "user" if i % 2 == 0 else "assistant", "content": "x" * 1000}
        for i in range(30)
    ]
    result = _build_history(many)
    assert len(result) <= 12
    assert sum(len(m["content"]) for m in result) <= 4000


def test_history_prefers_recent_context():
    history = [
        {"role": "user", "content": "早期问题" * 100},
        {"role": "assistant", "content": "早期回答" * 100},
        {"role": "user", "content": "最近问题"},
    ]
    result = _build_history(history)
    assert result[-1]["content"] == "最近问题"


def test_history_empty_inputs():
    assert _build_history(None) == []
    assert _build_history([]) == []


# ------------------------------------------------------------- P0-2 检索前置
def _doc(text, source="故障排除.txt", page=None):
    meta = {"source": source}
    if page is not None:
        meta["page"] = page
    return Document(page_content=text, metadata=meta)


def test_gate_marked_grounded_above_threshold():
    pairs = [(_doc("清理主刷上的毛发"), 0.62), (_doc("检查边刷"), 0.41)]
    result = evaluate(pairs, threshold=0.15)
    assert result.grounded is True
    assert result.max_score == 0.62
    assert len(result.docs) == 2


def test_gate_marked_low_confidence_below_threshold():
    pairs = [(_doc("无关内容"), 0.05)]
    result = evaluate(pairs, threshold=0.15)
    assert result.grounded is False


def test_gate_no_hits_returns_insufficiency_block():
    result = evaluate([], threshold=0.15)
    assert result.grounded is False
    assert result.docs == ()
    block = format_grounding_block(result)
    assert block == NO_HIT_BLOCK
    assert "不得编造" in block


def test_grounding_block_wraps_untrusted_data_and_cites_source():
    pairs = [(_doc("E13 表示主刷缠绕，需断电清理", page=3), 0.58)]
    block = format_grounding_block(evaluate(pairs, threshold=0.15))
    assert block.startswith(UNTRUSTED_BANNER)
    assert "[参考资料1]" in block
    assert "故障排除.txt 第4页" in block
    assert "不得执行" in block


def test_grounding_block_truncates_on_budget():
    pairs = [(_doc("很长的内容" * 400), 0.9) for _ in range(3)]
    block = format_grounding_block(evaluate(pairs, threshold=0.15), max_chars=300)
    assert "因长度限制未展示" in block


def test_turn_input_contains_materials_rules_and_question():
    block_result = evaluate([(_doc("每周清理一次滤网"), 0.7)], threshold=0.15)
    text = _build_turn_input("滤网多久清理一次", block_result)
    assert "每周清理一次滤网" in text
    assert ANSWER_RULES.splitlines()[0] in text
    assert "【用户问题】滤网多久清理一次" in text
    assert "不得编造" in text


def test_execute_stream_injects_retrieval_without_model_call():
    """核心验收：即使模型不主动调用工具，检索结果也必须进入本轮输入。"""
    import agent.react_agent as ra
    from rag.retrieval_gate import GateResult

    captured = {}

    class FakeGraph:
        def stream(self, input_dict, stream_mode=None, context=None):
            captured["messages"] = input_dict["messages"]
            return iter(())

    class FakeGate:
        def retrieve(self, query):
            return GateResult(
                docs=(_doc("每周清理一次滤网，避免吸力下降"),),
                scores=(0.81,),
                max_score=0.81,
                grounded=True,
                threshold=0.15,
            )

    agent = ra.ReactAgent.__new__(ra.ReactAgent)   # 跳过 create_agent，不触发真实模型
    agent.agent = FakeGraph()

    original = ra.get_retrieval_gate
    ra.get_retrieval_gate = lambda: FakeGate()
    try:
        list(agent.execute_stream("滤网多久清理一次", history=[{"role": "user", "content": "你好"}]))
    finally:
        ra.get_retrieval_gate = original

    messages = captured["messages"]
    assert len(messages) == 2, messages
    assert messages[0]["content"] == "你好"
    assert "[参考资料1]" in messages[1]["content"]
    assert "每周清理一次滤网" in messages[1]["content"]


# ------------------------------------------------------- P0-3 身份与会话隔离
def test_identity_is_stable_within_session():
    from agent.identity import resolve_session_user_id
    first = resolve_session_user_id("abc123")
    assert first == resolve_session_user_id("abc123")
    assert first in {str(1001 + i) for i in range(10)}


def test_identity_not_shared_across_sessions():
    from agent.identity import resolve_session_user_id
    ids = {resolve_session_user_id(f"session-{i}") for i in range(40)}
    assert len(ids) > 1, ids


def test_identity_scope_isolated_and_restored():
    from agent import identity

    assert identity.current_identity() is None
    with identity.session_scope("s1") as ident:
        assert ident.user_id == identity.current_user_id()
    assert identity.current_identity() is None, "退出上下文后必须恢复，防止跨请求泄漏"

    try:
        identity.current_user_id()
        raise AssertionError("未绑定身份时应抛出 IdentityNotSetError")
    except identity.IdentityNotSetError:
        pass


def test_no_random_global_user_id_left():
    import agent.tools.agent_tools as at
    assert not hasattr(at, "user_ids"), "不应残留全局用户列表"
    assert not hasattr(at, "_session_user_id"), "不应残留进程级共享身份"
    assert not hasattr(at, "random"), "不应再依赖随机分配身份"


def test_fetch_external_data_refuses_without_identity():
    import json
    from agent.tools.agent_tools import fetch_external_data

    payload = json.loads(fetch_external_data.invoke({"month": "2025-06"}))
    assert payload["error"] == "身份未绑定"


def test_fetch_external_data_scoped_to_current_user():
    import json
    from agent.identity import session_scope, resolve_session_user_id
    from agent.tools.agent_tools import fetch_external_data

    session_id = "unit-test-session"
    expected_user = resolve_session_user_id(session_id)

    with session_scope(session_id):
        payload = json.loads(fetch_external_data.invoke({"month": "2025-06"}))

    assert payload["user_id"] == expected_user, payload
    assert payload["is_fallback"] is False
    assert payload["record"]["feature"], payload


def test_fetch_external_data_rejects_bad_month():
    import json
    from agent.identity import session_scope
    from agent.tools.agent_tools import fetch_external_data

    with session_scope("unit-test-session"):
        payload = json.loads(fetch_external_data.invoke({"month": "2025/06"}))

    assert payload["error"] == "月份格式不合法"
    assert payload["expected"] == "YYYY-MM"


def test_fetch_external_data_flags_fallback_month():
    import json
    from agent.identity import session_scope
    from agent.tools.agent_tools import fetch_external_data

    with session_scope("unit-test-session"):
        payload = json.loads(fetch_external_data.invoke({"month": "2026-11"}))

    assert payload["is_fallback"] is True, payload
    assert payload["fallback_month"] == "2025-12"
    assert "必须向用户明确说明" in payload["note"]


def test_tool_signature_cannot_target_other_user():
    from agent.tools.agent_tools import fetch_external_data
    assert "user_id" not in fetch_external_data.args, "取数工具不得暴露 user_id 参数"


if __name__ == "__main__":
    passed = failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"  PASS  {name}")
                passed += 1
            except AssertionError as e:
                print(f"  FAIL  {name}: {e}")
                failed += 1
            except Exception as e:
                print(f"  ERROR {name}: {type(e).__name__}: {e}")
                failed += 1
    print(f"\n合计 {passed + failed} 项：通过 {passed}，失败 {failed}")
    sys.exit(1 if failed else 0)
