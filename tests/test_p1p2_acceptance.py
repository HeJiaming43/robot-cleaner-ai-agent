"""
P1 / P2 优化验收用例。

同样全部离线可运行：不调用模型、不访问向量库网络接口。
运行：python tests/test_p1p2_acceptance.py
"""
import json
import os
import sys
import tempfile
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from langchain_core.documents import Document

from agent.escalation import assess
from agent.react_agent import _iter_with_stall_timeout
from rag import ranking
from rag.vector_store import VectorStoreService
from utils import privacy, telemetry, warmup


def _doc(text, source="故障排除.txt"):
    return Document(page_content=text, metadata={"source": source})


# ------------------------------------------------------------ P2-5 脱敏与埋点
def test_mask_text_hides_pii():
    masked = privacy.mask_text("手机13812345678 邮箱zhang.san@example.com 密钥sk-abcdefghij123456")
    assert "13812345678" not in masked
    assert "138****5678" in masked
    assert "zhang.san@example.com" not in masked
    assert "sk-abcdef***" in masked


def test_brief_truncates_and_strips_newlines():
    text = "第一行\n" + "内容" * 100
    result = privacy.brief(text, limit=20)
    assert "\n" not in result
    assert result.endswith("…")
    assert len(result) <= 21


def test_percentile_matches_expected():
    values = [float(i) for i in range(1, 101)]
    assert telemetry.percentile(values, 0.5) == 50.5
    # 1..100 的线性插值 p95 ≈ 95.95，用区间断言避免实现细节耦合
    assert 95.0 <= telemetry.percentile(values, 0.95) <= 96.5
    assert telemetry.percentile([], 0.5) == 0.0
    assert telemetry.percentile([7.0], 0.95) == 7.0


def test_record_appends_jsonl():
    with tempfile.TemporaryDirectory() as tmp:
        original = telemetry._events_dir
        telemetry._events_dir = tmp
        try:
            telemetry.record("unit_test", request_id="r1", query="手机13812345678", hits=3)
            files = [f for f in os.listdir(tmp) if f.endswith(".jsonl")]
            assert files, "应生成事件文件"
            with open(os.path.join(tmp, files[0]), encoding="utf-8") as f:
                payload = json.loads(f.readline())
            assert payload["event"] == "unit_test"
            assert payload["hits"] == 3
            assert "13812345678" not in payload["query"], "埋点中的 PII 必须脱敏"
        finally:
            telemetry._events_dir = original


# ------------------------------------------------------------ P2-1 检索增强
def test_extract_key_terms_finds_codes():
    assert ranking.extract_key_terms("机器人报E13错误怎么办") == ["E13"]
    assert ranking.extract_key_terms("X20 和 s7 哪个好") == ["X20", "S7"]
    assert ranking.extract_key_terms("日常怎么保养") == []
    assert ranking.extract_key_terms("") == []


def test_lexical_overlap_bounds():
    assert ranking.lexical_overlap("滤网清理", "滤网清理步骤") == 1.0
    assert ranking.lexical_overlap("滤网清理", "充电座保养") == 0.0
    assert ranking.lexical_overlap("", "任意文本") == 0.0


def test_rerank_promotes_lexically_relevant_doc():
    irrelevant = _doc("充电座指示灯说明")
    relevant = _doc("滤网清理与更换周期")
    pairs = [(irrelevant, 0.51), (relevant, 0.50)]
    ordered = ranking.rerank("滤网清理", pairs, alpha=0.5)
    assert ordered[0][0].page_content == "滤网清理与更换周期"


def test_merge_keyword_hits_dedup_and_boost():
    base = [(_doc("滤网清理"), 0.4)]
    hits = [_doc("滤网清理"), _doc("E13 表示主刷缠绕")]
    merged = ranking.merge_keyword_hits(base, hits, k=5)
    assert len(merged) == 2, "重复内容不应重复占位"
    assert merged[-1][0].page_content.startswith("E13")
    assert merged[-1][1] > merged[0][1], "关键词命中应被加权"


def test_merge_keyword_hits_respects_k():
    base = [(_doc(f"文档{i}"), 0.4) for i in range(3)]
    merged = ranking.merge_keyword_hits(base, [_doc("新命中")], k=3)
    assert len(merged) == 3


def test_looks_like_scanned():
    assert ranking.looks_like_scanned([_doc(""), _doc("   ")]) is True
    assert ranking.looks_like_scanned([_doc("正常文本" * 20)]) is False
    assert ranking.looks_like_scanned([]) is False


# ------------------------------------------------------- P1-1 蓝绿重建判定
def test_collection_alternates_between_slots():
    svc = VectorStoreService.__new__(VectorStoreService)
    svc.collection_name = "agent"
    assert svc._next_collection_name() == "agent_v2"
    svc.collection_name = "agent_v2"
    assert svc._next_collection_name() == "agent"


def test_pointer_resolution_validates_content():
    with tempfile.TemporaryDirectory() as tmp:
        pointer = os.path.join(tmp, "ACTIVE_COLLECTION")
        svc = VectorStoreService.__new__(VectorStoreService)
        svc._pointer_path = lambda: pointer

        assert svc._resolve_active_collection() == "agent", "指针缺失应回退默认集合"

        with open(pointer, "w", encoding="utf-8") as f:
            f.write("agent_v2")
        assert svc._resolve_active_collection() == "agent_v2"

        with open(pointer, "w", encoding="utf-8") as f:
            f.write("../../etc/passwd")
        assert svc._resolve_active_collection() == "agent", "非法指针必须被拒绝"

        with open(pointer, "w", encoding="utf-8") as f:
            f.write("other_collection")
        assert svc._resolve_active_collection() == "agent", "非本库前缀的集合名应拒绝"


def test_pointer_write_is_atomic():
    with tempfile.TemporaryDirectory() as tmp:
        pointer = os.path.join(tmp, "ACTIVE_COLLECTION")
        svc = VectorStoreService.__new__(VectorStoreService)
        svc._pointer_path = lambda: pointer

        svc._write_active_pointer("agent_v2")
        assert open(pointer, encoding="utf-8").read() == "agent_v2"
        assert not os.path.exists(pointer + ".tmp"), "临时文件应被清理"


def test_rag_service_does_not_cache_retriever():
    import inspect
    from rag.rag_service import RagSummarizeService

    assert isinstance(inspect.getattr_static(RagSummarizeService, "retriever"), property), \
        "retriever 必须是动态属性，否则蓝绿切换后仍指向旧集合"


# -------------------------------------------------- 嵌入模型指纹（换模型必查）
def _manifest_stub(tmp: str, count: int = 10):
    from rag import vector_store as vsmod

    svc = vsmod.VectorStoreService.__new__(vsmod.VectorStoreService)
    svc.collection_name = "agent"
    svc._manifest_path = lambda: os.path.join(tmp, "index_manifest.json")
    svc.vector_store = type("S", (), {
        "_collection": type("C", (), {"count": lambda self: count})()
    })()
    return svc


def test_manifest_records_and_detects_model_change():
    from rag import vector_store as vsmod

    with tempfile.TemporaryDirectory() as tmp:
        svc = _manifest_stub(tmp)
        original = vsmod.rag_conf["embedding_model_name"]
        try:
            vsmod.rag_conf["embedding_model_name"] = "model-a"
            svc._write_manifest_entry("agent", files=6, chunks=123)
            assert svc.recorded_embedding_model() == "model-a"
            assert svc.embedding_model_mismatch() is None

            vsmod.rag_conf["embedding_model_name"] = "model-b"
            assert svc.embedding_model_mismatch() == ("model-a", "model-b"), \
                "换模型后必须判定为不一致，否则检索会静默劣化"
        finally:
            vsmod.rag_conf["embedding_model_name"] = original


def test_manifest_flags_legacy_index_without_fingerprint():
    with tempfile.TemporaryDirectory() as tmp:
        svc = _manifest_stub(tmp, count=328)
        assert svc.recorded_embedding_model() == ""
        assert svc.embedding_model_mismatch()[0] == "unknown", "有数据但无指纹应提示重建"


def test_manifest_ignores_empty_index_without_fingerprint():
    with tempfile.TemporaryDirectory() as tmp:
        svc = _manifest_stub(tmp, count=0)
        assert svc.embedding_model_mismatch() is None, "空索引无需提示重建"


def test_incremental_import_blocked_when_model_changed():
    from rag import vector_store as vsmod

    svc = vsmod.VectorStoreService.__new__(vsmod.VectorStoreService)
    svc.vector_store = object()
    svc.last_failures = []
    svc.last_warnings = []
    svc.embedding_model_mismatch = lambda: ("old-model", "new-model")

    try:
        svc.load_document()
        raise AssertionError("模型不一致时增量导入必须被拒绝")
    except RuntimeError as e:
        assert "全量重建" in str(e)


def test_manifest_entry_removed_with_collection():
    with tempfile.TemporaryDirectory() as tmp:
        svc = _manifest_stub(tmp)
        svc._write_manifest_entry("agent_v2", files=6, chunks=123)
        svc._drop_manifest_entry("agent_v2")
        assert svc.recorded_embedding_model("agent_v2") == ""


# ------------------------------------------------------------ P1-3 预热幂等
def test_start_preload_is_idempotent():
    original_run, original_started = warmup._run, warmup._started
    calls = []
    try:
        warmup._started = False
        warmup._run = lambda: calls.append(1)
        assert warmup.start_preload() is True
        assert warmup.start_preload() is False, "重复调用不应重复启动线程"
        time.sleep(0.05)
        assert calls == [1]
    finally:
        warmup._run = original_run
        warmup._started = original_started


# ------------------------------------------------------------ P1-5 静默超时
def test_stall_timeout_disabled_passes_through():
    assert list(_iter_with_stall_timeout(iter([1, 2, 3]), 0)) == [1, 2, 3]


def test_stall_timeout_raises_on_silence():
    def slow():
        time.sleep(0.5)
        yield "too late"

    try:
        list(_iter_with_stall_timeout(slow(), 0.1))
        raise AssertionError("静默超时应抛出 TimeoutError")
    except TimeoutError as e:
        assert "无输出" in str(e)


def test_stall_timeout_passes_fast_stream():
    assert list(_iter_with_stall_timeout(iter(["a", "b"]), 5)) == ["a", "b"]


def test_stall_timeout_propagates_contextvars():
    """看门狗把执行搬到工作线程后，工具仍必须能读到会话身份（ContextVar）。"""
    from agent.identity import current_user_id, session_scope

    def identity_reader():
        yield current_user_id()

    with session_scope("ctx-test"):
        expected = current_user_id()
        result = list(_iter_with_stall_timeout(identity_reader(), 5))

    assert result == [expected], "工作线程丢失 contextvars 会导致报告链路报「缺少会话身份」"


# ------------------------------------------------------------ P2-3 兜底与工单
def test_escalation_normal_answer_not_triggered():
    decision = assess("滤网建议每两周清理一次。[参考资料1]", "滤网多久清理")
    assert decision.should_escalate is False
    assert decision.reason == ""


def test_escalation_on_insufficient_knowledge():
    decision = assess("知识库中暂未查到该信息", "我们家的机器能洗地吗")
    assert decision.should_escalate is True
    assert "知识库" in decision.reason


def test_escalation_on_user_request():
    decision = assess("这个问题……", "我要转人工")
    assert decision.should_escalate is True
    assert decision.reason == "用户主动要求人工"


def test_escalation_on_error_and_empty():
    assert assess("抱歉", "问题", had_error=True).should_escalate is True
    assert assess("   ", "问题").should_escalate is True


def test_ticket_written_with_masked_content():
    from utils.tickets import create_ticket

    ticket = create_ticket("sess-1", "手机13812345678 报E13", "未查到", "知识库未覆盖该问题")
    assert ticket.startswith("T")
    path = os.path.join("data", "tickets", f"tickets_{time.strftime('%Y%m%d')}.jsonl")
    assert os.path.exists(path), "工单文件应被创建"
    with open(path, encoding="utf-8") as f:
        last = json.loads(f.readlines()[-1])
    assert last["ticket_id"] == ticket
    assert "13812345678" not in last["question"], "工单中的手机号必须脱敏"


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
