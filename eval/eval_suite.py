"""
评测基线（对应问题 P2-2：原评测集与知识库同源、且测的不是生产链路）。

与 eval/rag_eval.py 的区别：
1. 题目独立编写，不取自知识库中已有的问答对，避免「考题即语料」；
2. 被测对象是生产链路（检索前置 + ReAct Agent），不再是 rag_summarize 单链；
3. 增加越界拒答、引用标注、延迟分位三项原报告缺失的指标；
4. 输出 p50 / p95，而不是只给平均值；
5. 指标命名如实：回答非空率不再叫「问题解决率」。

用法：
  # 只测检索（不调用大模型，省 token，可在无模型额度时先跑通结构）
  python eval/eval_suite.py --no-llm

  # 完整评测（需要可用的 chat 模型）
  python eval/eval_suite.py --limit 12

  # 换自定义评测集
  python eval/eval_suite.py --json eval/gold_qa.json
"""
import argparse
import json
import os
import statistics
import sys
import time
from datetime import datetime

_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from utils.logger_handler import logger
from utils.path_tool import get_abs_path
from utils.telemetry import percentile

GOLD_PATH = "eval/gold_qa.json"
RESULT_DIR = "eval/results"

REFUSAL_MARKERS = ("暂未查到", "无法", "不能", "不清楚", "建议联系", "转人工", "超出")


def load_gold(path: str) -> dict:
    with open(get_abs_path(path), "r", encoding="utf-8") as f:
        return json.load(f)


def is_recall_hit(docs, key_points: list[str]) -> bool:
    text = " ".join(getattr(doc, "page_content", "") or "" for doc in docs).lower()
    return any(point.lower() in text for point in key_points)


def refused(answer: str) -> bool:
    """越界问题的正确行为是明确拒答，而不是硬编并用参考资料背书。"""
    return any(marker in (answer or "") for marker in REFUSAL_MARKERS)


def run_retrieval_only(items: list[dict]) -> None:
    from rag.retrieval_gate import get_retrieval_gate

    gate = get_retrieval_gate()
    hits = 0
    latencies = []
    rows = []

    for item in items:
        started = time.perf_counter()
        result = gate.retrieve(item["question"])
        elapsed = time.perf_counter() - started
        latencies.append(elapsed)

        hit = is_recall_hit(result.docs, item.get("key_points", []))
        hits += 1 if hit else 0
        rows.append({
            "question": item["question"],
            "hit": hit,
            "docs": len(result.docs),
            "max_score": round(result.max_score, 3),
            "retrieve_s": round(elapsed, 3),
        })
        print(f"[{'HIT ' if hit else 'MISS'}] {item['question'][:28]} | 召回 {len(result.docs)} | {elapsed:.2f}s")

    print("-" * 52)
    print(f"检索召回率（key_point 命中）: {hits / len(items):.1%}  ({hits}/{len(items)})")
    print(f"检索耗时 p50 / p95: {percentile(latencies, 0.5):.2f}s / {percentile(latencies, 0.95):.2f}s")
    save_report({"mode": "retrieval_only", "recall": hits / len(items), "rows": rows}, latencies)


def run_full(gold: dict, limit: int) -> None:
    from agent.react_agent import ReactAgent

    agent = ReactAgent()
    in_domain = gold.get("in_domain", [])[:limit] if limit else gold.get("in_domain", [])

    recall_hits = cited = 0
    retrieve_latencies, answer_latencies = [], []
    rows = []

    for item in in_domain:
        started = time.perf_counter()
        try:
            answer = "".join(agent.execute_stream(item["question"], session_id="eval-suite")).strip()
        except Exception as e:
            logger.error(f"[评测] 生成失败: {e}")
            answer = ""
        answer_latencies.append(time.perf_counter() - started)

        citation = "[参考资料" in answer
        cited += 1 if citation else 0
        rows.append({"question": item["question"], "citation": citation, "chars": len(answer)})
        print(f"[{'CITED' if citation else 'NO-CITE'}] {item['question'][:28]} | {len(answer)} 字")

    # 越界问题：检验是否会硬答
    refused_count = 0
    ood_total = len(gold.get("out_of_domain", []))
    for item in gold.get("out_of_domain", []):
        try:
            answer = "".join(agent.execute_stream(item["question"], session_id="eval-suite")).strip()
        except Exception as e:
            logger.error(f"[评测] 越界用例生成失败: {e}")
            answer = ""
        ok = refused(answer)
        refused_count += 1 if ok else 0
        print(f"[{'REFUSED' if ok else 'ANSWERED'}] {item['question'][:28]}")

    print("-" * 52)
    print(f"引用标注率: {cited / len(in_domain):.1%}  ({cited}/{len(in_domain)})")
    print(f"越界拒答率: {refused_count / ood_total:.1%}  ({refused_count}/{ood_total})" if ood_total else "越界拒答率: n/a")
    print(f"答案耗时 p50 / p95: {percentile(answer_latencies, 0.5):.2f}s / {percentile(answer_latencies, 0.95):.2f}s")

    save_report({
        "mode": "full",
        "citation_rate": cited / len(in_domain),
        "refusal_rate": refused_count / ood_total if ood_total else None,
        "rows": rows,
    }, answer_latencies)


def save_report(payload: dict, latencies: list[float]) -> None:
    os.makedirs(get_abs_path(RESULT_DIR), exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    path = get_abs_path(f"{RESULT_DIR}/eval_suite_{ts}.json")
    payload.update({
        "evaluated_at": datetime.now().isoformat(timespec="seconds"),
        "latency_p50": round(percentile(latencies, 0.5), 3),
        "latency_p95": round(percentile(latencies, 0.95), 3),
        "latency_mean": round(statistics.fmean(latencies), 3) if latencies else 0.0,
    })
    with open(path, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    print(f"明细已保存: {path}")


def main() -> None:
    parser = argparse.ArgumentParser(description="生产链路评测基线")
    parser.add_argument("--json", default=GOLD_PATH, help="评测集路径")
    parser.add_argument("--no-llm", action="store_true", help="只测检索，不调用大模型")
    parser.add_argument("--limit", type=int, default=0, help="仅评测前 N 条（0 表示全部）")
    args = parser.parse_args()

    gold = load_gold(args.json)
    in_domain = gold.get("in_domain", [])
    if not in_domain:
        logger.error("[评测] 评测集为空，退出")
        return

    print(f"评测集：{len(in_domain)} 条域内 + {len(gold.get('out_of_domain', []))} 条越界\n")

    if args.no_llm:
        run_retrieval_only(in_domain[: args.limit] if args.limit else in_domain)
    else:
        run_full(gold, args.limit)


if __name__ == "__main__":
    main()
