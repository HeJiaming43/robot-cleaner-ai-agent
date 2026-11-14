"""
RAG 知识库评测脚本（历史版本，建议改用 eval/eval_suite.py）

【本脚本的已知局限，勿直接作为准确率结论】
1. 评测集与知识库同源：题目来自 data/ 下的 100问 文件，答案几乎必然在库中，
   指标反映的是「能否取回已知答案」，不是泛化能力；
2. 被测链路与生产不一致：这里测 rag_summarize 单链，生产走「检索前置 + ReAct Agent」；
3. LLM 判分与被测模型同家族，存在自我偏好；
4. 抽样量小、单次运行、无置信区间，不适合做回归门禁。
新的评测基线见 eval/eval_suite.py（独立题目、生产链路、含越界拒答与 p95 延迟）。

指标：
  1. 检索命中率（Top-k 召回）：标准答案是否出现在 Top-k 检索片段中
  2. 回答准确率：LLM-as-judge 判定回答与标准答案要点一致的比例（--judge）
  3. 响应时间：检索耗时 / 生成耗时 / 总耗时
  4. 回答非空率：回答非空且未报错的比例（近似指标，不等于问题解决率）

用法：
  # 从单个 100问 文件解析 QA 对，采样 20 条，完整评测（含 LLM 判分）
  python eval/rag_eval.py --source data/扫地机器人100问2.txt --sample 20 --judge

  # 从全部 100问 文件解析，采样 30 条
  python eval/rag_eval.py --all --sample 30 --judge

  # 只测检索命中率（不调生成模型，省 token、速度快）
  python eval/rag_eval.py --source data/扫地机器人100问2.txt --sample 20 --no-answer

  # 使用自定义 JSON 评测集 [{"question": "...", "reference_answer": "..."}]
  python eval/rag_eval.py --json eval/qa_custom.json --sample 20 --judge

说明：
  - 100问 类文件为 Markdown 格式 "1. **问题？**\n- 答案"，可直接解析为评测集
  - 首次运行前请确保已执行 streamlit 导入知识库（chroma_db 存在）
  - 需要 DASHSCOPE_API_KEY 环境变量（或等价配置）
"""
import argparse
import json
import os
import random
import re
import sys
import time
from datetime import datetime

# 确保项目根目录在 sys.path 中，支持 `python eval/rag_eval.py` 直接运行
_PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if _PROJECT_ROOT not in sys.path:
    sys.path.insert(0, _PROJECT_ROOT)

from model.factory import chat_model
from rag.rag_service import RagSummarizeService
from utils.logger_handler import logger
from utils.path_tool import get_abs_path


# 100问 类文件：Markdown 格式 "1. **问题？**\n- 答案"
QA_FILES = [
    "data/扫地机器人100问2.txt",
    "data/扫拖一体机器人100问.txt",
]

RESULT_DIR = "eval/results"


def parse_qa_from_100wen(filepath: str) -> list[dict]:
    """解析 "1. **问题？**\n- 答案" 格式的问答文件。"""
    qa_pairs = []
    current_q = None
    try:
        with open(get_abs_path(filepath), "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                m = re.match(r"\d+\.\s*\*\*(.+?)\*\*", line)
                if m:
                    current_q = m.group(1).strip()
                elif line.startswith("-") and current_q:
                    answer = line.lstrip("- ").strip()
                    if answer:
                        qa_pairs.append({
                            "question": current_q,
                            "reference_answer": answer,
                            "source": os.path.basename(filepath),
                        })
                    current_q = None
    except FileNotFoundError:
        logger.error(f"[评测] 文件不存在: {filepath}")
    return qa_pairs


def load_qa_set(args) -> list[dict]:
    """加载评测集：自定义 JSON > 指定文件 > 全部 100问 文件。"""
    if args.json:
        with open(get_abs_path(args.json), "r", encoding="utf-8") as f:
            return json.load(f)

    sources = [args.source] if args.source else QA_FILES
    qa_pairs = []
    for src in sources:
        qa_pairs.extend(parse_qa_from_100wen(src))

    if not qa_pairs:
        logger.warning("[评测] 未解析到任何 QA 对，请检查文件格式")
    return qa_pairs


def normalize(text: str) -> str:
    """去除标点与空白，仅保留汉字/字母/数字，便于子串匹配。"""
    return re.sub(r"[^\w\u4e00-\u9fff]", "", text or "")


def is_hit(reference_answer: str, retrieved_docs) -> bool:
    """标准答案的前 15 个有效字符是否出现在任一检索片段中。"""
    key = normalize(reference_answer)[:15]
    if not key:
        return False
    return any(key in normalize(doc.page_content) for doc in retrieved_docs)


def judge_answer(question: str, reference_answer: str, model_answer: str) -> bool | None:
    """LLM-as-judge：判断模型回答是否覆盖标准答案核心要点。"""
    prompt = f"""你是检索增强问答系统的评测员，请判断模型回答是否正确回答了用户问题。

用户问题：{question}
标准答案：{reference_answer}
模型回答：{model_answer}

判定标准：
- 模型回答覆盖了标准答案的核心要点，且无关键错误信息 → 输出 1
- 模型回答缺失核心要点、含错误信息或答非所问 → 输出 0
- 只输出 1 或 0，不要输出其他内容。"""
    try:
        resp = chat_model.invoke(prompt)
        text = resp.content.strip() if hasattr(resp, "content") else str(resp).strip()
        return text.startswith("1")
    except Exception as e:
        logger.error(f"[评测] LLM 判分失败: {e}")
        return None


def main():
    parser = argparse.ArgumentParser(description="RAG 知识库评测")
    parser.add_argument("--source", type=str, default=None, help="单个 100问 文件路径")
    parser.add_argument("--all", action="store_true", help="从全部 100问 文件解析")
    parser.add_argument("--json", type=str, default=None, help="自定义 JSON 评测集路径")
    parser.add_argument("--sample", type=int, default=20, help="采样条数（0 表示全部）")
    parser.add_argument("--k", type=int, default=3, help="检索 Top-k")
    parser.add_argument("--no-answer", action="store_true", help="只测检索命中，不跑完整问答")
    parser.add_argument("--judge", action="store_true", help="用 LLM 判分回答准确率")
    parser.add_argument("--export", type=str, default="eval/qa_set_export.json", help="导出评测集路径")
    args = parser.parse_args()

    qa_pairs = load_qa_set(args)
    total = len(qa_pairs)
    if total == 0:
        logger.error("[评测] 评测集为空，退出")
        return

    # 采样
    if args.sample and args.sample < total:
        qa_pairs = random.sample(qa_pairs, args.sample)

    # 导出评测集，便于审阅/复用
    os.makedirs(os.path.dirname(get_abs_path(args.export)), exist_ok=True)
    with open(get_abs_path(args.export), "w", encoding="utf-8") as f:
        json.dump(qa_pairs, f, ensure_ascii=False, indent=2)
    logger.info(f"[评测] 评测集已导出: {args.export}（{len(qa_pairs)} 条）")

    # 初始化服务（加载向量库 + 嵌入模型）
    logger.info("[评测] 初始化 RAG 服务（首次加载可能较慢）…")
    service = RagSummarizeService()

    hit_count = 0
    judge_pass = 0
    judge_total = 0
    solved = 0
    retrieve_times = []
    answer_times = []
    total_times = []
    details = []

    for i, item in enumerate(qa_pairs, 1):
        q = item["question"]
        ref = item["reference_answer"]
        source = item.get("source", "unknown")

        # 1. 检索 + 命中判断
        t0 = time.perf_counter()
        docs = service.retrieve_docs(q)
        t1 = time.perf_counter()
        retrieve_time = t1 - t0
        retrieve_times.append(retrieve_time)
        hit = is_hit(ref, docs)
        if hit:
            hit_count += 1

        # 2. 完整问答 + 计时
        model_answer = ""
        error = None
        if not args.no_answer:
            try:
                model_answer = service.rag_summarize(q).strip()
            except Exception as e:
                model_answer = ""
                error = str(e)
            answer_times.append(time.perf_counter() - t1)
            total_times.append(time.perf_counter() - t0)

        # 3. LLM 判分（可选）
        judged = None
        if args.judge and not args.no_answer and model_answer and not error:
            judged = judge_answer(q, ref, model_answer)
            if judged is not None:
                judge_total += 1
                if judged:
                    judge_pass += 1

        # 4. 近似问题解决率
        if model_answer and not error:
            solved += 1

        details.append({
            "question": q,
            "source": source,
            "hit": hit,
            "judged": judged,
            "error": error,
            "retrieve_time": round(retrieve_time, 3),
            "answer": model_answer[:200] if model_answer else "",
        })

        status = "HIT" if hit else "MISS"
        if not args.no_answer:
            status += f" | {'PASS' if judged else 'FAIL' if judged is False else 'SKIP'}"
        print(f"[{i}/{len(qa_pairs)}] {status} | {q[:30]} | 检索 {retrieve_time:.2f}s")

    # ===== 汇总 =====
    n = len(qa_pairs)
    hit_rate = hit_count / n if n else 0
    judge_rate = judge_pass / judge_total if judge_total else None
    solve_rate = solved / n if n else 0

    lines = []
    lines.append("=" * 46)
    lines.append("RAG 知识库评测报告")
    lines.append("=" * 46)
    lines.append(f"评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"评测集: {n} 条 | 检索 Top-k: {args.k} | 判分: {'LLM' if args.judge else '否'}")
    lines.append("")
    lines.append(f"① 检索命中率 (Top-{args.k} 召回): {hit_rate:.1%}  ({hit_count}/{n})")
    if judge_rate is not None:
        lines.append(f"② 回答准确率 (LLM 判分): {judge_rate:.1%}  ({judge_pass}/{judge_total})")
    if not args.no_answer:
        lines.append(f"③ 回答非空率 (近似，不等于解决率): {solve_rate:.1%}  ({solved}/{n})")
        lines.append(f"④ 平均检索耗时: {sum(retrieve_times)/len(retrieve_times):.2f}s")
        lines.append(f"   平均生成耗时: {sum(answer_times)/len(answer_times):.2f}s")
        lines.append(f"   平均总耗时: {sum(total_times)/len(total_times):.2f}s")
    lines.append("=" * 46)

    report = "\n".join(lines)
    print("\n" + report)

    # 保存报告
    os.makedirs(get_abs_path(RESULT_DIR), exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_path = get_abs_path(f"{RESULT_DIR}/rag_eval_report_{ts}.txt")
    detail_path = get_abs_path(f"{RESULT_DIR}/rag_eval_details_{ts}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write(report + "\n")
    with open(detail_path, "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)
    logger.info(f"[评测] 报告已保存: {report_path}")
    logger.info(f"[评测] 明细已保存: {detail_path}")


if __name__ == "__main__":
    main()
