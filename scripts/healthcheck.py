"""
健康检查（对应问题 P2-6：容器化与健康探针缺失）。

刻意不初始化模型与向量库客户端：健康检查必须快且不依赖外网，
否则会出现「依赖抖动 → 容器被反复重启」的雪崩。
这里只校验：配置可解析、数据目录可读、向量库已完成落盘。
"""
import argparse
import os
import sqlite3
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from utils.config_handler import agent_conf, chroma_conf, prompts_conf, rag_conf
from utils.path_tool import get_abs_path


def check_configs() -> tuple[bool, str]:
    missing = [
        name for name, conf in (
            ("agent", agent_conf), ("chroma", chroma_conf),
            ("prompts", prompts_conf), ("rag", rag_conf),
        ) if not conf
    ]
    if missing:
        return False, f"配置缺失: {','.join(missing)}"
    return True, "配置可解析"


def check_knowledge_base() -> tuple[bool, str]:
    db_path = os.path.join(get_abs_path(chroma_conf["persist_directory"]), "chroma.sqlite3")
    if not os.path.exists(db_path):
        return False, f"向量库文件不存在: {db_path}"
    try:
        with sqlite3.connect(db_path) as con:
            count = con.execute("select count(*) from embeddings").fetchone()[0]
    except Exception as e:
        return False, f"向量库不可读: {e}"
    if count <= 0:
        return False, "向量库为空，请先导入知识库"
    return True, f"向量片段 {count} 条"


def check_data_dir() -> tuple[bool, str]:
    data_dir = get_abs_path(chroma_conf["data_path"])
    if not os.path.isdir(data_dir):
        return False, f"数据目录不存在: {data_dir}"
    return True, "数据目录可读"


def check_api_key() -> tuple[bool, str]:
    if not os.environ.get("DASHSCOPE_API_KEY"):
        return False, "DASHSCOPE_API_KEY 未设置"
    return True, "API Key 已配置"


def main() -> int:
    parser = argparse.ArgumentParser(description="容器健康检查")
    parser.add_argument("--require-key", action="store_true", help="额外校验 API Key")
    parser.add_argument(
        "--allow-empty-kb",
        action="store_true",
        help="允许缺失向量库（CI 环境无 chroma_db，容器运行时不应用此参数）",
    )
    args = parser.parse_args()

    checks = [("config", check_configs), ("data", check_data_dir)]
    if args.allow_empty_kb:
        print("[SKIP] chroma: 已指定 --allow-empty-kb，跳过向量库校验")
    else:
        checks.append(("chroma", check_knowledge_base))
    if args.require_key:
        checks.append(("api_key", check_api_key))

    failed = 0
    for name, fn in checks:
        ok, detail = fn()
        print(f"[{'OK  ' if ok else 'FAIL'}] {name}: {detail}")
        failed += 0 if ok else 1

    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
