import os
import re
import threading
from langchain_core.documents import Document
from model.factory import embed_model
from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger
from rag.ranking import extract_key_terms, looks_like_scanned, merge_keyword_hits, rerank


_ACTIVE_POINTER = "ACTIVE_COLLECTION"
_SAFE_NAME = re.compile(r"^[A-Za-z0-9_\-]{1,64}$")


class VectorStoreService:
    def __init__(self, collection_name: str | None = None):
        self._md5_lock = threading.Lock()
        self.last_failures: list[str] = []
        self.last_warnings: list[str] = []

        self.collection_name = collection_name or self._resolve_active_collection()
        self.vector_store = self._build_store(self.collection_name)

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    # ---------------------------------------------------------- 集合管理
    def _pointer_path(self) -> str:
        return os.path.join(get_abs_path(chroma_conf["persist_directory"]), _ACTIVE_POINTER)

    def _resolve_active_collection(self) -> str:
        """
        读取「当前生效集合」指针。
        蓝绿重建期间新旧集合并存，指针是唯一切换点；
        指针缺失或内容不可信时回退到配置中的默认集合名。
        """
        default_name = chroma_conf["collection_name"]
        path = self._pointer_path()
        try:
            if os.path.exists(path):
                with open(path, "r", encoding="utf-8") as f:
                    name = f.read().strip()
                if name and _SAFE_NAME.match(name) and name.startswith(default_name):
                    return name
                logger.warning(f"[知识库] 指针内容不可信，已回退默认集合: {name!r}")
        except Exception as e:
            logger.warning(f"[知识库] 读取生效指针失败，回退默认集合: {e}")
        return default_name

    def _write_active_pointer(self, name: str) -> None:
        """原子写指针：先写临时文件再 replace，避免读到半个文件名。"""
        path = self._pointer_path()
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp = path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(name)
        os.replace(tmp, path)

    def _build_store(self, name: str) -> Chroma:
        return Chroma(
            collection_name=name,
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
        )

    def _drop_collection(self, name: str) -> None:
        try:
            self._build_store(name).delete_collection()
            logger.info(f"[知识库] 已清理旧集合 {name}")
        except Exception as e:
            logger.warning(f"[知识库] 清理旧集合 {name} 失败（可忽略）: {e}")

    def _next_collection_name(self) -> str:
        """蓝绿切换的另一个槽位：agent ⇄ agent_v2。"""
        base = chroma_conf["collection_name"]
        return base if self.collection_name != base else f"{base}_v2"

    # ---------------------------------------------------------- 检索
    def get_retriever(self, k: int | None = None):
        return self.vector_store.as_retriever(search_kwargs={"k": k or chroma_conf["k"]})

    def _keyword_search(self, query: str, k: int) -> list[Document]:
        """
        关键词通道：只针对错误码 / 型号这类精确词（如 E13、X20）。
        纯向量对这类短代号不敏感，精确包含匹配能显著提升首答命中率。
        """
        terms = extract_key_terms(query)
        if not terms:
            return []

        hits: list[Document] = []
        for term in terms:
            try:
                raw = self.vector_store.get(where_document={"$contains": term}, limit=k)
            except Exception as e:
                logger.debug(f"[知识库检索] 关键词通道不可用: {e}")
                continue
            for content, metadata in zip(raw.get("documents", []), raw.get("metadatas", [])):
                hits.append(Document(page_content=content or "", metadata=metadata or {}))
        return hits

    def search_with_scores(self, query: str, k: int | None = None) -> list[tuple[Document, float]]:
        """
        带相关度返回检索结果，供「检索前置」判断资料可信度。

        流程：向量召回 → 关键词通道补召回 → 词法重排定序。
        注意：重排只改变顺序，返回分数仍是原始相关度，
        以保证阈值判定语义不被归一化破坏。
        """
        top_k = k or chroma_conf["k"]
        try:
            pairs = self.vector_store.similarity_search_with_relevance_scores(query, k=top_k)
        except Exception as e:
            logger.warning(f"[知识库检索] 带分数检索失败，降级为普通检索: {e}")
            try:
                docs = self.get_retriever(top_k).invoke(query)
            except Exception as inner:
                logger.error(f"[知识库检索] 普通检索同样失败: {inner}")
                raise
            pairs = [(doc, 0.0) for doc in docs]

        normalized: list[tuple[Document, float]] = []
        for doc, score in pairs:
            try:
                normalized.append((doc, float(score)))
            except (TypeError, ValueError):
                normalized.append((doc, 0.0))

        if chroma_conf.get("keyword_boost", True):
            hits = self._keyword_search(query, top_k)
            if hits:
                logger.info(f"[知识库检索] 关键词通道补召回 {len(hits)} 条: {extract_key_terms(query)}")
            normalized = merge_keyword_hits(normalized, hits, top_k)

        if chroma_conf.get("rerank", True) and len(normalized) > 1:
            ordering = rerank(query, normalized, float(chroma_conf.get("rerank_alpha", 0.7)))
            position = {id(doc): index for index, (doc, _) in enumerate(ordering)}
            normalized.sort(key=lambda item: position.get(id(item[0]), len(position)))

        return normalized

    # ---------------------------------------------------------- 建库
    def load_document(self, target: Chroma | None = None) -> int:
        """
        增量导入 data 目录下的文档。
        target 用于蓝绿重建：写入新集合而不影响线上正在服务的集合。
        """
        store = target or self.vector_store
        self.last_failures = []
        self.last_warnings = []

        md5_store_path = get_abs_path(chroma_conf["md5_hex_store"])

        def get_file_documents(read_path: str):
            if read_path.endswith(".txt"):
                return txt_loader(read_path)
            if read_path.endswith(".pdf"):
                return pdf_loader(read_path)
            return []

        allowed_files_path: list[str] = listdir_with_allowed_type(
            get_abs_path(chroma_conf["data_path"]),
            tuple(chroma_conf["allow_knowledge_file_type"]),
        )

        processed_count = 0
        with self._md5_lock:
            existing_md5s: set[str] = set()
            if os.path.exists(md5_store_path):
                with open(md5_store_path, "r", encoding="utf-8") as f:
                    existing_md5s = {line.strip() for line in f if line.strip()}

            for path in allowed_files_path:
                try:
                    md5_hex = get_file_md5_hex(path)
                    if md5_hex is None:
                        logger.warning(f"[加载知识库]{path} MD5计算失败，跳过")
                        self.last_failures.append(os.path.basename(path))
                        continue

                    if md5_hex in existing_md5s:
                        logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                        continue

                    documents: list[Document] = get_file_documents(path)
                    if not documents:
                        logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                        self.last_failures.append(os.path.basename(path))
                        continue

                    # 扫描版 PDF 没有文本层：原实现静默入库 0 片段，用户以为导入成功
                    if path.endswith(".pdf") and looks_like_scanned(documents):
                        message = f"{os.path.basename(path)} 疑似扫描件（缺少文本层），需 OCR 后才可检索"
                        logger.warning(f"[加载知识库]{message}")
                        self.last_warnings.append(message)
                        continue

                    split_document: list[Document] = self.spliter.split_documents(documents)
                    if not split_document:
                        logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                        self.last_failures.append(os.path.basename(path))
                        continue

                    store.add_documents(split_document)

                    existing_md5s.add(md5_hex)
                    with open(md5_store_path, "a", encoding="utf-8") as f:
                        f.write(md5_hex + "\n")

                    logger.info(f"[加载知识库]{path} 内容加载成功")
                    processed_count += 1
                except Exception as e:
                    logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                    self.last_failures.append(os.path.basename(path))
                    continue

        return processed_count

    def rebuild_knowledge_base(self) -> int:
        """
        全量重建，采用蓝绿切换：
        1. 在另一个集合槽位全量构建，期间线上查询不受影响；
        2. 构建成功才写指针并切换内存引用，最后清理旧集合；
        3. 任一步失败则删除新集合、恢复 md5 记录，旧集合继续服务。
        原实现先 delete_collection 再重建，重建窗口内的查询全部失败。
        """
        previous_name = self.collection_name
        target_name = self._next_collection_name()
        md5_path = get_abs_path(chroma_conf["md5_hex_store"])
        backup_path = md5_path + ".bak"
        had_md5 = os.path.exists(md5_path)

        if had_md5:
            os.replace(md5_path, backup_path)
        logger.info(f"[重建知识库] 目标集合 {target_name}，md5 去重记录已备份")

        staging = None
        try:
            staging = self._build_store(target_name)
            try:
                staging.delete_collection()
                staging = self._build_store(target_name)
            except Exception as e:
                logger.warning(f"[重建知识库] 清理残留集合失败（可忽略）: {e}")
                staging = self._build_store(target_name)

            count = self.load_document(target=staging)
            if count == 0:
                raise RuntimeError("重建后入库文件数为 0，已终止切换以免清空线上知识库")

            self._write_active_pointer(target_name)
            self.vector_store = staging
            self.collection_name = target_name

            if os.path.exists(backup_path):
                os.remove(backup_path)

            self._drop_collection(previous_name)
            logger.info(f"[重建知识库] 切换完成：{previous_name} → {target_name}，共 {count} 个文件")
            return count
        except Exception as e:
            logger.error(f"[重建知识库] 失败并回滚: {e}", exc_info=True)
            if had_md5 and os.path.exists(backup_path):
                os.replace(backup_path, md5_path)
            if staging is not None:
                try:
                    staging.delete_collection()
                except Exception:
                    pass
            raise


_shared_instance: VectorStoreService | None = None
_shared_lock = threading.Lock()


def get_shared_vector_store() -> VectorStoreService:
    """
    进程内共享单例。
    Agent 的检索前置、rag_summarize 工具、UI 的知识库状态读取都取用同一实例，
    避免重复初始化 Chroma 客户端与嵌入模型（原实现存在两份实例，冷启动翻倍）。
    """
    global _shared_instance
    if _shared_instance is None:
        with _shared_lock:
            if _shared_instance is None:
                _shared_instance = VectorStoreService()
    return _shared_instance


if __name__ == "__main__":
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-" * 20)
