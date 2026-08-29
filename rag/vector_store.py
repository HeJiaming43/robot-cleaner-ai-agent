import os
from langchain_core.documents import Document
from model.factory import embed_model
from langchain_chroma import Chroma
from utils.config_handler import chroma_conf
from utils.path_tool import get_abs_path
from langchain_text_splitters import RecursiveCharacterTextSplitter
from utils.file_handler import pdf_loader, txt_loader, listdir_with_allowed_type, get_file_md5_hex
from utils.logger_handler import logger


class VectorStoreService:
    def __init__(self):
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
        )

        self.spliter = RecursiveCharacterTextSplitter(
            chunk_size=chroma_conf["chunk_size"],
            chunk_overlap=chroma_conf["chunk_overlap"],
            separators=chroma_conf["separators"],
            length_function=len,
        )

    def get_retriever(self):
        return self.vector_store.as_retriever(search_kwargs={"k": chroma_conf["k"]})

    def load_document(self):
        md5_store_path = get_abs_path(chroma_conf["md5_hex_store"])

        # 一次性读入已有 MD5 集合，避免后续逐行重复读文件
        existing_md5s: set[str] = set()
        if os.path.exists(md5_store_path):
            with open(md5_store_path, "r", encoding="utf-8") as f:
                existing_md5s = {line.strip() for line in f if line.strip()}

        def check_md5_hex(md5_for_check: str) -> bool:
            return md5_for_check in existing_md5s

        def save_md5_hex(md5_for_check: str):
            existing_md5s.add(md5_for_check)
            with open(md5_store_path, "a", encoding="utf-8") as f:
                f.write(md5_for_check + "\n")

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
        for path in allowed_files_path:
            md5_hex = get_file_md5_hex(path)
            if md5_hex is None:
                logger.warning(f"[加载知识库]{path} MD5计算失败，跳过")
                continue

            if check_md5_hex(md5_hex):
                logger.info(f"[加载知识库]{path}内容已经存在知识库内，跳过")
                continue

            try:
                documents: list[Document] = get_file_documents(path)

                if not documents:
                    logger.warning(f"[加载知识库]{path}内没有有效文本内容，跳过")
                    continue

                split_document: list[Document] = self.spliter.split_documents(documents)

                if not split_document:
                    logger.warning(f"[加载知识库]{path}分片后没有有效文本内容，跳过")
                    continue

                self.vector_store.add_documents(split_document)
                save_md5_hex(md5_hex)

                logger.info(f"[加载知识库]{path} 内容加载成功")
                processed_count += 1
            except Exception as e:
                logger.error(f"[加载知识库]{path}加载失败：{str(e)}", exc_info=True)
                continue

        return processed_count

    def rebuild_knowledge_base(self) -> int:
        md5_path = get_abs_path(chroma_conf["md5_hex_store"])
        if os.path.exists(md5_path):
            os.remove(md5_path)
            logger.info(f"[重建知识库] 已重置 MD5 去重记录: {md5_path}")

        try:
            self.vector_store.delete_collection()
            logger.info("[重建知识库] 已删除原向量集合，开始全量重建")
        except Exception as e:
            logger.warning(f"[重建知识库] 删除向量集合失败（可能原本不存在）: {e}")

        # 删除集合后重新初始化，确保后续 add_documents 能正常工作
        self.vector_store = Chroma(
            collection_name=chroma_conf["collection_name"],
            embedding_function=embed_model,
            persist_directory=get_abs_path(chroma_conf["persist_directory"]),
        )

        return self.load_document()


if __name__ == "__main__":
    vs = VectorStoreService()
    vs.load_document()
    retriever = vs.get_retriever()
    res = retriever.invoke("迷路")
    for r in res:
        print(r.page_content)
        print("-" * 20)
