"""
总结服务类：用户提问，搜索参考资料，将提问和参考资料提交给模型，让模型总结回复
"""
from langchain_core.documents import Document
from langchain_core.output_parsers import StrOutputParser
from rag.vector_store import VectorStoreService
from utils.prompt_loader import load_rag_prompts
from langchain_core.prompts import PromptTemplate
from model.factory import chat_model


class RagSummarizeService(object):
    def __init__(self, vector_store: VectorStoreService | None = None):
        # 允许注入共享的 VectorStoreService，避免重复实例化
        self.vector_store = vector_store or VectorStoreService()
        self.prompt_text = load_rag_prompts()
        self.prompt_template = PromptTemplate.from_template(self.prompt_text)
        self.model = chat_model
        self.chain = self._init_chain()


    def _init_chain(self):
        chain = self.prompt_template | self.model | StrOutputParser()
        return chain

    @property
    def retriever(self):
        """
        每次访问现取，不缓存实例。
        原实现在构造期缓存 retriever，知识库蓝绿切换后仍指向旧集合，
        会持续检索已下线的内容（对应问题 P1-1）。
        """
        return self.vector_store.get_retriever()

    def retrieve_docs(self, query: str) -> list[Document]:
        return self.vector_store.get_retriever().invoke(query)

    def rag_summarize(self, query: str) -> str:

        context_docs = self.retrieve_docs(query)

        context = ""
        counter = 0
        for doc in context_docs:
            counter += 1
            context += f"[参考资料{counter}]: 参考资料: {doc.page_content} | 参考元数据: {doc.metadata}\n"

        return self.chain.invoke(
            {
                "input": query,
                "context": context,
            }
        )

if __name__ == "__main__":
    rag = RagSummarizeService()
    print(rag.rag_summarize("小户型适合哪种扫地机器人"))
