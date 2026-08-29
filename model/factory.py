import os
from abc import ABC, abstractmethod
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_community.chat_models.tongyi import ChatTongyi
from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self):
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> BaseChatModel:
        return ChatTongyi(model = rag_conf["chat_model_name"], streaming=True)


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Embeddings:
        return DashScopeEmbeddings(model = rag_conf["embedding_model_name"])


_chat_model_instance = None
_embed_model_instance = None


def get_chat_model() -> BaseChatModel:
    """懒加载单例：首次调用时才初始化 ChatTongyi。"""
    global _chat_model_instance
    if _chat_model_instance is None:
        _chat_model_instance = ChatModelFactory().generator()
    return _chat_model_instance


def get_embed_model() -> Embeddings:
    """懒加载单例：首次调用时才初始化 DashScopeEmbeddings。"""
    global _embed_model_instance
    if _embed_model_instance is None:
        _embed_model_instance = EmbeddingsFactory().generator()
    return _embed_model_instance


# 模块级懒加载：兼容 `from model.factory import chat_model` 的旧写法，
# 且返回的是真实 Runnable 对象，不破坏 LangChain LCEL 管道（prompt | model | parser）。
def __getattr__(name):
    if name == "chat_model":
        return get_chat_model()
    if name == "embed_model":
        return get_embed_model()
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
