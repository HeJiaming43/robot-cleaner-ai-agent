import os
from abc import ABC, abstractmethod
from langchain_core.embeddings import Embeddings
from langchain_core.language_models import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_openai import ChatOpenAI
from utils.config_handler import rag_conf


# DashScope 的 OpenAI 兼容端点。
# 原生 SDK 端点（chat_models.tongyi.ChatTongyi）对新模型（如 qwen3.8-flash）会报
# "url error, please check url"，因为旧版 SDK 里没有该模型的端点映射；
# 兼容端点按请求体里的 model 字段路由，新模型无需升级 SDK 即可使用。
DASHSCOPE_COMPATIBLE_BASE_URL = "https://dashscope.aliyuncs.com/compatible-mode/v1"


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self):
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> BaseChatModel:
        # max_retries：底层请求的失败重试次数，缓解网络抖动与限流瞬时错误（问题 P1-5）
        kwargs = {}
        if not rag_conf.get("chat_enable_thinking", True):
            # qwen3.8-flash 等模型默认开启思考，实测输出 token 约 3 倍、耗时约 2 倍。
            # 客服场景对首字与总时长敏感，且多步流程已由提示词写死，故默认关闭，可配置打开。
            kwargs["extra_body"] = {"enable_thinking": False}

        return ChatOpenAI(
            model = rag_conf["chat_model_name"],
            base_url = str(rag_conf.get("chat_base_url") or DASHSCOPE_COMPATIBLE_BASE_URL),
            api_key = os.environ.get("DASHSCOPE_API_KEY", ""),
            streaming = True,
            max_retries = int(rag_conf.get("max_retries", 2)),
            **kwargs,
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Embeddings:
        return DashScopeEmbeddings(model = rag_conf["embedding_model_name"])


_chat_model_instance = None
_embed_model_instance = None


def get_chat_model() -> BaseChatModel:
    """懒加载单例：首次调用时才初始化对话模型。"""
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
