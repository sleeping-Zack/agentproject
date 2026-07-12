"""模型工厂入口。

历史上 chat_model 是模块级单例（一个固定的 ChatTongyi）。引入多模型路由后：
    - 仍然导出 chat_model（向后兼容，等同于"默认 provider"）
    - 新增 model_router，业务侧调用 model_router.invoke(fn) 时会按健康度
      和租户路由，主模型不可用自动降级
    - 业务代码逐步迁移到 router，过渡期两套并存
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional

from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from langchain_core.embeddings import Embeddings

from model.router import DEFAULT_SCENE, ModelRouter, build_default_router_from_config
from utils.config_handler import rag_conf


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(
        self,
        scene: str = DEFAULT_SCENE,
        tenant_id: Optional[str] = None,
    ) -> Optional[Embeddings | BaseChatModel]:
        return model_router.invoke(
            lambda model: model,
            scene=scene,
            tenant_id=tenant_id,
        )


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


model_router: ModelRouter = build_default_router_from_config(rag_conf)

_chat_model = None
_embed_model = None


def _get_chat_model():
    global _chat_model
    if _chat_model is None:
        _chat_model = ChatModelFactory().generator()
    return _chat_model


def _get_embed_model():
    global _embed_model
    if _embed_model is None:
        _embed_model = EmbeddingsFactory().generator()
    return _embed_model


class _LazyModel:
    def __init__(self, getter):
        object.__setattr__(self, "_getter", getter)

    def __getattr__(self, name):
        return getattr(object.__getattribute__(self, "_getter")(), name)

    def resolve(self):
        """Return the real model so LCEL can preserve BaseChatModel semantics."""
        return object.__getattribute__(self, "_getter")()

    def __call__(self, *args, **kwargs):
        target = self.resolve()
        invoke = getattr(target, "invoke", None)
        if invoke is not None:
            return invoke(*args, **kwargs)
        if callable(target):
            return target(*args, **kwargs)
        raise TypeError(f"lazy model target is not invokable: {type(target).__name__}")


chat_model = _LazyModel(_get_chat_model)
embed_model = _LazyModel(_get_embed_model)
