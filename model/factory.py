from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Optional
from langchain_core.embeddings import Embeddings
from langchain_community.chat_models.tongyi import BaseChatModel
from langchain_community.embeddings import DashScopeEmbeddings
from utils.config_handler import rag_conf
from model.providers import ProviderConfig, build_model_provider


class BaseModelFactory(ABC):
    @abstractmethod
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        pass


class ChatModelFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        provider = build_model_provider(
            ProviderConfig(
                provider=rag_conf.get("model_provider", "tongyi"),
                model_name=rag_conf["chat_model_name"],
            )
        )
        return provider.as_langchain_model()


class EmbeddingsFactory(BaseModelFactory):
    def generator(self) -> Optional[Embeddings | BaseChatModel]:
        return DashScopeEmbeddings(model=rag_conf["embedding_model_name"])


chat_model = ChatModelFactory().generator()
embed_model = EmbeddingsFactory().generator()
