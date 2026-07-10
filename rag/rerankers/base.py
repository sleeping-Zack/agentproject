"""Reranker 接口。Rerank 只对少量候选做精排，返回新的顺序和 rerank_score。"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import List

from rag.schemas import RetrievalCandidate


class BaseReranker(ABC):
    @abstractmethod
    def rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        top_n: int = 5,
    ) -> List[RetrievalCandidate]:
        raise NotImplementedError
