"""本地 BGE-Reranker，走 sentence-transformers 的 CrossEncoder。

sentence-transformers / torch 都是重依赖，且 CI 不装 rerank extras。
所以：
    - 全部 lazy import：类实例化不 import torch
    - 首次 rerank() 才加载模型；失败则退化为原顺序（不阻断主流程）
"""
from __future__ import annotations

from typing import List, Optional

from rag.rerankers.base import BaseReranker
from rag.schemas import RetrievalCandidate


class BGEReranker(BaseReranker):
    def __init__(self, model_name: str = "BAAI/bge-reranker-v2-m3"):
        self.model_name = model_name
        self._model = None
        self._load_failed = False

    def _ensure_model(self) -> Optional[object]:
        if self._model is not None or self._load_failed:
            return self._model
        try:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)
        except Exception:
            self._load_failed = True
            self._model = None
        return self._model

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        top_n: int = 5,
    ) -> List[RetrievalCandidate]:
        if not candidates:
            return []
        model = self._ensure_model()
        if model is None:
            return candidates[:top_n]

        pairs = [(query, c.document.page_content) for c in candidates]
        try:
            scores = model.predict(pairs)
        except Exception:
            return candidates[:top_n]

        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)
        candidates.sort(key=lambda c: c.rerank_score, reverse=True)
        return candidates[:top_n]
