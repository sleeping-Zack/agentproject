"""Dense 向量检索：直接使用 Chroma 的 similarity_search_with_relevance_scores。

不再走 as_retriever()，因为那个接口不暴露 score，导致上层没法做加权/融合。
"""
from __future__ import annotations

from typing import List

from rag.schemas import RetrievalCandidate, stable_doc_id


class DenseRetriever:
    def __init__(self, vector_store):
        self.vector_store = vector_store

    def retrieve(self, query: str, k: int = 20) -> List[RetrievalCandidate]:
        try:
            pairs = self.vector_store.similarity_search_with_relevance_scores(query, k=k)
        except Exception:
            docs = self.vector_store.similarity_search(query, k=k)
            pairs = [(doc, 0.0) for doc in docs]

        candidates: List[RetrievalCandidate] = []
        for index, (doc, score) in enumerate(pairs):
            candidates.append(
                RetrievalCandidate(
                    doc_id=stable_doc_id(doc, fallback_index=index),
                    document=doc,
                    dense_score=float(score) if score is not None else 0.0,
                )
            )
        return candidates
