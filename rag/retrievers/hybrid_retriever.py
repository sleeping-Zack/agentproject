"""Hybrid Retriever：Dense + BM25 双路召回，用 Reciprocal Rank Fusion 融合。

为什么用 RRF 而不是加权分数：
    - Dense 分数（0~1 或余弦距离）和 BM25 分数（无上界）尺度差太大，直接加权很敏感。
    - RRF 只看排名，不看分值，跨检索器融合更稳。

reranker 可选，通过 config `retrieval.enable_reranker` 控制。
"""
from __future__ import annotations

from typing import Dict, List, Optional

from rag.rerankers.base import BaseReranker
from rag.retrievers.bm25_retriever import BM25Retriever
from rag.retrievers.dense_retriever import DenseRetriever
from rag.schemas import RetrievalCandidate


def rrf_score(rank: int, k: int = 60) -> float:
    return 1.0 / (k + rank)


def fuse_rrf(
    ranked_lists: List[List[RetrievalCandidate]],
    k: int = 60,
) -> List[RetrievalCandidate]:
    """按 doc_id 融合多路排名。返回按 fusion_score 降序的候选列表。"""
    fused: Dict[str, RetrievalCandidate] = {}
    for candidates in ranked_lists:
        for rank, cand in enumerate(candidates, start=1):
            existing = fused.get(cand.doc_id)
            if existing is None:
                existing = RetrievalCandidate(
                    doc_id=cand.doc_id,
                    document=cand.document,
                    dense_score=cand.dense_score,
                    sparse_score=cand.sparse_score,
                    fusion_score=0.0,
                )
                fused[cand.doc_id] = existing
            else:
                if cand.dense_score is not None:
                    existing.dense_score = max(existing.dense_score or 0.0, cand.dense_score)
                if cand.sparse_score is not None:
                    existing.sparse_score = max(existing.sparse_score or 0.0, cand.sparse_score)
            existing.fusion_score = (existing.fusion_score or 0.0) + rrf_score(rank, k=k)
    return sorted(fused.values(), key=lambda c: c.fusion_score or 0.0, reverse=True)


class HybridRetriever:
    def __init__(
        self,
        dense: DenseRetriever,
        bm25: Optional[BM25Retriever] = None,
        reranker: Optional[BaseReranker] = None,
        dense_k: int = 20,
        bm25_k: int = 20,
        rrf_k: int = 60,
        rerank_top_n: int = 20,
        final_k: int = 5,
    ):
        self.dense = dense
        self.bm25 = bm25
        self.reranker = reranker
        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.rrf_k = rrf_k
        self.rerank_top_n = rerank_top_n
        self.final_k = final_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalCandidate]:
        top_k = top_k or self.final_k

        dense_candidates = self.dense.retrieve(query, k=self.dense_k)
        bm25_candidates: List[RetrievalCandidate] = []
        if self.bm25 is not None and self.bm25.is_ready():
            bm25_candidates = self.bm25.retrieve(query, k=self.bm25_k)

        if bm25_candidates:
            fused = fuse_rrf([dense_candidates, bm25_candidates], k=self.rrf_k)
        else:
            fused = list(dense_candidates)
            for rank, cand in enumerate(fused, start=1):
                cand.fusion_score = rrf_score(rank, k=self.rrf_k)

        if self.reranker is not None and fused:
            head = fused[: self.rerank_top_n]
            reranked = self.reranker.rerank(query, head)
            tail = fused[self.rerank_top_n :]
            return (reranked + tail)[:top_k]

        return fused[:top_k]
