"""Hybrid Retriever：Dense + BM25 双路召回，用 Reciprocal Rank Fusion 融合。

为什么用 RRF 而不是加权分数：
    - Dense 分数（0~1 或余弦距离）和 BM25 分数（无上界）尺度差太大，直接加权很敏感。
    - RRF 只看排名，不看分值，跨检索器融合更稳。

reranker 可选，通过 config `retrieval.enable_reranker` 控制。
"""
from __future__ import annotations

import re
from dataclasses import dataclass
from math import isfinite
from typing import Dict, List, Optional, Sequence

from rag.rerankers.base import BaseReranker
from rag.retrievers.bm25_retriever import BM25Retriever
from rag.retrievers.dense_retriever import DenseRetriever
from rag.schemas import RetrievalCandidate


_MODEL_OR_ERROR_CODE = re.compile(
    r"(?i)(?<![a-z0-9])(?:[a-z]{1,8}[-_]?\d{1,8}[a-z0-9_-]*)(?![a-z0-9])"
)
_NUMERIC_CONSTRAINT = re.compile(
    r"(?i)\d+(?:\.\d+)?\s*(?:pa|kpa|w|v|mah|mm|cm|m|db|h|min|%|％|"
    r"毫米|厘米|米|平方米|分钟|小时|天|度|摄氏度|级)"
)


@dataclass(frozen=True)
class RerankDecision:
    apply: bool
    reason: str


def decide_rerank(query: str, *, bypass_exact_queries: bool) -> RerankDecision:
    """精确标识符和数值约束优先保留 BM25/RRF 排名。"""
    normalized = query.strip()
    if not normalized:
        return RerankDecision(False, "empty_query")
    if not bypass_exact_queries:
        return RerankDecision(True, "routing_disabled")
    if _MODEL_OR_ERROR_CODE.search(normalized):
        return RerankDecision(False, "exact_identifier")
    if _NUMERIC_CONSTRAINT.search(normalized):
        return RerankDecision(False, "numeric_constraint")
    return RerankDecision(True, "semantic_query")


def rrf_score(rank: int, k: int = 60) -> float:
    if rank < 1 or k < 0:
        raise ValueError("RRF rank must be >= 1 and k must be >= 0")
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
                    meta=dict(cand.meta),
                )
                fused[cand.doc_id] = existing
            else:
                if cand.dense_score is not None:
                    existing.dense_score = max(existing.dense_score or 0.0, cand.dense_score)
                if cand.sparse_score is not None:
                    existing.sparse_score = max(existing.sparse_score or 0.0, cand.sparse_score)
            retrieved_by = set(existing.meta.get("retrieved_by") or [])
            retrieved_by.update(cand.meta.get("retrieved_by") or [])
            existing.meta["retrieved_by"] = sorted(retrieved_by)
            if cand.dense_score is not None:
                existing.meta["dense_rank"] = rank
            if cand.sparse_score is not None:
                existing.meta["bm25_rank"] = rank
            existing.fusion_score = (existing.fusion_score or 0.0) + rrf_score(rank, k=k)
    ranked = sorted(fused.values(), key=lambda c: (-(c.fusion_score or 0.0), c.doc_id))
    for rank, candidate in enumerate(ranked, start=1):
        candidate.meta["hybrid_rank"] = rank
    return ranked


def fuse_rrf_with_anchor(
    ranked_lists: Sequence[Sequence[RetrievalCandidate]],
    *,
    k: int,
    anchor_k: Optional[int],
) -> List[RetrievalCandidate]:
    """扩大候选池时锚定原头部，避免新增尾部候选破坏已验证的 TopK。"""
    full_lists = [list(candidates) for candidates in ranked_lists]
    expanded = fuse_rrf(full_lists, k=k)
    if anchor_k is None:
        return expanded
    if anchor_k <= 0:
        raise ValueError("fusion anchor k must be greater than zero")
    if all(len(candidates) <= anchor_k for candidates in full_lists):
        return expanded

    anchored = fuse_rrf(
        [candidates[:anchor_k] for candidates in full_lists],
        k=k,
    )
    anchored_ids = {candidate.doc_id for candidate in anchored}
    result = anchored + [
        candidate for candidate in expanded if candidate.doc_id not in anchored_ids
    ]
    for rank, candidate in enumerate(result, start=1):
        candidate.meta["hybrid_rank"] = rank
        candidate.meta["fusion_anchored"] = True
    return result


def fuse_hybrid_and_rerank(
    hybrid_candidates: Sequence[RetrievalCandidate],
    reranked_candidates: Sequence[RetrievalCandidate],
    *,
    hybrid_weight: float,
    rerank_weight: float,
    k: int,
) -> List[RetrievalCandidate]:
    """用排名而非不可比的原始分数，保守融合 RRF 与 Cross-Encoder。"""
    if (
        not isfinite(hybrid_weight)
        or not isfinite(rerank_weight)
        or hybrid_weight < 0
        or rerank_weight < 0
        or hybrid_weight + rerank_weight <= 0
    ):
        raise ValueError("rerank fusion weights must be non-negative and not both zero")
    if k < 0:
        raise ValueError("rerank fusion k must be >= 0")

    total_weight = hybrid_weight + rerank_weight
    normalized_hybrid_weight = hybrid_weight / total_weight
    normalized_rerank_weight = rerank_weight / total_weight
    rerank_ranks = {
        candidate.doc_id: rank
        for rank, candidate in enumerate(reranked_candidates, start=1)
    }

    fused = list(hybrid_candidates)
    for fallback_rank, candidate in enumerate(fused, start=1):
        hybrid_rank = int(candidate.meta.get("hybrid_rank") or fallback_rank)
        rerank_rank = rerank_ranks.get(candidate.doc_id, len(fused) + fallback_rank)
        candidate.meta["rerank_rank"] = rerank_rank
        candidate.ranking_score = (
            normalized_hybrid_weight * rrf_score(hybrid_rank, k=k)
            + normalized_rerank_weight * rrf_score(rerank_rank, k=k)
        )
        candidate.meta["ranking_strategy"] = "weighted_rrf"

    return sorted(
        fused,
        key=lambda candidate: (
            -(candidate.ranking_score or 0.0),
            int(candidate.meta.get("hybrid_rank") or 10**9),
            candidate.doc_id,
        ),
    )


def _complete_reranked_head(
    head: Sequence[RetrievalCandidate],
    returned: Sequence[RetrievalCandidate],
) -> List[RetrievalCandidate]:
    original_by_id = {candidate.doc_id: candidate for candidate in head}
    completed: List[RetrievalCandidate] = []
    seen = set()
    for candidate in returned:
        original = original_by_id.get(candidate.doc_id)
        if original is None or candidate.doc_id in seen:
            continue
        if candidate.rerank_score is not None:
            original.rerank_score = candidate.rerank_score
        completed.append(original)
        seen.add(candidate.doc_id)
    completed.extend(candidate for candidate in head if candidate.doc_id not in seen)
    return completed


def rerank_fused_candidates(
    query: str,
    fused: Sequence[RetrievalCandidate],
    *,
    reranker: Optional[BaseReranker],
    rerank_top_n: int,
    strategy: str = "replace",
    hybrid_weight: float = 0.7,
    rerank_weight: float = 0.3,
    fusion_k: int = 10,
    bypass_exact_queries: bool = False,
) -> List[RetrievalCandidate]:
    """对融合候选应用统一路由、契约防御和最终排序策略。"""
    candidates = list(fused)
    if reranker is None or not candidates or rerank_top_n <= 0:
        return candidates
    if strategy not in {"replace", "shadow", "weighted_rrf"}:
        raise ValueError(f"unsupported rerank strategy: {strategy}")

    decision = decide_rerank(query, bypass_exact_queries=bypass_exact_queries)
    for rank, candidate in enumerate(candidates, start=1):
        candidate.rerank_score = None
        candidate.ranking_score = None
        candidate.meta["rerank_applied"] = False
        candidate.meta["rerank_evaluated"] = False
        candidate.meta["rerank_reason"] = decision.reason
        candidate.meta["final_rank"] = rank
        candidate.meta["ranking_strategy"] = "hybrid"
    if not decision.apply:
        return candidates

    head = candidates[:rerank_top_n]
    returned = reranker.rerank(query, list(head), top_n=len(head))
    completed = _complete_reranked_head(head, returned)
    for rank, candidate in enumerate(completed, start=1):
        candidate.meta["rerank_rank"] = rank

    if not all(candidate.rerank_score is not None for candidate in head):
        for candidate in candidates:
            candidate.meta["rerank_reason"] = "reranker_unavailable"
        return candidates

    if strategy == "shadow":
        ordered_head = head
        for candidate in ordered_head:
            candidate.meta["ranking_strategy"] = "shadow"
    elif strategy == "weighted_rrf":
        ordered_head = fuse_hybrid_and_rerank(
            head,
            completed,
            hybrid_weight=hybrid_weight,
            rerank_weight=rerank_weight,
            k=fusion_k,
        )
    else:
        ordered_head = completed
        for candidate in ordered_head:
            candidate.meta["ranking_strategy"] = "replace"

    result = ordered_head + candidates[rerank_top_n:]
    head_ids = {candidate.doc_id for candidate in head}
    for rank, candidate in enumerate(result, start=1):
        in_rerank_head = candidate.doc_id in head_ids
        candidate.meta["rerank_evaluated"] = in_rerank_head
        candidate.meta["rerank_applied"] = in_rerank_head and strategy != "shadow"
        if not in_rerank_head:
            candidate.meta["rerank_reason"] = "outside_rerank_window"
        candidate.meta["final_rank"] = rank
    return result


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
        rerank_strategy: str = "replace",
        rerank_hybrid_weight: float = 0.7,
        rerank_model_weight: float = 0.3,
        rerank_fusion_k: int = 10,
        rerank_bypass_exact_queries: bool = False,
        fusion_anchor_k: Optional[int] = None,
    ):
        if dense_k <= 0 or bm25_k <= 0:
            raise ValueError("dense_k and bm25_k must be greater than zero")
        if rrf_k < 0 or rerank_fusion_k < 0:
            raise ValueError("RRF k values must be >= 0")
        if rerank_top_n < 0 or final_k < 0:
            raise ValueError("rerank_top_n and final_k must be >= 0")
        if fusion_anchor_k is not None and fusion_anchor_k <= 0:
            raise ValueError("fusion_anchor_k must be greater than zero")
        if rerank_strategy not in {"replace", "shadow", "weighted_rrf"}:
            raise ValueError(f"unsupported rerank strategy: {rerank_strategy}")
        if (
            not isfinite(rerank_hybrid_weight)
            or not isfinite(rerank_model_weight)
            or rerank_hybrid_weight < 0
            or rerank_model_weight < 0
            or rerank_hybrid_weight + rerank_model_weight <= 0
        ):
            raise ValueError(
                "rerank weights must be finite, non-negative and not both zero"
            )
        self.dense = dense
        self.bm25 = bm25
        self.reranker = reranker
        self.dense_k = dense_k
        self.bm25_k = bm25_k
        self.rrf_k = rrf_k
        self.rerank_top_n = rerank_top_n
        self.final_k = final_k
        self.rerank_strategy = rerank_strategy
        self.rerank_hybrid_weight = rerank_hybrid_weight
        self.rerank_model_weight = rerank_model_weight
        self.rerank_fusion_k = rerank_fusion_k
        self.rerank_bypass_exact_queries = rerank_bypass_exact_queries
        self.fusion_anchor_k = fusion_anchor_k

    def retrieve(self, query: str, top_k: Optional[int] = None) -> List[RetrievalCandidate]:
        top_k = self.final_k if top_k is None else top_k
        if top_k <= 0:
            return []

        dense_candidates = self.dense.retrieve(query, k=self.dense_k)
        bm25_candidates: List[RetrievalCandidate] = []
        if self.bm25 is not None and self.bm25.is_ready():
            bm25_candidates = self.bm25.retrieve(query, k=self.bm25_k)

        if bm25_candidates:
            fused = fuse_rrf_with_anchor(
                [dense_candidates, bm25_candidates],
                k=self.rrf_k,
                anchor_k=self.fusion_anchor_k,
            )
        else:
            fused = list(dense_candidates)
            for rank, cand in enumerate(fused, start=1):
                cand.fusion_score = rrf_score(rank, k=self.rrf_k)
                cand.meta.setdefault("dense_rank", rank)
                cand.meta.setdefault("retrieved_by", ["dense"])
                cand.meta["hybrid_rank"] = rank

        if self.reranker is not None and fused:
            ranked = rerank_fused_candidates(
                query,
                fused,
                reranker=self.reranker,
                rerank_top_n=self.rerank_top_n,
                strategy=self.rerank_strategy,
                hybrid_weight=self.rerank_hybrid_weight,
                rerank_weight=self.rerank_model_weight,
                fusion_k=self.rerank_fusion_k,
                bypass_exact_queries=self.rerank_bypass_exact_queries,
            )
            return ranked[:top_k]

        return fused[:top_k]
