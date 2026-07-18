"""HybridRetriever：RRF 融合行为 + BM25 缺席时降级为纯 Dense。"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.retrievers.hybrid_retriever import (
    HybridRetriever,
    decide_rerank,
    fuse_hybrid_and_rerank,
    fuse_rrf,
    fuse_rrf_with_anchor,
    rrf_score,
)
from rag.schemas import RetrievalCandidate


class _StubDense:
    def __init__(self, candidates):
        self._candidates = candidates

    def retrieve(self, query, k=20):
        return list(self._candidates)


class _StubBM25:
    def __init__(self, candidates, ready=True):
        self._candidates = candidates
        self._ready = ready

    def is_ready(self):
        return self._ready

    def retrieve(self, query, k=20):
        return list(self._candidates)


def _cand(doc_id: str, dense=None, sparse=None) -> RetrievalCandidate:
    return RetrievalCandidate(
        doc_id=doc_id,
        document=Document(page_content=doc_id, metadata={"doc_id": doc_id}),
        dense_score=dense,
        sparse_score=sparse,
    )


def test_rrf_score_decreases_with_rank():
    assert rrf_score(1) > rrf_score(2) > rrf_score(10)


def test_fuse_rrf_merges_by_doc_id():
    dense = [_cand("A", dense=0.9), _cand("B", dense=0.8)]
    bm25 = [_cand("B", sparse=5.0), _cand("C", sparse=4.0)]

    fused = fuse_rrf([dense, bm25])
    ids = [c.doc_id for c in fused]
    assert set(ids) == {"A", "B", "C"}
    b = next(c for c in fused if c.doc_id == "B")
    a = next(c for c in fused if c.doc_id == "A")
    assert b.fusion_score > a.fusion_score, "同时被两路召回的 B 应该融合分更高"


def test_expanded_rrf_pool_preserves_anchored_head_order():
    dense = [_cand("A", dense=0.9), _cand("B", dense=0.8), _cand("X", dense=0.7)]
    bm25 = [_cand("B", sparse=5.0), _cand("A", sparse=4.0), _cand("X", sparse=3.0)]

    fused = fuse_rrf_with_anchor([dense, bm25], k=60, anchor_k=2)

    assert [candidate.doc_id for candidate in fused] == ["A", "B", "X"]
    assert fused[0].meta["fusion_anchored"] is True


def test_hybrid_without_bm25_falls_back_to_dense():
    dense = _StubDense([_cand("A", dense=0.9), _cand("B", dense=0.8)])
    bm25 = _StubBM25([], ready=False)

    hybrid = HybridRetriever(dense=dense, bm25=bm25, final_k=2)
    result = hybrid.retrieve("q")
    assert [c.doc_id for c in result] == ["A", "B"]
    assert result[0].fusion_score is not None


def test_hybrid_applies_reranker_only_on_head():
    dense = _StubDense([
        _cand("A", dense=0.9),
        _cand("B", dense=0.8),
        _cand("C", dense=0.7),
    ])
    bm25 = _StubBM25([], ready=False)

    class ReverseReranker:
        def rerank(self, query, candidates, top_n=5):
            reversed_list = list(reversed(candidates))
            for i, c in enumerate(reversed_list):
                c.rerank_score = float(len(reversed_list) - i)
            return reversed_list[:top_n]

    hybrid = HybridRetriever(
        dense=dense,
        bm25=bm25,
        reranker=ReverseReranker(),
        rerank_top_n=2,
        final_k=3,
    )
    result = hybrid.retrieve("q")
    # 前 2 个被 rerank 反转成 [B, A]，C 保持在末尾
    assert [c.doc_id for c in result] == ["B", "A", "C"]


def test_hybrid_never_drops_head_candidates_when_reranker_returns_subset():
    dense = _StubDense([_cand(str(index), dense=1.0 - index / 100) for index in range(25)])

    class TruncatingReranker:
        def rerank(self, query, candidates, top_n=5):
            # 模拟旧 BGE 默认只返回 5 条的行为。
            return list(candidates[:5])

    hybrid = HybridRetriever(
        dense=dense,
        bm25=None,
        reranker=TruncatingReranker(),
        rerank_top_n=20,
        final_k=10,
    )

    result = hybrid.retrieve("q")
    assert [candidate.doc_id for candidate in result] == [str(index) for index in range(10)]


def test_hybrid_top_k_zero_returns_no_candidates():
    hybrid = HybridRetriever(dense=_StubDense([_cand("A", dense=0.9)]), final_k=1)
    assert hybrid.retrieve("q", top_k=0) == []


def test_weighted_rerank_fusion_preserves_strong_hybrid_signal():
    candidates = [_cand("A"), _cand("B"), _cand("C")]
    for rank, candidate in enumerate(candidates, start=1):
        candidate.meta["hybrid_rank"] = rank
        candidate.rerank_score = float(rank)

    fused = fuse_hybrid_and_rerank(
        candidates,
        list(reversed(candidates)),
        hybrid_weight=0.7,
        rerank_weight=0.3,
        k=10,
    )

    assert [candidate.doc_id for candidate in fused] == ["A", "B", "C"]
    assert all(candidate.ranking_score is not None for candidate in fused)


def test_query_router_bypasses_model_numbers_and_numeric_constraints():
    assert decide_rerank("X20 出现 E15 错误码", bypass_exact_queries=True).reason == (
        "exact_identifier"
    )
    assert decide_rerank("门槛高度 20mm", bypass_exact_queries=True).reason == (
        "numeric_constraint"
    )
    assert decide_rerank("为什么清扫效果变差", bypass_exact_queries=True).apply is True


def test_hybrid_bypasses_reranker_for_exact_identifier_query():
    class CountingReranker:
        calls = 0

        def rerank(self, query, candidates, top_n=5):
            self.calls += 1
            return list(candidates)

    reranker = CountingReranker()
    hybrid = HybridRetriever(
        dense=_StubDense([_cand("A", dense=0.9), _cand("B", dense=0.8)]),
        reranker=reranker,
        rerank_bypass_exact_queries=True,
        final_k=2,
    )

    result = hybrid.retrieve("X20 出现 E15 错误码")

    assert reranker.calls == 0
    assert [candidate.doc_id for candidate in result] == ["A", "B"]
    assert result[0].meta["rerank_reason"] == "exact_identifier"


def test_weighted_strategy_falls_back_when_reranker_does_not_score_candidates():
    class BrokenReranker:
        def rerank(self, query, candidates, top_n=5):
            return list(reversed(candidates))

    hybrid = HybridRetriever(
        dense=_StubDense([_cand("A", dense=0.9), _cand("B", dense=0.8)]),
        reranker=BrokenReranker(),
        rerank_strategy="weighted_rrf",
        final_k=2,
    )

    result = hybrid.retrieve("清扫效果为什么变差")

    assert [candidate.doc_id for candidate in result] == ["A", "B"]
    assert result[0].meta["rerank_reason"] == "reranker_unavailable"


def test_shadow_strategy_scores_candidates_without_changing_user_order():
    class ReverseReranker:
        def rerank(self, query, candidates, top_n=5):
            result = list(reversed(candidates))
            for index, candidate in enumerate(result):
                candidate.rerank_score = float(len(result) - index)
            return result

    hybrid = HybridRetriever(
        dense=_StubDense([_cand("A", dense=0.9), _cand("B", dense=0.8)]),
        reranker=ReverseReranker(),
        rerank_strategy="shadow",
        final_k=2,
    )

    result = hybrid.retrieve("清扫效果为什么变差")

    assert [candidate.doc_id for candidate in result] == ["A", "B"]
    assert result[0].meta["rerank_evaluated"] is True
    assert result[0].meta["rerank_applied"] is False
    assert result[0].meta["ranking_strategy"] == "shadow"


@pytest.mark.parametrize(
    "kwargs",
    [
        {"rerank_strategy": "unknown"},
        {"rerank_hybrid_weight": float("nan")},
        {"rerank_model_weight": -0.1},
        {"fusion_anchor_k": 0},
    ],
)
def test_hybrid_rejects_invalid_rerank_configuration(kwargs):
    with pytest.raises(ValueError):
        HybridRetriever(dense=_StubDense([_cand("A")]), **kwargs)
