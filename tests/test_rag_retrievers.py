"""Hybrid Retriever 单元测试：BM25、RRF 融合、Hybrid 组合。

不接真 Chroma / jieba / sentence-transformers，全部用 fake。
"""
from __future__ import annotations

import pytest
from langchain_core.documents import Document

from rag.retrievers.hybrid_retriever import HybridRetriever, fuse_rrf, rrf_score
from rag.schemas import RetrievalCandidate


class FakeDense:
    def __init__(self, candidates):
        self._candidates = candidates

    def retrieve(self, query, k=20):
        return self._candidates[:k]


class FakeBM25:
    def __init__(self, candidates, ready=True):
        self._candidates = candidates
        self._ready = ready

    def is_ready(self):
        return self._ready

    def retrieve(self, query, k=20):
        return self._candidates[:k]


def _cand(doc_id, content="", dense=None, sparse=None):
    return RetrievalCandidate(
        doc_id=doc_id,
        document=Document(page_content=content, metadata={"doc_id": doc_id}),
        dense_score=dense,
        sparse_score=sparse,
    )


def test_rrf_score_monotonic_and_positive():
    assert rrf_score(1) > rrf_score(2) > rrf_score(100) > 0


def test_fuse_rrf_merges_by_doc_id_and_sorts_by_fusion_score():
    dense = [_cand("a", dense=0.9), _cand("b", dense=0.6), _cand("c", dense=0.5)]
    bm25 = [_cand("c", sparse=8.0), _cand("a", sparse=3.0), _cand("d", sparse=1.0)]

    fused = fuse_rrf([dense, bm25], k=60)

    ids = [c.doc_id for c in fused]
    assert set(ids) == {"a", "b", "c", "d"}
    # a 在两路都排名靠前，c 在 bm25 第 1，dense 第 3，应该在前列
    assert ids[0] in {"a", "c"}
    # b 只出现在 dense 一路
    assert fused[ids.index("b")].fusion_score == pytest.approx(rrf_score(2))
    # a 在 dense=1, bm25=2
    a_score = fused[ids.index("a")].fusion_score
    assert a_score == pytest.approx(rrf_score(1) + rrf_score(2))


def test_hybrid_falls_back_to_dense_when_bm25_unavailable():
    dense = [_cand("a", dense=0.9), _cand("b", dense=0.5)]
    hybrid = HybridRetriever(
        dense=FakeDense(dense),
        bm25=None,
        rrf_k=60,
        final_k=2,
    )
    result = hybrid.retrieve("x")
    assert [c.doc_id for c in result] == ["a", "b"]
    # fallback 路径也应写入 fusion_score，方便下游按同一字段读
    assert result[0].fusion_score is not None


def test_hybrid_uses_bm25_when_ready():
    dense = [_cand("a", dense=0.9), _cand("b", dense=0.5)]
    bm25 = [_cand("b", sparse=9.0), _cand("c", sparse=3.0)]
    hybrid = HybridRetriever(
        dense=FakeDense(dense),
        bm25=FakeBM25(bm25),
        rrf_k=60,
        final_k=3,
    )
    result = hybrid.retrieve("x")
    ids = [c.doc_id for c in result]
    assert set(ids) == {"a", "b", "c"}
    # b 在两路都出现，融合分应最高
    assert ids[0] == "b"


def test_hybrid_reranker_swaps_order_of_top_n():
    dense = [_cand("a", dense=0.9), _cand("b", dense=0.5), _cand("c", dense=0.3)]

    class FlipReranker:
        def rerank(self, query, candidates, top_n=5):
            # 简单反转 head 顺序，验证 hybrid 会用 rerank 后的排序
            result = list(reversed(candidates))
            for index, candidate in enumerate(result):
                candidate.rerank_score = float(len(result) - index)
            return result

    hybrid = HybridRetriever(
        dense=FakeDense(dense),
        bm25=None,
        reranker=FlipReranker(),
        rerank_top_n=3,
        final_k=3,
    )
    result = hybrid.retrieve("x")
    assert [c.doc_id for c in result] == ["c", "b", "a"]
