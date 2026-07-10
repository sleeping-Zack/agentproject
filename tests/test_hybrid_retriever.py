"""HybridRetriever：RRF 融合行为 + BM25 缺席时降级为纯 Dense。"""
from __future__ import annotations

from langchain_core.documents import Document

from rag.retrievers.hybrid_retriever import HybridRetriever, fuse_rrf, rrf_score
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
