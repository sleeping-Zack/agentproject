from langchain_core.documents import Document

from observability.context import bind_request_context
from rag.rag_service import RagSummarizeService
from rag.schemas import RetrievalCandidate
from services.cache import SemanticCache


class FakeChain:
    def invoke(self, payload):
        assert payload["input"] == "怎么保养滤网"
        assert "滤网每周清理" in payload["context"]
        return "建议每周清理滤网。"


class FakeHybrid:
    def __init__(self, candidates):
        self._candidates = candidates
        self.calls = 0

    def retrieve(self, query):
        self.calls += 1
        return self._candidates


class CountingChain:
    def __init__(self):
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        assert "滤网每周清理" in payload["context"]
        return f"建议每周清理滤网。回答版本 {self.calls}"


def _cached_service():
    service = RagSummarizeService.__new__(RagSummarizeService)
    service._semantic_cache = SemanticCache(
        embedder=lambda _: [1.0, 0.0],
        threshold=0.95,
        ttl=10,
    )
    service._retrieval_cfg = {"version": "hybrid-rrf-v1"}
    service._prompt_version = "rag_summarize:v1"
    service._chain = CountingChain()

    doc = Document(
        page_content="滤网每周清理",
        metadata={"source": "manual.pdf", "chunk_id": "c1"},
    )
    candidate = RetrievalCandidate(
        doc_id="manual.pdf#c1",
        document=doc,
        dense_score=0.82,
        fusion_score=0.5,
    )
    service._hybrid = FakeHybrid([candidate])
    return service


def test_rag_summarize_result_returns_structured_evidence():
    service = RagSummarizeService.__new__(RagSummarizeService)
    service._semantic_cache = None
    service._chain = FakeChain()
    service._retrieval_cfg = {}

    doc = Document(
        page_content="滤网每周清理",
        metadata={"source": "manual.pdf", "chunk_id": "c1"},
    )
    candidate = RetrievalCandidate(
        doc_id="manual.pdf#c1",
        document=doc,
        dense_score=0.82,
        fusion_score=0.5,
    )
    service._hybrid = FakeHybrid([candidate])

    result = service.rag_summarize_result("怎么保养滤网")

    assert result.answer.startswith("建议每周清理")
    assert result.evidence[0].id == "manual.pdf#c1"
    assert result.evidence[0].content == "滤网每周清理"
    assert result.evidence[0].score == 0.5


def test_semantic_cache_hit_restores_complete_rag_result():
    service = _cached_service()

    first = service.rag_summarize_result("怎么保养滤网", tenant_id="tenant-a")
    second = service.rag_summarize_result("怎么保养滤网", tenant_id="tenant-a")

    assert service._hybrid.calls == 1
    assert service._chain.calls == 1
    assert second is not first
    assert second.answer == first.answer
    assert second.evidence == first.evidence
    assert second.citations == first.citations
    assert second.evidence[0].id == "manual.pdf#c1"
    assert second.citations[0].evidence_id == "manual.pdf#c1"


def test_semantic_cache_misses_after_corpus_version_change():
    service = _cached_service()

    first = service.rag_summarize_result(
        "怎么保养滤网",
        tenant_id="tenant-a",
        corpus_version="corpus-v1",
    )
    second = service.rag_summarize_result(
        "怎么保养滤网",
        tenant_id="tenant-a",
        corpus_version="corpus-v2",
    )

    assert service._hybrid.calls == 2
    assert service._chain.calls == 2
    assert first.answer != second.answer


def test_semantic_cache_isolates_tenants():
    service = _cached_service()

    with bind_request_context(tenant_id="tenant-a"):
        tenant_a = service.rag_summarize_result("怎么保养滤网")
    with bind_request_context(tenant_id="tenant-b"):
        tenant_b = service.rag_summarize_result("怎么保养滤网")
    with bind_request_context(tenant_id="tenant-a"):
        tenant_a_cached = service.rag_summarize_result("怎么保养滤网")

    assert service._hybrid.calls == 2
    assert service._chain.calls == 2
    assert tenant_a.answer != tenant_b.answer
    assert tenant_a_cached.answer == tenant_a.answer


def test_rag_refuses_before_model_call_when_no_safe_evidence():
    service = RagSummarizeService.__new__(RagSummarizeService)
    service._semantic_cache = None
    service._retrieval_cfg = {}
    service._hybrid = FakeHybrid([])

    class ExplodingChain:
        def invoke(self, _payload):
            raise AssertionError("model must not run without evidence")

    service._chain = ExplodingChain()

    result = service.rag_summarize_result("量子计算股票明天会涨吗")

    assert result.answer.startswith("请求未执行")
    assert result.evidence == []
    assert result.verification["reasons"] == ["evidence_required"]


def test_rag_refuses_low_relevance_out_of_domain_retrieval_before_model_call():
    service = RagSummarizeService.__new__(RagSummarizeService)
    service._semantic_cache = None
    service._retrieval_cfg = {
        "min_dense_relevance": 0.15,
        "min_sparse_relevance": 1.0,
    }
    candidate = RetrievalCandidate(
        doc_id="irrelevant#1",
        document=Document(
            page_content="扫地机器人未来可能支持自动倒垃圾。",
            metadata={"source": "manual.txt", "chunk_id": "1"},
        ),
        dense_score=0.04,
        fusion_score=0.02,
    )
    service._hybrid = FakeHybrid([candidate])

    class ExplodingChain:
        def invoke(self, _payload):
            raise AssertionError("model must not run for out-of-domain retrieval")

    service._chain = ExplodingChain()

    result = service.rag_summarize_result("量子计算股票明天会涨吗")

    assert result.answer.startswith("请求未执行")
    assert result.verification["reasons"] == ["retrieval_relevance_below_threshold"]


def test_rag_refuses_generated_claim_that_is_not_grounded_in_evidence():
    service = RagSummarizeService.__new__(RagSummarizeService)
    service._semantic_cache = None
    service._retrieval_cfg = {}
    service.verify_generation = True
    service.verifier = None
    service._chain = type(
        "UnsupportedChain",
        (),
        {"invoke": lambda self, payload: "可以直接用水冲洗电机。"},
    )()
    candidate = RetrievalCandidate(
        doc_id="manual.pdf#c1",
        document=Document(
            page_content="滤网应每周拆下并使用干布清理。",
            metadata={"source": "manual.pdf", "chunk_id": "c1"},
        ),
        dense_score=0.8,
        fusion_score=0.5,
    )
    service._hybrid = FakeHybrid([candidate])

    result = service.rag_summarize_result("滤网如何维护")

    assert result.answer.startswith("请求未执行")
    assert result.verification["passed"] is False
    assert "unsupported_claim_rate_exceeded" in result.verification["reasons"]
