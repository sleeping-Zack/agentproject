from langchain_core.documents import Document

from rag.rag_service import RagSummarizeService
from rag.schemas import RetrievalCandidate


class FakeChain:
    def invoke(self, payload):
        assert payload["input"] == "怎么保养滤网"
        assert "滤网每周清理" in payload["context"]
        return "建议每周清理滤网。"


class FakeHybrid:
    def __init__(self, candidates):
        self._candidates = candidates

    def retrieve(self, query):
        return self._candidates


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
