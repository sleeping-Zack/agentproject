from langchain_core.documents import Document

import rag.rag_service as rag_service
from rag.rag_service import RagSummarizeService


class FakeChain:
    def invoke(self, payload):
        assert payload["input"] == "怎么保养滤网"
        assert "滤网每周清理" in payload["context"]
        return "建议每周清理滤网。"


def test_rag_summarize_result_returns_structured_evidence(monkeypatch):
    service = RagSummarizeService.__new__(RagSummarizeService)
    service._semantic_cache = None
    service._chain = FakeChain()
    docs = [
        Document(
            page_content="滤网每周清理",
            metadata={"source": "manual.pdf", "chunk_id": "c1", "score": 0.82},
        )
    ]
    service.retriever_docs = lambda query: docs
    monkeypatch.setattr(rag_service, "hybrid_rank", lambda *args, **kwargs: docs)

    result = service.rag_summarize_result("怎么保养滤网")

    assert result.answer.startswith("建议每周清理")
    assert result.evidence[0].id == "manual.pdf#c1"
    assert result.evidence[0].content == "滤网每周清理"
    assert result.evidence[0].score == 0.82
