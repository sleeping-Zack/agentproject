from langchain_core.documents import Document

from rag.rerankers.bge_reranker import BGEReranker
from rag.schemas import RetrievalCandidate


def _candidates():
    return [
        RetrievalCandidate("a", Document(page_content="A")),
        RetrievalCandidate("b", Document(page_content="B")),
    ]


def test_bge_reranker_is_operational_only_after_real_successful_prediction():
    reranker = BGEReranker()
    reranker._model = type("Model", (), {"predict": lambda self, pairs: [0.1, 0.9]})()

    ranked = reranker.rerank("query", _candidates(), top_n=2)

    assert [item.doc_id for item in ranked] == ["b", "a"]
    assert reranker.is_active is True
    assert reranker.is_operational is True
    assert reranker.successful_calls == 1
    assert reranker.failed_calls == 0


def test_bge_reranker_marks_inference_failure_as_non_operational():
    def fail(_pairs):
        raise RuntimeError("inference failed")

    reranker = BGEReranker()
    reranker._model = type("Model", (), {"predict": staticmethod(fail)})()

    ranked = reranker.rerank("query", _candidates(), top_n=2)

    assert [item.doc_id for item in ranked] == ["a", "b"]
    assert reranker.is_active is True
    assert reranker.is_operational is False
    assert reranker.failed_calls == 1
    assert reranker.last_error == "inference failed"


def test_bge_reranker_rejects_partial_score_vectors():
    reranker = BGEReranker()
    reranker._model = type("Model", (), {"predict": lambda self, pairs: [0.5]})()

    reranker.rerank("query", _candidates(), top_n=2)

    assert reranker.is_operational is False
    assert reranker.failed_calls == 1
    assert "1 scores for 2 candidates" in reranker.last_error
