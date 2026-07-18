from langchain_core.documents import Document

from rag.rerankers.bge_reranker import BGEReranker, build_rerank_passage
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


def test_build_rerank_passage_includes_stable_metadata_before_content():
    candidate = RetrievalCandidate(
        "doc",
        Document(
            page_content="检查滚刷并清理缠绕毛发",
            metadata={
                "source_name": "X20故障手册.txt",
                "product_model": "X20",
                "section_title": "滚刷故障",
                "page": 8,
            },
        ),
    )

    passage = build_rerank_passage(candidate, max_chars=200)

    assert passage.startswith("文档：X20故障手册.txt")
    assert "产品型号：X20" in passage
    assert "章节：滚刷故障" in passage
    assert "页码：8" in passage
    assert passage.endswith("正文：检查滚刷并清理缠绕毛发")


def test_build_rerank_passage_respects_character_limit():
    candidate = RetrievalCandidate(
        "doc",
        Document(page_content="内容" * 100, metadata={"source_name": "手册.txt"}),
    )

    assert len(build_rerank_passage(candidate, max_chars=40)) == 40


def test_bge_reranker_sends_structured_passages_to_model():
    captured = []

    class Model:
        def predict(self, pairs):
            captured.extend(pairs)
            return [0.5]

    reranker = BGEReranker(max_document_chars=200)
    reranker._model = Model()
    candidate = RetrievalCandidate(
        "doc",
        Document(page_content="正文", metadata={"document_title": "维护保养"}),
    )

    reranker.rerank("怎么维护", [candidate], top_n=1)

    assert captured == [("怎么维护", "文档：维护保养\n正文：正文")]
    assert reranker.last_latency_ms is not None
