from __future__ import annotations

from langchain_core.documents import Document

from rag.rerankers.factory import build_reranker
from rag.rerankers.remote_reranker import RemoteReranker
from rag.schemas import RetrievalCandidate


def _candidate(doc_id: str) -> RetrievalCandidate:
    return RetrievalCandidate(
        doc_id=doc_id,
        document=Document(
            page_content=f"content {doc_id}",
            metadata={"doc_id": doc_id, "section_title": "Troubleshooting"},
        ),
    )


def test_remote_reranker_scores_all_candidates_before_reordering():
    captured = {}

    def transport(endpoint, payload, timeout):
        captured.update(endpoint=endpoint, payload=payload, timeout=timeout)
        return {"scores": [0.1, 0.9]}

    reranker = RemoteReranker(
        "http://reranker.internal/rerank",
        timeout_seconds=0.5,
        transport=transport,
    )
    first, second = _candidate("first"), _candidate("second")

    result = reranker.rerank("query", [first, second], top_n=2)

    assert [item.doc_id for item in result] == ["second", "first"]
    assert first.rerank_score == 0.1
    assert second.rerank_score == 0.9
    assert captured["timeout"] == 0.5
    assert captured["payload"]["top_n"] == 2
    assert "章节：Troubleshooting" in captured["payload"]["documents"][0]


def test_remote_reranker_timeout_opens_circuit_and_preserves_hybrid_order():
    calls = 0

    def transport(_endpoint, _payload, _timeout):
        nonlocal calls
        calls += 1
        raise TimeoutError("slow service")

    reranker = RemoteReranker(
        "http://reranker.internal/rerank",
        failure_threshold=1,
        recovery_timeout=60,
        transport=transport,
    )
    candidates = [_candidate("first"), _candidate("second")]

    assert reranker.rerank("query", candidates, top_n=2) == candidates
    assert reranker.rerank("query", candidates, top_n=2) == candidates
    assert calls == 1
    assert reranker.failed_calls == 1
    assert reranker.short_circuited_calls == 1
    assert all(candidate.rerank_score is None for candidate in candidates)


def test_remote_reranker_rejects_partial_response_without_partial_scores():
    reranker = RemoteReranker(
        "http://reranker.internal/rerank",
        transport=lambda *_args: {"results": [{"index": 0, "score": 0.9}]},
    )
    candidates = [_candidate("first"), _candidate("second")]

    assert reranker.rerank("query", candidates, top_n=2) == candidates
    assert all(candidate.rerank_score is None for candidate in candidates)
    assert reranker.last_error == "ValueError"


def test_reranker_factory_builds_remote_backend():
    reranker = build_reranker(
        {
            "enable_reranker": True,
            "reranker_backend": "remote",
            "reranker_url": "http://reranker.internal/rerank",
        }
    )

    assert isinstance(reranker, RemoteReranker)
