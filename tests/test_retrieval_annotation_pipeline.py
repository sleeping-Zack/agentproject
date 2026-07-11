from __future__ import annotations

import json
import threading
from dataclasses import dataclass, field
from typing import Any

import pytest

from scripts.generate_retrieval_golden import (
    DEFAULT_OUTPUT,
    QueryRetrievalError,
    QueryTimeoutError,
    RetrievalRuntime,
    VersionInfo,
    generate_candidate_file,
    merge_route_candidates,
    retrieve_candidate_pool,
)
from scripts.split_retrieval_labels import build_golden, load_reviewed_labels
from scripts.validate_retrieval_manifest import (
    ManifestValidationError,
    sha256_file,
    validate_manifest,
)


@dataclass
class FakeDocument:
    page_content: str
    metadata: dict[str, Any]


@dataclass
class FakeCandidate:
    doc_id: str
    document: FakeDocument
    dense_score: float | None = None
    sparse_score: float | None = None
    fusion_score: float | None = None
    rerank_score: float | None = None
    meta: dict[str, Any] = field(default_factory=dict)


@pytest.fixture
def versions() -> VersionInfo:
    return VersionInfo(
        corpus_version="corpus-v1",
        corpus_hash="a" * 64,
        chunk_version="chunk-v2",
        embedding_model="fake-embedding",
        retrieval_version="hybrid-v3",
    )


def candidate(
    doc_id: str,
    text: str,
    *,
    dense_score: float | None = None,
    sparse_score: float | None = None,
    fusion_score: float | None = None,
    rerank_score: float | None = None,
    meta: dict[str, Any] | None = None,
) -> FakeCandidate:
    return FakeCandidate(
        doc_id=doc_id,
        document=FakeDocument(
            page_content=text,
            metadata={
                "doc_id": doc_id,
                "source_name": "knowledge.txt",
                "chunk_version": "chunk-v2",
            },
        ),
        dense_score=dense_score,
        sparse_score=sparse_score,
        fusion_score=fusion_score,
        rerank_score=rerank_score,
        meta=meta or {},
    )


def test_merge_route_candidates_unions_all_routes_without_keyword_filter(versions):
    dense_a = candidate("a", "text that does not contain an expected keyword", dense_score=0.9)
    dense_b = candidate("b", "shared", dense_score=0.8)
    bm25_b = candidate("b", "shared", sparse_score=8.0)
    bm25_c = candidate("c", "sparse only", sparse_score=6.0)
    hybrid_b = candidate(
        "b",
        "shared",
        dense_score=0.8,
        sparse_score=8.0,
        fusion_score=0.03,
        rerank_score=0.95,
        meta={"dense_rank": 2, "bm25_rank": 1, "hybrid_rank": 1, "rerank_rank": 1},
    )
    hybrid_a = candidate(
        "a",
        "text that does not contain an expected keyword",
        dense_score=0.9,
        fusion_score=0.02,
        meta={"dense_rank": 1, "hybrid_rank": 2},
    )

    merged = merge_route_candidates(
        {
            "dense": [dense_a, dense_b],
            "bm25": [bm25_b, bm25_c],
            "hybrid": [hybrid_b, hybrid_a],
        },
        versions,
    )

    assert [item["doc_id"] for item in merged] == ["b", "a", "c"]
    by_id = {item["doc_id"]: item for item in merged}
    assert by_id["b"]["retrieved_by"] == ["dense", "bm25", "hybrid"]
    assert by_id["b"]["dense_rank"] == 2
    assert by_id["b"]["bm25_rank"] == 1
    assert by_id["b"]["hybrid_rank"] == 1
    assert by_id["b"]["rerank_rank"] == 1
    assert by_id["c"]["dense_rank"] is None
    assert by_id["a"]["chunk_text"].startswith("text that does not contain")
    required = {
        "chunk_text",
        "source",
        "metadata",
        "retrieved_by",
        "dense_rank",
        "dense_score",
        "bm25_rank",
        "sparse_score",
        "hybrid_rank",
        "fusion_score",
        "rerank_rank",
        "rerank_score",
        "corpus_version",
        "chunk_version",
        "embedding_model",
        "retrieval_version",
    }
    assert required <= set(by_id["b"])


class FakeRetriever:
    def __init__(self, results):
        self.results = results
        self.calls = []

    def is_ready(self):
        return True

    def retrieve(self, query, k):
        self.calls.append((query, k))
        return list(self.results)


class FakeReranker:
    def __init__(self):
        self.calls = 0

    def rerank(self, query, candidates, top_n):
        self.calls += 1
        for score, item in enumerate(candidates, start=1):
            item.rerank_score = float(score)
        return list(reversed(candidates))


def test_retrieve_candidate_pool_reuses_dense_and_bm25_results(versions):
    dense = FakeRetriever(
        [candidate("a", "dense", dense_score=0.9), candidate("b", "shared", dense_score=0.8)]
    )
    bm25 = FakeRetriever(
        [candidate("b", "shared", sparse_score=5.0), candidate("c", "sparse", sparse_score=4.0)]
    )
    reranker = FakeReranker()
    runtime = RetrievalRuntime(dense=dense, bm25=bm25, reranker=reranker, rerank_top_n=3)

    result = retrieve_candidate_pool("query", runtime=runtime, top_k=2, versions=versions)

    assert dense.calls == [("query", 2)]
    assert bm25.calls == [("query", 2)]
    assert reranker.calls == 1
    assert {item["doc_id"] for item in result} == {"a", "b", "c"}
    assert sum("hybrid" in item["retrieved_by"] for item in result) == 2


def test_generate_candidate_file_resumes_completed_cases(tmp_path, versions):
    output = tmp_path / "candidates.jsonl"
    output.write_text(json.dumps({"case_id": "one"}) + "\n", encoding="utf-8")
    calls = []

    def loader(query):
        calls.append(query)
        return []

    result = generate_candidate_file(
        [{"id": "one", "query": "q1"}, {"id": "two", "query": "q2"}],
        output_path=output,
        candidate_loader=loader,
        versions=versions,
        timeout_seconds=1,
        resume=True,
    )

    assert result == {"written": 1, "skipped": 1}
    assert calls == ["q2"]
    assert len(output.read_text(encoding="utf-8").splitlines()) == 2


def test_retrieval_error_log_does_not_expose_provider_message(tmp_path, versions, capsys):
    def loader(_query):
        raise RuntimeError("request failed for api_key=sk-secret-value")

    with pytest.raises(QueryRetrievalError):
        generate_candidate_file(
            [{"id": "one", "query": "q1"}],
            output_path=tmp_path / "candidates.jsonl",
            candidate_loader=loader,
            versions=versions,
            timeout_seconds=1,
        )

    captured = capsys.readouterr()
    assert "sk-secret-value" not in captured.err
    assert '"error_type": "RuntimeError"' in captured.err


def test_query_timeout_stops_before_next_case(tmp_path, versions):
    calls = []

    def loader(query):
        calls.append(query)
        threading.Event().wait(0.1)
        return []

    with pytest.raises(QueryTimeoutError):
        generate_candidate_file(
            [{"id": "one", "query": "q1"}, {"id": "two", "query": "q2"}],
            output_path=tmp_path / "candidates.jsonl",
            candidate_loader=loader,
            versions=versions,
            timeout_seconds=0.01,
        )
    assert calls == ["q1"]


def reviewed_case(case_id, labels=None, status="reviewed"):
    return {
        "case_id": case_id,
        "query": f"query {case_id}",
        "labels": labels
        if labels is not None
        else [{"doc_id": f"doc-{case_id}", "grade": 3, "rationale": "direct answer"}],
        "review_status": status,
        "reviewed_by": "reviewer",
    }


def test_build_golden_is_seeded_and_preserves_four_grade_judgements():
    rows = [reviewed_case(str(index)) for index in range(4)]
    rows[0]["labels"].append({"doc_id": "negative", "grade": 0, "rationale": "irrelevant"})

    first = build_golden(rows, seed=20260710)
    second = build_golden(list(reversed(rows)), seed=20260710)

    assert first == second
    assert {row["split"] for row in first} == {"dev", "test"}
    assert sum(row["split"] == "dev" for row in first) == 2
    case_zero = next(row for row in first if row["id"] == "0")
    assert "negative" not in case_zero["relevant_doc_ids"]
    assert case_zero["relevance"]["negative"] == 0


@pytest.mark.parametrize(
    "row, message",
    [
        (reviewed_case("pending", status="pending"), "review_status"),
        (reviewed_case("empty", labels=[]), "non-empty"),
        (
            reviewed_case(
                "bad-grade",
                labels=[{"doc_id": "doc", "grade": 4, "rationale": "bad"}],
            ),
            "0 to 3",
        ),
        (
            reviewed_case(
                "no-positive",
                labels=[{"doc_id": "doc", "grade": 0, "rationale": "irrelevant"}],
            ),
            "positive label",
        ),
    ],
)
def test_load_reviewed_labels_rejects_unusable_rows(tmp_path, row, message):
    path = tmp_path / "labels.jsonl"
    path.write_text(json.dumps(row) + "\n", encoding="utf-8")
    with pytest.raises(ValueError, match=message):
        load_reviewed_labels(path)


def valid_manifest(source_hash):
    return {
        "schema_version": 1,
        "corpus_version": "v1",
        "corpus_hash": "a" * 64,
        "chunk_config": {"chunk_version": "v2", "chunk_size": 200, "chunk_overlap": 20},
        "embedding_model": "embedding-v1",
        "retrieval_version": "retrieval-v1",
        "label_scale": {"0": "none", "1": "low", "2": "medium", "3": "high"},
        "split_seed": 42,
        "source_files": {"data/source.txt": source_hash},
    }


def test_validate_manifest_checks_sources_and_runtime_versions(tmp_path):
    source = tmp_path / "data" / "source.txt"
    source.parent.mkdir()
    source.write_text("frozen corpus", encoding="utf-8")
    manifest = valid_manifest(sha256_file(source))

    result = validate_manifest(
        manifest,
        project_root=tmp_path,
        chroma_config={
            "corpus_version": "v1",
            "chunk_version": "v2",
            "chunk_size": 200,
            "chunk_overlap": 20,
            "retrieval": {"version": "retrieval-v1"},
        },
        rag_config={"embedding_model_name": "embedding-v1"},
        expected_corpus_hash="a" * 64,
    )

    assert result["source_file_count"] == 1
    assert result["retrieval_version"] == "retrieval-v1"


@pytest.mark.parametrize("mismatch", ["source", "chunk", "embedding", "retrieval", "corpus"])
def test_validate_manifest_rejects_stale_provenance(tmp_path, mismatch):
    source = tmp_path / "data" / "source.txt"
    source.parent.mkdir()
    source.write_text("frozen corpus", encoding="utf-8")
    manifest = valid_manifest(sha256_file(source))
    chroma = {
        "corpus_version": "v1",
        "chunk_version": "v2",
        "chunk_size": 200,
        "chunk_overlap": 20,
        "retrieval": {"version": "retrieval-v1"},
    }
    rag = {"embedding_model_name": "embedding-v1"}
    expected_hash = "a" * 64
    if mismatch == "source":
        source.write_text("changed", encoding="utf-8")
    elif mismatch == "chunk":
        chroma["chunk_size"] = 201
    elif mismatch == "embedding":
        rag["embedding_model_name"] = "embedding-v2"
    elif mismatch == "retrieval":
        chroma["retrieval"]["version"] = "retrieval-v2"
    else:
        expected_hash = "b" * 64

    with pytest.raises(ManifestValidationError):
        validate_manifest(
            manifest,
            project_root=tmp_path,
            chroma_config=chroma,
            rag_config=rag,
            expected_corpus_hash=expected_hash,
        )


def test_candidate_output_default_is_annotation_file():
    assert DEFAULT_OUTPUT == "evals/annotations/retrieval_candidates_v1.jsonl"
