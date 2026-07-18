from __future__ import annotations

import json

import pytest

from scripts.build_rerank_dev_data import (
    build_dev_golden,
    mine_hard_negatives,
    validate_dev_annotations,
)
from scripts.generate_retrieval_golden import (
    exclude_locked_queries,
    load_cases,
    validate_locked_query_aliases,
)
from scripts.prepare_chunk_experiments import prepare_experiments
from scripts.tune_rerank_fusion import evaluate_weight_grid, require_dev_cases


def test_candidate_source_generates_stable_ids_and_excludes_locked_queries(tmp_path):
    source = tmp_path / "source.jsonl"
    source.write_text(
        "\n".join(
            [
                json.dumps({"query": "  New   Query "}),
                json.dumps({"query": "Locked Query"}),
            ]
        ),
        encoding="utf-8",
    )
    cases = load_cases(source)

    assert cases[0]["id"].startswith("rag-")
    assert load_cases(source)[0]["id"] == cases[0]["id"]
    filtered = exclude_locked_queries(cases, [{"query": " locked   query "}])
    assert [case["query"] for case in filtered] == ["New   Query"]


def test_locked_query_alias_must_reference_an_audited_test_case():
    with pytest.raises(ValueError, match="unknown locked test id"):
        validate_locked_query_aliases(
            [
                {
                    "query": "semantic rewrite",
                    "locked_test_id": "missing",
                    "reason": "same intent",
                }
            ],
            [{"id": "test-1", "query": "locked query"}],
        )


def _candidate_row(query="dev query"):
    return {
        "case_id": "dev-1",
        "query": query,
        "split": "dev",
        "candidates": [
            {
                "doc_id": "positive",
                "chunk_text": "direct answer",
                "source": "guide.txt",
                "hybrid_rank": 3,
                "rerank_rank": 2,
            },
            {
                "doc_id": "hard-negative",
                "chunk_text": "looks relevant but is wrong",
                "source": "faq.txt",
                "hybrid_rank": 1,
                "rerank_rank": 1,
            },
            {
                "doc_id": "weak-negative",
                "chunk_text": "unrelated",
                "source": "faq.txt",
                "hybrid_rank": 10,
                "rerank_rank": 8,
            },
        ],
    }


def _labelled_case(query="dev query"):
    return {
        "case_id": "dev-1",
        "query": query,
        "split": "dev",
        "review_status": "reviewed",
        "reviewed_by": "human-reviewer",
        "labels": [
            {"doc_id": "positive", "grade": 3, "rationale": "answers query"},
            {"doc_id": "hard-negative", "grade": 0, "rationale": "wrong product"},
            {"doc_id": "weak-negative", "grade": 1, "rationale": "related only"},
        ],
    }


def test_dev_builder_uses_only_explicit_grade_zero_as_hard_negative():
    labelled = [_labelled_case()]
    candidate_rows = [_candidate_row()]
    indexed = validate_dev_annotations(
        labelled,
        candidate_rows,
        [{"query": "locked test query"}],
    )

    golden = build_dev_golden(labelled)
    hard_negatives = mine_hard_negatives(labelled, indexed, max_negatives=5)

    assert golden[0]["split"] == "dev"
    assert golden[0]["relevance"]["hard-negative"] == 0
    assert [item["doc_id"] for item in hard_negatives[0]["positives"]] == ["positive"]
    assert [item["doc_id"] for item in hard_negatives[0]["hard_negatives"]] == [
        "hard-negative"
    ]


def test_dev_builder_rejects_locked_test_query():
    with pytest.raises(ValueError, match="overlaps locked test"):
        validate_dev_annotations(
            [_labelled_case("same query")],
            [_candidate_row("same query")],
            [{"query": " SAME   QUERY "}],
        )


def _report_case():
    return {
        "id": "dev-1",
        "retrieved": [
            {
                "doc_id": "partial",
                "hybrid_rank": 1,
                "rerank_rank": 2,
                "rerank_evaluated": True,
            },
            {
                "doc_id": "direct",
                "hybrid_rank": 2,
                "rerank_rank": 1,
                "rerank_evaluated": True,
            },
        ],
    }


def test_fusion_tuning_selects_improvement_without_recall_regression():
    golden = [
        {
            "id": "dev-1",
            "query": "query",
            "split": "dev",
            "relevant_doc_ids": ["partial", "direct"],
            "relevance": {"partial": 1, "direct": 3},
        }
    ]

    result = evaluate_weight_grid(
        golden,
        [_report_case()],
        model_weights=[0.0, 0.75],
        k=2,
        fusion_k=0,
    )

    assert result["recommended"]["model_weight"] == 0.75
    assert result["recommended"]["recall_regressed_case_ids"] == []
    assert result["recommended"]["ndcg_at_k"] > result["baseline"]["ndcg_at_k"]


def test_fusion_tuning_refuses_test_split():
    with pytest.raises(ValueError, match="dev cases only"):
        require_dev_cases([{"id": "test-1", "split": "test"}], min_cases=1)


def test_chunk_experiments_use_isolated_storage_and_do_not_mutate_base(tmp_path):
    base_config = {
        "collection_name": "agent",
        "chunk_version": "v2",
        "chunk_size": 200,
        "chunk_overlap": 20,
        "persist_directory": "storage/chroma",
        "md5_hex_store": "storage/md5.text",
        "retrieval": {"bm25_index_path": "storage/bm25.pkl", "enable_reranker": True},
    }
    base_manifest = {"chunk_config": {"chunk_version": "v2"}}

    plan = prepare_experiments(
        base_config,
        base_manifest,
        variants=[(200, 20), (350, 50)],
        output_dir=tmp_path,
    )

    experiments = plan["experiments"]
    assert len({item["collection_name"] for item in experiments}) == 2
    assert len({item["persist_directory"] for item in experiments}) == 2
    assert all("storage/experiments/" in item["persist_directory"] for item in experiments)
    assert base_config["persist_directory"] == "storage/chroma"
    generated = (tmp_path / "chunk-350-50" / "chroma.yml").read_text(encoding="utf-8")
    assert "enable_reranker: false" in generated
    assert "-m rag.vector_store" in experiments[0]["commands"]["index"]
    assert "-m scripts.validate_retrieval_manifest" in experiments[0]["commands"]["validate"]
    assert "-m scripts.generate_retrieval_golden" in experiments[0]["commands"][
        "generate_dev_candidates"
    ]
