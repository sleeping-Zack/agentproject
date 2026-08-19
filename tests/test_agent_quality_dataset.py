import json
from pathlib import Path

from scripts.build_agent_quality_dataset import (
    CATEGORY_TARGETS,
    _sha256_file,
    build_dataset,
)
from scripts.validate_agent_quality_dataset import validate_dataset


CHECKED_IN_DATASET = Path("evals/agent_quality/v1")


def _load_jsonl(path: Path):
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def test_checked_in_agent_quality_dataset_passes_full_validation():
    result = validate_dataset(CHECKED_IN_DATASET.resolve())

    assert result["passed"] is True, result["errors"]
    assert result["errors"] == []
    assert result["warnings"] == []
    assert result["coverage"]["case_count"] == 175
    assert result["coverage"]["family_count"] == 175
    assert result["coverage"]["by_category"] == CATEGORY_TARGETS


def test_agent_quality_dataset_build_is_deterministic(tmp_path):
    first_dir = tmp_path / "first"
    second_dir = tmp_path / "second"
    first_manifest = build_dataset(first_dir)
    second_manifest = build_dataset(second_dir)

    assert first_manifest["dataset_sha256"] == second_manifest["dataset_sha256"]
    for file_name in ("dev.jsonl", "test.jsonl", "regression.jsonl", "coverage_report.json"):
        assert (first_dir / file_name).read_bytes() == (second_dir / file_name).read_bytes()
    assert validate_dataset(first_dir)["passed"] is True
    assert validate_dataset(second_dir)["passed"] is True


def test_text_file_hash_is_stable_across_platform_line_endings(tmp_path):
    lf_path = tmp_path / "lf.jsonl"
    crlf_path = tmp_path / "crlf.jsonl"
    lf_path.write_bytes(b'{"case_id":"case-1"}\n')
    crlf_path.write_bytes(b'{"case_id":"case-1"}\r\n')

    assert _sha256_file(lf_path) == _sha256_file(crlf_path)


def test_dataset_keeps_candidate_status_until_independent_human_review():
    manifest = json.loads((CHECKED_IN_DATASET / "manifest.json").read_text(encoding="utf-8"))
    cases = [
        case
        for split in ("dev", "test", "regression")
        for case in _load_jsonl(CHECKED_IN_DATASET / f"{split}.jsonl")
    ]

    assert manifest["status"] == "candidate_pending_human_review"
    assert manifest["review_policy"]["production_golden_allowed"] is False
    assert {case["provenance"]["review_status"] for case in cases} == {
        "pending_second_reviewer"
    }


def test_validator_rejects_frozen_test_query_in_dev(tmp_path):
    dataset_dir = tmp_path / "dataset"
    build_dataset(dataset_dir)
    dev_path = dataset_dir / "dev.jsonl"
    rows = _load_jsonl(dev_path)
    rows[0]["query"] = "扫地机器人吸力下降怎么处理"
    dev_path.write_text(
        "\n".join(json.dumps(row, ensure_ascii=False, sort_keys=True) for row in rows) + "\n",
        encoding="utf-8",
    )

    result = validate_dataset(dataset_dir)

    assert result["passed"] is False
    assert any("frozen retrieval test query leaked" in error for error in result["errors"])


def test_case_schema_declares_all_dataset_contract_fields():
    schema = json.loads(Path("evals/agent_quality/schema_v1.json").read_text(encoding="utf-8"))
    required = set(schema["required"])

    assert schema["$id"] == "agent-quality-case-v1"
    assert {
        "case_id",
        "family_id",
        "split",
        "category",
        "scene",
        "query",
        "labels",
        "expected",
        "references",
        "provenance",
    }.issubset(required)
