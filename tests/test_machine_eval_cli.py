import json

import pytest

from scripts.evaluate_agent_quality import build_evaluation_items, main


def _dataset():
    return {
        "case-a": {
            "case_id": "case-a",
            "dataset_version": "agent-quality-v1",
            "split": "dev",
            "category": "knowledge_fault",
            "scene": "knowledge_qa",
            "query": "测试问题",
            "turns": [],
            "context": {},
            "labels": {"risk_level": "L1", "capability_tags": ["retrieval"]},
            "references": [],
            "expected": {
                "outcome": "completed",
                "tools": [],
                "facts": ["测试事实"],
                "forbidden_facts": [],
                "requires_citation": False,
                "requires_artifact": False,
            },
        }
    }


def test_cli_builder_joins_dataset_and_run_results():
    items = build_evaluation_items(
        _dataset(),
        [{"case_id": "case-a", "agent_answer": "测试事实", "status": "completed"}],
        splits=["dev"],
    )

    assert items[0]["expected"]["outcome"] == "completed"
    assert items[0]["agent_answer"] == "测试事实"
    assert items[0]["risk_level"] == "L1"


def test_cli_builder_preserves_experiment_lineage_and_performance():
    items = build_evaluation_items(
        _dataset(),
        [
            {
                "case_id": "case-a",
                "agent_answer": "测试事实",
                "status": "completed",
                "model_metadata": {"model": "candidate-v2", "temperature": 0},
                "latency_ms": 123.4,
                "estimated_cost": 0.012,
                "cost_mode": "estimated",
                "tokens_in": 20,
                "tokens_out": 10,
            }
        ],
        splits=["dev"],
    )

    item = items[0]
    assert item["family_id"] == "case-a"
    assert item["model_metadata"]["model"] == "candidate-v2"
    assert item["latency_ms"] == 123.4
    assert item["estimated_cost"] == 0.012
    assert item["cost_mode"] == "estimated"
    assert item["tokens_in"] == 20
    assert item["tokens_out"] == 10


def test_cli_without_judge_writes_candidate_report(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "dev.jsonl").write_text(
        json.dumps(next(iter(_dataset().values())), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps(
            {"case_id": "case-a", "agent_answer": "测试事实", "status": "completed"},
            ensure_ascii=False,
        )
        + "\n",
        encoding="utf-8",
    )
    report = tmp_path / "report.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_agent_quality.py",
            "--dataset-dir",
            str(dataset_dir),
            "--results",
            str(results),
            "--report",
            str(report),
        ],
    )

    main()

    payload = json.loads(report.read_text(encoding="utf-8"))
    assert payload["summary"]["deterministic_pass_rate"] == 1.0
    assert payload["summary"]["judge_not_run_count"] == 1
    assert payload["production_gate"]["failures"] == ["judge_not_run"]


def test_cli_gate_fails_when_production_gate_is_not_approved(monkeypatch, tmp_path):
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "dev.jsonl").write_text(
        json.dumps(next(iter(_dataset().values())), ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps(
            {"case_id": "case-a", "agent_answer": "测试事实", "status": "completed"}
        )
        + "\n",
        encoding="utf-8",
    )
    monkeypatch.setattr(
        "sys.argv",
        [
            "evaluate_agent_quality.py",
            "--dataset-dir",
            str(dataset_dir),
            "--results",
            str(results),
            "--gate",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
