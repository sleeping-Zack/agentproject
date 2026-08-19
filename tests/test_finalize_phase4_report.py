import json

import pytest

from human_eval.service import HumanEvalService
from scripts.finalize_phase4_report import _validate_human_export, main
from services.human_eval_store import SQLiteHumanEvalStore


def _scores():
    return {
        "task_completion": 3,
        "factual_correctness": 3,
        "tool_use": 3,
        "instruction_following": 3,
        "groundedness": 3,
        "safety": 3,
        "response_quality": 3,
    }


def _submission():
    return {
        "valid": True,
        "invalid_reason": None,
        "scores": _scores(),
        "vetoes": [],
        "rationales": {},
        "confidence": "high",
        "duration_seconds": 10,
    }


def _write_inputs(tmp_path):
    case = {
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
    result = {
        "case_id": "case-a",
        "agent_answer": "测试事实",
        "status": "completed",
        "trace": [],
        "evidence": [],
    }
    dataset_dir = tmp_path / "dataset"
    dataset_dir.mkdir()
    (dataset_dir / "dev.jsonl").write_text(
        json.dumps(case, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    results = tmp_path / "results.jsonl"
    results.write_text(
        json.dumps(result, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    return dataset_dir, results, case, result


def _create_batch(db_path, case, result, *, close):
    service = HumanEvalService(SQLiteHumanEvalStore(db_path))
    progress = service.create_batch(
        tenant_id="tenant-a",
        created_by="operator-a",
        name="candidate-human-eval",
        dataset_version="agent-quality-v1",
        items=[
            {
                "case_id": case["case_id"],
                "query": case["query"],
                "turns": [],
                "scene": case["scene"],
                "risk_level": "L1",
                "capability_tags": ["retrieval"],
                "context": {},
                "agent_answer": result["agent_answer"],
                "trace": [],
                "evidence": [],
                "references": [],
                "policy_context": {},
                "oracle": {"expected": case["expected"]},
            }
        ],
        reviewer_ids=["reviewer-a", "reviewer-b"],
        qc_rate=0,
    )
    batch_id = progress["batch_id"]
    if close:
        for reviewer in ("reviewer-a", "reviewer-b"):
            task = service.next_task(
                batch_id=batch_id,
                tenant_id="tenant-a",
                reviewer_id=reviewer,
            )
            service.submit(
                assignment_id=task["assignment_id"],
                tenant_id="tenant-a",
                reviewer_id=reviewer,
                submission=_submission(),
            )
        service.close_batch(
            batch_id=batch_id,
            tenant_id="tenant-a",
            actor_id="operator-a",
        )
    return batch_id


def test_closed_batch_generates_diagnostic_report_without_judge(monkeypatch, tmp_path):
    dataset_dir, results, case, result = _write_inputs(tmp_path)
    db_path = tmp_path / "human-eval.db"
    batch_id = _create_batch(db_path, case, result, close=True)
    report_path = tmp_path / "reports" / "candidate.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "finalize_phase4_report.py",
            "--dataset-dir",
            str(dataset_dir),
            "--results",
            str(results),
            "--db",
            str(db_path),
            "--batch-id",
            batch_id,
            "--tenant",
            "tenant-a",
            "--variant",
            "candidate",
            "--report",
            str(report_path),
        ],
    )

    main()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    export = json.loads(
        (report_path.parent / "candidate.human-export.json").read_text(encoding="utf-8")
    )
    assert report["run_metadata"]["variant"] == "candidate"
    assert report["run_metadata"]["evaluation_mode"] == "diagnostic"
    assert report["human_alignment"]["human_batch_id"] == batch_id
    assert report["production_gate"] == {
        "status": "blocked",
        "passed": None,
        "failures": ["judge_not_run"],
    }
    assert export["batch_status"] == "closed"
    assert export["pending_final_count"] == 0


def test_active_batch_is_rejected_before_any_report_is_written(monkeypatch, tmp_path):
    dataset_dir, results, case, result = _write_inputs(tmp_path)
    db_path = tmp_path / "human-eval.db"
    batch_id = _create_batch(db_path, case, result, close=False)
    report_path = tmp_path / "candidate.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "finalize_phase4_report.py",
            "--dataset-dir",
            str(dataset_dir),
            "--results",
            str(results),
            "--db",
            str(db_path),
            "--batch-id",
            batch_id,
            "--tenant",
            "tenant-a",
            "--variant",
            "candidate",
            "--report",
            str(report_path),
        ],
    )

    with pytest.raises(ValueError, match="must be closed"):
        main()

    assert not report_path.exists()
    assert not (tmp_path / "candidate.human-export.json").exists()


def test_human_export_and_run_result_case_sets_must_match_exactly():
    human_export = {
        "batch_status": "closed",
        "pending_final_count": 0,
        "records": [
            {
                "case_id": "case-a",
                "payload": {"query": "问题 A", "agent_answer": "回答 A"},
            }
        ],
    }
    items = [
        {"case_id": "case-a", "query": "问题 A", "agent_answer": "回答 A"},
        {"case_id": "case-b", "query": "问题 B", "agent_answer": "回答 B"},
    ]

    with pytest.raises(ValueError, match="case sets must match exactly"):
        _validate_human_export(human_export, items)


def test_judge_requires_explicit_pinned_identity(monkeypatch, tmp_path):
    report_path = tmp_path / "candidate.json"
    monkeypatch.setattr(
        "sys.argv",
        [
            "finalize_phase4_report.py",
            "--results",
            str(tmp_path / "missing-results.jsonl"),
            "--batch-id",
            "batch-a",
            "--tenant",
            "tenant-a",
            "--variant",
            "candidate",
            "--report",
            str(report_path),
            "--judge",
        ],
    )

    with pytest.raises(ValueError, match="explicit pinned --judge-id"):
        main()
