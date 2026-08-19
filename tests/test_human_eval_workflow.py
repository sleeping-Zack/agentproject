import sqlite3

import pytest

from human_eval.service import HumanEvalService
from scripts.prepare_human_eval_batch import build_items
from services.human_eval_store import SQLiteHumanEvalStore


def _scores(default=3):
    return {
        "task_completion": default,
        "factual_correctness": default,
        "tool_use": default,
        "instruction_following": default,
        "groundedness": default,
        "safety": default,
        "response_quality": default,
    }


def _good_submission():
    return {
        "valid": True,
        "invalid_reason": None,
        "scores": _scores(),
        "vetoes": [],
        "rationales": {},
        "confidence": "high",
        "duration_seconds": 30,
    }


def _submission_for(query, reviewer):
    if query == "case-2" and reviewer == "reviewer-b":
        scores = _scores()
        scores["task_completion"] = 1
        return {
            **_good_submission(),
            "scores": scores,
            "rationales": {"task_completion": "核心任务只完成了一部分"},
        }
    if query == "case-3":
        return {
            **_good_submission(),
            "valid": False,
            "invalid_reason": "missing_reference",
            "scores": _scores(None),
            "rationales": {"validity": "缺少验证答案所需的权威资料"},
        }
    if query == "case-4" and reviewer == "reviewer-a":
        scores = _scores()
        scores["safety"] = 0
        return {
            **_good_submission(),
            "scores": scores,
            "vetoes": ["unsafe_action"],
            "rationales": {
                "safety": "回答包含可能损坏设备的危险步骤",
                "veto:unsafe_action": "危险步骤可直接执行且没有阻止说明",
            },
        }
    return _good_submission()


def _items():
    return [
        {
            "case_id": f"source-{index}",
            "query": f"case-{index}",
            "scene": "tool_execution",
            "risk_level": "L2",
            "capability_tags": ["tool_selection"],
            "context": {
                "model_name": "must-not-leak",
                "nested": {"provider": "must-not-leak", "allowed": "visible"},
            },
            "agent_answer": f"answer-{index}",
            "trace": [{"tool": "get_weather", "result": "ok"}],
            "oracle": {"expected_outcome": "completed", "model": "operator-only"},
        }
        for index in range(1, 5)
    ]


def test_double_blind_qc_adjudication_reporting_and_export(tmp_path):
    db_path = tmp_path / "human-eval.db"
    store = SQLiteHumanEvalStore(db_path)
    service = HumanEvalService(store)
    progress = service.create_batch(
        tenant_id="tenant-a",
        created_by="operator-a",
        name="pilot-v1",
        dataset_version="agent-quality-v1",
        items=_items(),
        reviewer_ids=["reviewer-a", "reviewer-b"],
        qc_rate=1.0,
        seed=7,
    )
    batch_id = progress["batch_id"]

    assert progress["assignment_count"] == 8
    assert progress["qc_status"] == {"waiting": 4}
    assert service.next_task(
        batch_id=batch_id, tenant_id="tenant-a", reviewer_id="not-assigned"
    ) is None
    with pytest.raises(KeyError):
        service.next_task(
            batch_id=batch_id, tenant_id="tenant-b", reviewer_id="reviewer-a"
        )

    assignment_ids = {}
    cross_tenant_checked = False
    for reviewer in ("reviewer-a", "reviewer-b"):
        while True:
            task = service.next_task(
                batch_id=batch_id,
                tenant_id="tenant-a",
                reviewer_id=reviewer,
            )
            if task is None:
                break
            query = task["payload"]["query"]
            assignment_ids[(query, reviewer)] = task["assignment_id"]
            assert "case_id" not in task
            assert "oracle" not in task
            assert "model_name" not in task["payload"]["context"]
            assert "provider" not in task["payload"]["context"]["nested"]
            assert task["payload"]["context"]["nested"]["allowed"] == "visible"
            assert "pass_conditions" not in task["rubric"]
            if not cross_tenant_checked:
                with pytest.raises(KeyError):
                    service.submit(
                        assignment_id=task["assignment_id"],
                        tenant_id="tenant-b",
                        reviewer_id=reviewer,
                        submission=_submission_for(query, reviewer),
                    )
                cross_tenant_checked = True
            service.submit(
                assignment_id=task["assignment_id"],
                tenant_id="tenant-a",
                reviewer_id=reviewer,
                submission=_submission_for(query, reviewer),
            )

    progress = service.progress(batch_id=batch_id, tenant_id="tenant-a")
    assert progress["submitted_assignments"] == 8
    assert progress["completed_items"] == 4
    assert progress["disagreement_count"] == 2
    assert progress["pending_adjudication_count"] == 2
    assert progress["qc_status"] == {"pending": 4}
    with pytest.raises(ValueError, match="qc_incomplete"):
        service.close_batch(
            batch_id=batch_id, tenant_id="tenant-a", actor_id="operator-a"
        )

    qc_queue = service.qc_queue(batch_id=batch_id, tenant_id="tenant-a")
    assert len(qc_queue) == 4
    assert all(len(item["assignments"]) == 2 for item in qc_queue)
    assert all("reviewer_id" not in str(item) for item in qc_queue)
    assert qc_queue[0]["oracle_context"]["expected_outcome"] == "completed"
    case_one_qc = next(item for item in qc_queue if item["payload"]["query"] == "case-1")
    returned_assignment_id = assignment_ids[("case-1", "reviewer-a")]
    with pytest.raises(ValueError, match="QC reviewer must be independent"):
        service.review_qc(
            batch_id=batch_id,
            tenant_id="tenant-a",
            item_id=case_one_qc["item_id"],
            reviewer_id="reviewer-a",
            decision="accepted",
            note="不能质检自己的评分",
        )
    service.review_qc(
        batch_id=batch_id,
        tenant_id="tenant-a",
        item_id=case_one_qc["item_id"],
        reviewer_id="quality-owner",
        decision="returned",
        note="任务完成度依据不够具体，请复核",
        returned_assignments=[returned_assignment_id],
    )

    returned_task = service.next_task(
        batch_id=batch_id, tenant_id="tenant-a", reviewer_id="reviewer-a"
    )
    assert returned_task["assignment_id"] == returned_assignment_id
    assert returned_task["returned_for_revision"] is True
    assert returned_task["revision"] == 1
    service.submit(
        assignment_id=returned_assignment_id,
        tenant_id="tenant-a",
        reviewer_id="reviewer-a",
        submission=_good_submission(),
    )

    for item in service.qc_queue(batch_id=batch_id, tenant_id="tenant-a"):
        service.review_qc(
            batch_id=batch_id,
            tenant_id="tenant-a",
            item_id=item["item_id"],
            reviewer_id="quality-owner",
            decision="accepted",
            note="评分依据、适用性与 Rubric 一致",
        )

    disagreements = service.disagreements(batch_id=batch_id, tenant_id="tenant-a")
    assert {item["payload"]["query"] for item in disagreements} == {"case-2", "case-4"}
    assert all("oracle_context" in item for item in disagreements)
    assert all("reviewer_id" not in str(item) for item in disagreements)
    with pytest.raises(ValueError, match="adjudicator must be independent"):
        service.adjudicate(
            batch_id=batch_id,
            tenant_id="tenant-a",
            item_id=disagreements[0]["item_id"],
            adjudicator_id="reviewer-a",
            submission=_good_submission(),
        )
    for item in disagreements:
        service.adjudicate(
            batch_id=batch_id,
            tenant_id="tenant-a",
            item_id=item["item_id"],
            adjudicator_id="adjudicator-a",
            submission=_good_submission(),
        )

    report = service.report(batch_id=batch_id, tenant_id="tenant-a")
    assert report["pair_count"] == 4
    assert report["resolved_case_count"] == 4
    assert report["pending_final_count"] == 0
    assert report["invalid_case_rate"] == 0.25
    assert report["human_case_pass_rate"] == 1.0
    assert report["adjudication_rate"] == 0.5
    assert report["quality_gate"]["status"] == "insufficient_sample"
    assert set(report["reviewer_metrics"]) != {"reviewer-a", "reviewer-b"}

    exported = service.export(batch_id=batch_id, tenant_id="tenant-a")
    assert exported["record_count"] == 4
    assert exported["pending_final_count"] == 0
    assert all(record["final"] is not None for record in exported["records"])
    assert all("oracle_context" not in record for record in exported["records"])
    assert "reviewer_id" not in str(exported)
    invalid_record = next(
        record for record in exported["records"] if record["payload"]["query"] == "case-3"
    )
    assert invalid_record["final"]["source"] == "dual_consensus"

    closed = service.close_batch(
        batch_id=batch_id, tenant_id="tenant-a", actor_id="operator-a"
    )
    assert closed["status"] == "closed"
    with pytest.raises(ValueError, match="not active"):
        service.next_task(
            batch_id=batch_id, tenant_id="tenant-a", reviewer_id="reviewer-a"
        )

    events = service.audit_events(batch_id=batch_id, tenant_id="tenant-a")
    event_types = {event["event_type"] for event in events}
    assert {"batch_created", "qc_returned", "item_adjudicated", "batch_closed"} <= event_types
    with sqlite3.connect(db_path) as connection:
        version_count, current_count = connection.execute(
            "SELECT COUNT(*), SUM(is_current) FROM human_eval_annotations "
            "WHERE assignment_id = ?",
            (returned_assignment_id,),
        ).fetchone()
    assert version_count == 2
    assert current_count == 1


def test_batch_requires_two_unique_reviewers(tmp_path):
    service = HumanEvalService(SQLiteHumanEvalStore(tmp_path / "human-eval.db"))

    with pytest.raises(ValueError, match="two unique reviewers"):
        service.create_batch(
            tenant_id="tenant-a",
            created_by="operator-a",
            name="invalid-batch",
            dataset_version="agent-quality-v1",
            items=_items()[:1],
            reviewer_ids=["same-reviewer", "same-reviewer"],
        )


def test_run_results_are_joined_with_dataset_and_metadata_stays_in_oracle():
    dataset = {
        "case-a": {
            "case_id": "case-a",
            "split": "dev",
            "query": "测试问题",
            "turns": [],
            "scene": "knowledge_qa",
            "context": {"document": "visible"},
            "labels": {"risk_level": "L1", "capability_tags": ["retrieval"]},
            "references": [{"type": "local_file", "uri": "data/example.txt"}],
            "expected": {"outcome": "completed"},
            "provenance": {"review_status": "pending_second_reviewer"},
        }
    }
    results = [
        {
            "case_id": "case-a",
            "agent_answer": "测试答案",
            "trace": [],
            "evidence": [],
            "approval_records": [{"approval_id": "approval-1", "status": "approved"}],
            "planner_steps": [{"id": "t1", "kind": "rag_qa"}],
            "tool_calls": [{"tool_name": "rag_summarize", "status": "success"}],
            "model_metadata": {"model_name": "private-model", "variant": "B"},
        }
    ]

    items = build_items(dataset, results, splits=["dev"], sample_size=None, seed=3)

    assert items[0]["agent_answer"] == "测试答案"
    assert items[0]["approval_records"][0]["approval_id"] == "approval-1"
    assert items[0]["planner_steps"][0]["id"] == "t1"
    assert items[0]["tool_calls"][0]["tool_name"] == "rag_summarize"
    assert "model_metadata" not in items[0]
    assert items[0]["oracle"]["run_metadata"]["model_name"] == "private-model"
