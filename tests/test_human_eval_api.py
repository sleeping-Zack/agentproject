from fastapi.testclient import TestClient

import api.server as server
from api.server import app
from human_eval.service import HumanEvalService
from services.human_eval_store import SQLiteHumanEvalStore


def _headers(principal_id, role="user", tenant_id="human-eval-api"):
    return {
        "X-Tenant-ID": tenant_id,
        "X-User-Role": role,
        "X-Principal-ID": principal_id,
    }


def _annotation():
    return {
        "valid": True,
        "invalid_reason": None,
        "scores": {
            "task_completion": 3,
            "factual_correctness": 3,
            "tool_use": 3,
            "instruction_following": 3,
            "groundedness": 3,
            "safety": 3,
            "response_quality": 3,
        },
        "vetoes": [],
        "rationales": {},
        "confidence": "high",
        "duration_seconds": 20,
    }


def test_human_eval_api_enforces_roles_identity_and_blinding(monkeypatch, tmp_path):
    monkeypatch.setenv("AGENT_AUTH_MODE", "legacy_headers")
    service = HumanEvalService(SQLiteHumanEvalStore(tmp_path / "human-eval.db"))
    monkeypatch.setattr(server, "human_eval_service", service)
    client = TestClient(app)
    request = {
        "name": "api-pilot",
        "dataset_version": "agent-quality-v1",
        "reviewer_ids": ["reviewer-a", "reviewer-b"],
        "qc_rate": 0,
        "items": [
            {
                "case_id": "private-case-id",
                "query": "如何处理当前故障？",
                "context": {"model_name": "hidden-model", "device": "robot"},
                "agent_answer": "请先检查设备状态。",
                "oracle": {"expected": "先检查状态"},
            }
        ],
    }

    forbidden = client.post(
        "/human-eval/batches",
        headers=_headers("reviewer-a"),
        json=request,
    )
    assert forbidden.status_code == 403

    created = client.post(
        "/human-eval/batches",
        headers=_headers("operator-a", role="operator"),
        json=request,
    )
    assert created.status_code == 200
    batch_id = created.json()["batch_id"]

    claimed = client.get(
        f"/human-eval/batches/{batch_id}/tasks/next",
        headers=_headers("reviewer-a"),
    )
    assert claimed.status_code == 200
    task = claimed.json()["task"]
    assert "case_id" not in task
    assert "oracle" not in str(task)
    assert "hidden-model" not in str(task)

    wrong_identity = client.post(
        f"/human-eval/tasks/{task['assignment_id']}/submit",
        headers=_headers("reviewer-b"),
        json=_annotation(),
    )
    assert wrong_identity.status_code == 404

    submitted = client.post(
        f"/human-eval/tasks/{task['assignment_id']}/submit",
        headers=_headers("reviewer-a"),
        json=_annotation(),
    )
    assert submitted.status_code == 200

    other_task = client.get(
        f"/human-eval/batches/{batch_id}/tasks/next",
        headers=_headers("reviewer-b"),
    ).json()["task"]
    other_submitted = client.post(
        f"/human-eval/tasks/{other_task['assignment_id']}/submit",
        headers=_headers("reviewer-b"),
        json=_annotation(),
    )
    assert other_submitted.status_code == 200

    user_report = client.get(
        f"/human-eval/batches/{batch_id}/report",
        headers=_headers("reviewer-a"),
    )
    assert user_report.status_code == 403

    report = client.get(
        f"/human-eval/batches/{batch_id}/report",
        headers=_headers("operator-a", role="operator"),
    )
    assert report.status_code == 200
    assert report.json()["pair_count"] == 1

    closed = client.post(
        f"/human-eval/batches/{batch_id}/close",
        headers=_headers("operator-a", role="operator"),
    )
    assert closed.status_code == 200
    assert closed.json()["status"] == "closed"

    audit = client.get(
        f"/human-eval/batches/{batch_id}/audit",
        headers=_headers("operator-a", role="operator"),
    )
    assert audit.status_code == 200
    assert audit.json()[-1]["event_type"] == "batch_closed"
