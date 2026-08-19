from fastapi.testclient import TestClient

from api.server import app


def _headers(role="operator"):
    return {
        "X-Tenant-ID": "machine-eval-api",
        "X-User-Role": role,
        "X-Principal-ID": "eval-operator" if role == "operator" else "reviewer-a",
    }


def _item():
    return {
        "case_id": "case-api-1",
        "agent_answer": "包含测试事实",
        "status": "completed",
        "expected": {
            "outcome": "completed",
            "tools": [],
            "facts": ["测试事实"],
            "forbidden_facts": [],
            "requires_citation": False,
            "requires_artifact": False,
        },
    }


def test_machine_eval_api_requires_operator_and_returns_candidate_result(monkeypatch):
    monkeypatch.setenv("AGENT_AUTH_MODE", "legacy_headers")
    client = TestClient(app)

    forbidden = client.post(
        "/machine-eval/runs",
        headers=_headers("user"),
        json={"items": [_item()], "judge_enabled": False},
    )
    response = client.post(
        "/machine-eval/runs",
        headers=_headers(),
        json={"items": [_item()], "judge_enabled": False},
    )

    assert forbidden.status_code == 403
    assert response.status_code == 200
    payload = response.json()
    assert payload["summary"]["deterministic_pass_rate"] == 1.0
    assert payload["production_gate"] == {
        "status": "blocked",
        "passed": None,
        "failures": ["judge_not_run"],
    }
    assert payload["run_metadata"]["tenant_id"] == "machine-eval-api"
