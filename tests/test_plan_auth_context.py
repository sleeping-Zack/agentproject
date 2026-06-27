from types import SimpleNamespace

from fastapi.testclient import TestClient

import api.server as server


def test_plan_endpoint_uses_auth_context_and_header_tenant(monkeypatch):
    calls = []

    def fake_run_plan(query, request_id, tenant_id):
        calls.append({"query": query, "request_id": request_id, "tenant_id": tenant_id})
        return SimpleNamespace(plan=[], results=[], answer="planned")

    monkeypatch.setattr(server.agent, "run_plan", fake_run_plan)
    client = TestClient(server.app)

    response = client.post(
        "/plan",
        headers={"X-API-Key": "dev-api-key", "X-Tenant-ID": "tenant-header"},
        json={"message": "帮我规划一下", "tenant_id": "tenant-body"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "planned"
    assert calls[0]["tenant_id"] == "tenant-header"


def test_plan_endpoint_rejects_invalid_auth_role_header():
    client = TestClient(server.app)

    response = client.post(
        "/plan",
        headers={
            "X-API-Key": "dev-api-key",
            "X-Tenant-ID": "tenant-a",
            "X-User-Role": "superadmin",
        },
        json={"message": "帮我规划一下"},
    )

    assert response.status_code == 401
