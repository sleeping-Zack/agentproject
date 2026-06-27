from fastapi.testclient import TestClient

import api.server as server
from api.server import app
from services.approval_store import SQLiteApprovalStore


def test_health_endpoint():
    client = TestClient(app)

    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_tool_manifest_endpoint_exports_allowed_tools():
    client = TestClient(app)

    response = client.get("/tools/manifest")

    assert response.status_code == 200
    manifest = response.json()
    assert manifest["protocol"] == "mcp"
    assert any(tool["name"] == "rag_summarize" for tool in manifest["tools"])


def test_harness_run_creates_pending_approval_for_sensitive_report():
    client = TestClient(app)

    response = client.post(
        "/harness/run",
        headers={"X-API-Key": "dev-api-key", "X-Tenant-ID": "api-test"},
        json={
            "message": "生成本月使用记录报告",
            "session_id": "api-harness-test",
            "scene": "report",
            "user_role": "user",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "pending_approval"
    assert payload["approval_id"]

    approval = client.get(
        f"/approvals/{payload['approval_id']}",
        headers={"X-API-Key": "dev-api-key", "X-Tenant-ID": "api-test"},
    )
    assert approval.status_code == 200
    assert approval.json()["status"] == "pending"

    approved = client.post(
        f"/approvals/{payload['approval_id']}/approve",
        headers={
            "X-API-Key": "dev-api-key",
            "X-Tenant-ID": "api-test",
            "X-User-Role": "operator",
        },
        json={"decided_by": "tester"},
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"


def test_approval_api_requires_operator_and_matching_tenant(monkeypatch, tmp_path):
    approval_store = SQLiteApprovalStore(str(tmp_path / "approvals.db"))
    approval = approval_store.create_pending(
        request_id="req-tenant",
        tenant_id="tenant-a",
        user_role="user",
        tool_name="fetch_external_data",
        args={"user_id": "1001", "month": "2025-09"},
        reason="tool requires approval",
    )
    monkeypatch.setattr(server, "approval_store", approval_store)
    client = TestClient(app)

    user_approve = client.post(
        f"/approvals/{approval.approval_id}/approve",
        headers={"X-API-Key": "dev-api-key", "X-Tenant-ID": "tenant-a"},
        json={"decided_by": "body-admin"},
    )
    assert user_approve.status_code == 403

    cross_tenant = client.get(
        f"/approvals/{approval.approval_id}",
        headers={"X-API-Key": "dev-api-key", "X-Tenant-ID": "tenant-b"},
    )
    assert cross_tenant.status_code == 404

    operator_approve = client.post(
        f"/approvals/{approval.approval_id}/approve",
        headers={
            "X-API-Key": "dev-api-key",
            "X-Tenant-ID": "tenant-a",
            "X-User-Role": "operator",
        },
        json={"decided_by": "body-admin"},
    )
    assert operator_approve.status_code == 200
    assert operator_approve.json()["decided_by"] == "operator:tenant-a"
