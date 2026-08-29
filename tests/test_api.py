from types import SimpleNamespace

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
    expected = {
        "rag_summarize",
        "lookup_error_code",
        "get_product_specs",
        "create_support_ticket",
    }
    assert expected <= {tool["name"] for tool in manifest["tools"]}

    mcp_response = client.post(
        "/mcp",
        headers={"X-API-Key": "dev-api-key", "X-Tenant-ID": "api-test"},
        json={"jsonrpc": "2.0", "id": 1, "method": "tools/list", "params": {}},
    )
    assert mcp_response.status_code == 200
    assert expected <= {
        tool["name"] for tool in mcp_response.json()["result"]["tools"]
    }


def test_harness_run_returns_citation_evidence_contract(monkeypatch):
    evidence = [
        {
            "id": "doc:4",
            "source": "故障排除.txt",
            "excerpt": "清理风道杂物。",
            "chunk_index": 4,
        }
    ]

    class Runner:
        def run(self, task):
            return SimpleNamespace(
                request_id=task.request_id,
                state=SimpleNamespace(status="completed", error=None, budget=None),
                answer="清理风道杂物 [doc:4]。",
                approval_id=None,
                artifacts=[],
                verifier=None,
                evidence=evidence,
            )

    monkeypatch.setattr(server, "harness_runner", Runner())
    monkeypatch.setattr(server.trace_recorder, "export_trace", lambda _request_id: {})
    monkeypatch.setattr(server.store, "save_trace", lambda *_args, **_kwargs: None)
    client = TestClient(app)

    response = client.post(
        "/harness/run",
        headers={"X-API-Key": "dev-api-key", "X-Tenant-ID": "api-test"},
        json={"message": "清洁效果下降", "session_id": "citation-contract"},
    )

    assert response.status_code == 200
    assert response.json()["evidence"] == evidence

    chat_response = client.post(
        "/chat",
        headers={"X-API-Key": "dev-api-key", "X-Tenant-ID": "api-test"},
        json={"message": "清洁效果下降", "session_id": "citation-contract"},
    )

    assert chat_response.status_code == 200
    assert chat_response.json()["evidence"] == evidence


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
    assert payload["error"] is None
    assert payload["budget"]["remaining_tokens"] <= payload["budget"]["max_tokens"]

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


def test_operator_can_list_tenant_pending_approvals(monkeypatch, tmp_path):
    approval_store = SQLiteApprovalStore(str(tmp_path / "approvals.db"))
    expected = approval_store.create_pending(
        request_id="req-list",
        tenant_id="tenant-a",
        principal_id="user-1",
        user_role="user",
        tool_name="fetch_external_data",
        args={"user_id": "1002", "month": "2025-09"},
        reason="cross-user report",
    )
    approval_store.create_pending(
        request_id="req-other-tenant",
        tenant_id="tenant-b",
        principal_id="user-2",
        user_role="user",
        tool_name="fetch_external_data",
        args={"user_id": "1003", "month": "2025-09"},
        reason="cross-user report",
    )
    monkeypatch.setattr(server, "approval_store", approval_store)
    client = TestClient(app)

    user_response = client.get(
        "/approvals",
        headers={
            "X-API-Key": "dev-api-key",
            "X-Tenant-ID": "tenant-a",
            "X-User-Role": "user",
        },
    )
    operator_response = client.get(
        "/approvals",
        params={"status": "pending"},
        headers={
            "X-API-Key": "dev-api-key",
            "X-Tenant-ID": "tenant-a",
            "X-User-Role": "operator",
        },
    )

    assert user_response.status_code == 403
    assert operator_response.status_code == 200
    assert [item["approval_id"] for item in operator_response.json()] == [
        expected.approval_id
    ]
