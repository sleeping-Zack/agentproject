from fastapi.testclient import TestClient

from api.server import app


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
        headers={"X-API-Key": "dev-api-key", "X-Tenant-ID": "api-test"},
        json={"decided_by": "tester"},
    )
    assert approved.status_code == 200
    assert approved.json()["status"] == "approved"
