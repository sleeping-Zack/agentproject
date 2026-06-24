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
