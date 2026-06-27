from fastapi.testclient import TestClient

import api.server as server
from services.artifact_store import SQLiteArtifactStore


def test_artifact_detail_endpoint_returns_payload(monkeypatch, tmp_path):
    artifact_store = SQLiteArtifactStore(str(tmp_path / "artifacts.db"))
    artifact = artifact_store.save_artifact(
        request_id="req-artifact",
        tenant_id="tenant-artifact",
        artifact_type="answer",
        name="final-answer",
        payload={"answer": "报告内容"},
    )
    monkeypatch.setattr(server, "artifact_store", artifact_store)
    client = TestClient(server.app)

    response = client.get(
        f"/artifact/{artifact.artifact_id}",
        headers={"X-API-Key": "dev-api-key", "X-Tenant-ID": "tenant-artifact"},
    )

    assert response.status_code == 200
    assert response.json()["artifact"]["payload"]["answer"] == "报告内容"
