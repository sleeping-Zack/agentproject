from fastapi.testclient import TestClient

import api.server as server
from agent.long_term_memory import LongTermMemoryService
from services.memory_store import SQLiteMemoryStore


def _headers(**overrides):
    headers = {
        "X-API-Key": "dev-api-key",
        "X-Tenant-ID": "tenant-a",
        "X-Principal-ID": "user-1",
    }
    headers.update(overrides)
    return headers


def test_memory_api_requires_explicit_principal_id(tmp_path, monkeypatch):
    monkeypatch.setattr(
        server.agent,
        "long_term_memory",
        LongTermMemoryService(SQLiteMemoryStore(str(tmp_path / "memory.db"))),
    )
    client = TestClient(server.app)

    response = client.get(
        "/memory",
        headers={"X-API-Key": "dev-api-key", "X-Tenant-ID": "tenant-a"},
    )

    assert response.status_code == 400
    assert response.json()["detail"] == "X-Principal-ID is required for cross-session memory"


def test_memory_api_can_remember_correct_list_and_forget(tmp_path, monkeypatch):
    monkeypatch.setattr(
        server.agent,
        "long_term_memory",
        LongTermMemoryService(SQLiteMemoryStore(str(tmp_path / "memory.db"))),
    )
    client = TestClient(server.app)

    created = client.post(
        "/memory",
        headers=_headers(),
        json={"key": "profile.city", "value": "深圳", "category": "stable_profile"},
    )
    corrected = client.post(
        "/memory",
        headers=_headers(),
        json={"key": "profile.city", "value": "上海", "category": "stable_profile"},
    )
    listed = client.get("/memory", headers=_headers())
    forgotten = client.request(
        "DELETE", "/memory", headers=_headers(), json={"key": "profile.city"}
    )

    assert created.status_code == 200 and created.json()["version"] == 1
    assert corrected.status_code == 200 and corrected.json()["version"] == 2
    assert [(item["key"], item["value"]) for item in listed.json()] == [
        ("profile.city", "上海")
    ]
    assert forgotten.json() == {"deleted": 1}
    assert client.get("/memory", headers=_headers()).json() == []
