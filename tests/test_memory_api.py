from fastapi.testclient import TestClient

import api.server as server
from agent.long_term_memory import LongTermMemoryService
from services.memory_store import SQLiteMemoryStore
from services.persistence import SQLiteStore


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


def test_api_rejects_forged_principal_when_key_is_server_bound(monkeypatch):
    monkeypatch.setenv("AGENT_AUTH_MODE", "principal_api_key")
    monkeypatch.setenv(
        "AGENT_API_PRINCIPALS_JSON",
        '{"bound-key":{"tenant_id":"tenant-a","user_id":"user-1","role":"user"}}',
    )
    client = TestClient(server.app)

    response = client.get(
        "/memory",
        headers={
            "X-API-Key": "bound-key",
            "X-Tenant-ID": "tenant-a",
            "X-Principal-ID": "user-2",
            "X-User-Role": "user",
        },
    )

    assert response.status_code == 401
    assert "does not match authenticated identity" in response.json()["detail"]


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


def test_memory_api_can_review_a_pending_conflict(tmp_path, monkeypatch):
    long_term = LongTermMemoryService(
        SQLiteMemoryStore(str(tmp_path / "memory.db"))
    )
    monkeypatch.setattr(server.agent, "long_term_memory", long_term)
    long_term.remember(
        "tenant-a", "user-1", "profile.city", "深圳", "stable_profile"
    )
    long_term.process_turn(
        "tenant-a", "user-1", "session-1", "request-1", "我住在上海", "收到"
    )
    pending = next(
        item
        for item in long_term.list_memories(
            "tenant-a", "user-1", include_inactive=True
        )
        if item.status == "pending_confirmation"
    )
    client = TestClient(server.app)

    response = client.post(
        f"/memory/{pending.memory_id}/review",
        headers=_headers(),
        json={"decision": "accept"},
    )

    assert response.status_code == 200
    assert response.json()["value"] == "上海"
    assert response.json()["status"] == "active"


def test_memory_layers_and_session_history_are_exposed_per_user(tmp_path, monkeypatch):
    session_store = SQLiteStore(str(tmp_path / "agent.db"))
    memory_store = SQLiteMemoryStore(str(tmp_path / "memory.db"))
    long_term = LongTermMemoryService(memory_store)
    monkeypatch.setattr(server, "store", session_store)
    monkeypatch.setattr(server.agent, "long_term_memory", long_term)

    session_store.save_session_message(
        "session-1", "user", "我的型号是 S10", tenant_id="tenant-a",
        user_id="user-1", request_id="request-1",
    )
    session_store.save_session_message(
        "session-1", "assistant", "已记录", tenant_id="tenant-a",
        user_id="user-1", request_id="request-1",
    )
    session_store.save_session_message(
        "session-2", "user", "其他人的消息", tenant_id="tenant-a",
        user_id="user-2", request_id="request-2",
    )
    long_term.process_turn(
        "tenant-a", "user-1", "session-1", "request-1", "我的型号是 S10", "已记录"
    )
    memory_store.save_summary(
        "tenant-a", "user-1", "session-1", "用户设备为 S10", 2, "summary-v1"
    )
    candidate = long_term.propose_procedure(
        "标准排查", "先检查滤网", agent_version="v1", tenant_id="tenant-a"
    )
    long_term.approve_procedure(candidate.procedure_id)
    client = TestClient(server.app)

    sessions = client.get("/sessions", headers=_headers()).json()
    messages = client.get("/sessions/session-1/messages", headers=_headers()).json()
    events = client.get("/memory/events", headers=_headers()).json()
    summaries = client.get("/memory/summaries", headers=_headers()).json()
    procedures = client.get("/memory/procedures", headers=_headers()).json()

    assert [item["session_id"] for item in sessions] == ["session-1"]
    assert messages["messages"][0]["content"] == "我的型号是 S10"
    assert events[0]["request_id"] == "request-1"
    assert summaries[0]["summary"] == "用户设备为 S10"
    assert procedures[0]["title"] == "标准排查"
    assert client.get(
        "/sessions/session-2/messages", headers=_headers()
    ).status_code == 404


def test_candidate_procedures_are_not_visible_to_regular_user(tmp_path, monkeypatch):
    long_term = LongTermMemoryService(SQLiteMemoryStore(str(tmp_path / "memory.db")))
    monkeypatch.setattr(server.agent, "long_term_memory", long_term)
    long_term.propose_procedure(
        "待认证流程", "候选内容", agent_version="v1", tenant_id="tenant-a"
    )
    client = TestClient(server.app)

    response = client.get(
        "/memory/procedures?status=candidate", headers=_headers()
    )

    assert response.status_code == 403
