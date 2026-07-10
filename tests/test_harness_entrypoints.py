from types import SimpleNamespace

from fastapi.testclient import TestClient

import api.server as server


class FakeRunner:
    def __init__(self):
        self.calls = []

    def run(self, task):
        from observability.tracing import trace_recorder

        self.calls.append(task)
        trace_recorder.start_trace(task.request_id, task.session_id)
        return SimpleNamespace(
            request_id=task.request_id,
            answer="harness answer",
            approval_id=None,
            artifacts=[],
            verifier=None,
            state=SimpleNamespace(status="completed"),
        )


def test_chat_endpoint_uses_harness_runner(monkeypatch):
    fake_runner = FakeRunner()
    monkeypatch.setattr(server, "harness_runner", fake_runner)

    def legacy_execute_stream(*args, **kwargs):
        raise AssertionError("legacy agent path was used")

    monkeypatch.setattr(server.agent, "execute_stream", legacy_execute_stream)
    client = TestClient(server.app)

    response = client.post(
        "/chat",
        headers={"X-API-Key": "dev-api-key", "X-Tenant-ID": "tenant-entry"},
        json={"message": "怎么保养滤网", "session_id": "entry-chat"},
    )

    assert response.status_code == 200
    assert response.json()["answer"] == "harness answer"
    assert fake_runner.calls
    assert fake_runner.calls[0].tenant_id == "tenant-entry"


def test_chat_stream_endpoint_uses_harness_runner(monkeypatch):
    fake_runner = FakeRunner()
    monkeypatch.setattr(server, "harness_runner", fake_runner)

    def legacy_execute_stream(*args, **kwargs):
        raise AssertionError("legacy stream path was used")

    monkeypatch.setattr(server.agent, "execute_stream", legacy_execute_stream)
    client = TestClient(server.app)

    with client.stream(
        "POST",
        "/chat/stream",
        headers={"X-API-Key": "dev-api-key", "X-Tenant-ID": "tenant-entry"},
        json={"message": "怎么保养滤网", "session_id": "entry-stream"},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "harness answer" in body
    assert fake_runner.calls


def _persistent_runner(tmp_path, message_store):
    from agent.memory import ConversationMemory
    from agent.runner import AgentBackendResult, AgentRunner
    from services.approval_store import SQLiteApprovalStore
    from services.artifact_store import SQLiteArtifactStore

    class Backend:
        def __call__(self, task, state):
            return AgentBackendResult(answer="persisted answer")

    return AgentRunner(
        backend=Backend(),
        approval_store=SQLiteApprovalStore(str(tmp_path / "approvals.db")),
        artifact_store=SQLiteArtifactStore(str(tmp_path / "artifacts.db")),
        conversation_memory=ConversationMemory(store=message_store),
    )


def test_chat_persists_each_message_once_via_runner(monkeypatch, tmp_path):
    from services.persistence import SQLiteStore

    message_store = SQLiteStore(str(tmp_path / "messages.db"))
    monkeypatch.setattr(server, "store", message_store)
    monkeypatch.setattr(server, "harness_runner", _persistent_runner(tmp_path, message_store))
    client = TestClient(server.app)

    response = client.post(
        "/chat",
        headers={"X-API-Key": "dev-api-key", "X-Tenant-ID": "tenant-persist"},
        json={"message": "普通请求", "session_id": "ordinary"},
    )

    assert response.status_code == 200
    assert message_store.get_session_messages("ordinary", tenant_id="tenant-persist") == [
        {"role": "user", "content": "普通请求"},
        {"role": "assistant", "content": "persisted answer"},
    ]


def test_chat_stream_persists_each_message_once_via_runner(monkeypatch, tmp_path):
    from services.persistence import SQLiteStore

    message_store = SQLiteStore(str(tmp_path / "messages.db"))
    monkeypatch.setattr(server, "store", message_store)
    monkeypatch.setattr(server, "harness_runner", _persistent_runner(tmp_path, message_store))
    client = TestClient(server.app)

    with client.stream(
        "POST",
        "/chat/stream",
        headers={"X-API-Key": "dev-api-key", "X-Tenant-ID": "tenant-persist"},
        json={"message": "流式请求", "session_id": "stream"},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert "persisted answer" in body
    assert message_store.get_session_messages("stream", tenant_id="tenant-persist") == [
        {"role": "user", "content": "流式请求"},
        {"role": "assistant", "content": "persisted answer"},
    ]


def test_harness_run_ignores_body_user_role(monkeypatch, tmp_path):
    from agent.runner import AgentBackendResult, AgentRunner
    from services.approval_store import SQLiteApprovalStore
    from services.artifact_store import SQLiteArtifactStore

    class Backend:
        def __call__(self, task, state):
            return AgentBackendResult(answer="should not run")

    monkeypatch.setattr(
        server,
        "harness_runner",
        AgentRunner(
            backend=Backend(),
            approval_store=SQLiteApprovalStore(str(tmp_path / "approvals.db")),
            artifact_store=SQLiteArtifactStore(str(tmp_path / "artifacts.db")),
        ),
    )
    client = TestClient(server.app)

    response = client.post(
        "/harness/run",
        headers={
            "X-API-Key": "dev-api-key",
            "X-Tenant-ID": "tenant-entry",
            "X-User-Role": "user",
        },
        json={
            "message": "生成本月使用记录报告",
            "session_id": "entry-harness",
            "scene": "report",
            "user_role": "admin",
        },
    )

    assert response.status_code == 200
    assert response.json()["status"] == "pending_approval"
