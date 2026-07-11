import asyncio
import time
from types import SimpleNamespace
from uuid import uuid4

import pytest
from fastapi.testclient import TestClient

import api.server as server
from agent.react_agent import ReactAgent
from agent.runner import AgentBackendResult, AgentRunner, AgentTask
from agent.verifier import VerifyResult
from observability.event_bus import (
    AgentEvent,
    EventBus,
    EventStreamConflictError,
    event_bus,
)
from services.approval_store import SQLiteApprovalStore
from services.artifact_store import SQLiteArtifactStore


class _AcceptVerifier:
    def verify(self, **_kwargs):
        return VerifyResult(passed=True, action="accept", score=1.0)


def _runner(tmp_path, backend, **kwargs):
    return AgentRunner(
        backend=backend,
        verifier=_AcceptVerifier(),
        approval_store=SQLiteApprovalStore(str(tmp_path / "approvals.db")),
        artifact_store=SQLiteArtifactStore(str(tmp_path / "artifacts.db")),
        **kwargs,
    )


def test_runner_streams_tokens_before_terminal_event(tmp_path):
    class Backend:
        def __call__(self, task, _state):
            event_bus.publish(task.request_id, "token_delta", {"delta": "A"})
            event_bus.publish(task.request_id, "token_delta", {"delta": "B"})
            return AgentBackendResult(answer="AB")

    runner = _runner(tmp_path, Backend())
    task = AgentTask(query="stream", request_id=str(uuid4()))

    async def collect():
        return [event async for event in runner.run_stream(task)]

    try:
        events = asyncio.run(collect())
        event_types = [event.event_type for event in events]
        assert event_types.count("token_delta") == 2
        assert event_types.index("token_delta") < event_types.index("run_completed")
        assert [event.sequence for event in events] == list(range(1, len(events) + 1))
    finally:
        event_bus.discard(task.request_id)


def test_runner_emits_heartbeat_while_backend_is_idle(tmp_path):
    class SlowBackend:
        def __call__(self, _task, _state):
            time.sleep(0.08)
            return AgentBackendResult(answer="done")

    runner = _runner(tmp_path, SlowBackend())
    task = AgentTask(query="slow", request_id=str(uuid4()))

    async def collect():
        return [
            event
            async for event in runner.run_stream(task, heartbeat_seconds=0.02)
        ]

    try:
        events = asyncio.run(collect())
        assert any(event.event_type == "heartbeat" for event in events)
        assert events[-1].event_type == "run_completed"
    finally:
        event_bus.discard(task.request_id)


def test_event_bus_atomically_binds_stream_identity():
    bus = EventBus()
    identity = {"tenant_id": "tenant-a", "session_id": "session-a"}

    assert bus.open("request", identity=identity) is True
    assert bus.open("request", identity=identity) is False
    with pytest.raises(EventStreamConflictError):
        bus.open(
            "request",
            identity={"tenant_id": "tenant-b", "session_id": "session-a"},
        )


def test_react_agent_publishes_each_model_chunk():
    request_id = str(uuid4())
    react_agent = ReactAgent.__new__(ReactAgent)
    react_agent.memory = SimpleNamespace(get_messages=lambda *_args, **_kwargs: [])

    class StreamStub:
        def stream(self, *_args, **_kwargs):
            yield {
                "type": "messages",
                "data": (SimpleNamespace(text="A", content="A"), {}),
            }
            yield {
                "type": "messages",
                "data": (SimpleNamespace(text="B", content="B"), {}),
            }

    react_agent.agent = StreamStub()
    event_bus.open(request_id)

    try:
        chunks = list(
            react_agent.execute_stream(
                "query",
                request_id=request_id,
                emit_events=True,
            )
        )
        events = event_bus.replay(request_id)
        assert chunks == ["A", "AB"]
        assert [event.payload["delta"] for event in events] == ["A", "B"]
    finally:
        event_bus.discard(request_id)


def test_api_stream_formats_sequenced_events(monkeypatch):
    class StreamingRunner:
        async def run_stream(self, task, last_event_id=0):
            yield AgentEvent(
                task.request_id,
                "token_delta",
                last_event_id + 1,
                time.time(),
                {"delta": "A"},
            )
            yield AgentEvent(
                task.request_id,
                "token_delta",
                last_event_id + 2,
                time.time(),
                {"delta": "B"},
            )
            yield AgentEvent(
                task.request_id,
                "run_completed",
                last_event_id + 3,
                time.time(),
                {"status": "completed", "answer": "AB"},
            )

    monkeypatch.setattr(server, "harness_runner", StreamingRunner())
    client = TestClient(server.app)
    request_id = str(uuid4())

    with client.stream(
        "POST",
        "/chat/stream",
        headers={"X-API-Key": "dev-api-key", "Last-Event-ID": "4"},
        json={"message": "stream", "request_id": request_id},
    ) as response:
        body = "".join(response.iter_text())

    assert response.status_code == 200
    assert response.headers["X-Request-ID"] == request_id
    assert "id: 5\nevent: token_delta" in body
    assert "id: 6\nevent: token_delta" in body
    assert "id: 7\nevent: run_completed" in body
    assert '"answer":"AB"' in body


def test_api_rejects_cross_tenant_stream_reuse():
    client = TestClient(server.app)
    request_id = str(uuid4())
    event_bus.open(
        request_id,
        identity={
            "tenant_id": "tenant-a",
            "session_id": "default",
            "query_sha256": AgentRunner.stream_identity(
                AgentTask(query="stream", tenant_id="tenant-a")
            )["query_sha256"],
        },
    )

    try:
        response = client.post(
            "/chat/stream",
            headers={"X-API-Key": "dev-api-key", "X-Tenant-ID": "tenant-b"},
            json={"message": "stream", "request_id": request_id},
        )
        assert response.status_code == 409
    finally:
        event_bus.discard(request_id)
