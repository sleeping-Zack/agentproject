import pytest

from services.frontend_api import AgentApiClient, AgentApiError, parse_sse_lines


class FakeResponse:
    def __init__(self, payload=None, *, status_code=200, lines=None, text=""):
        self._payload = payload
        self.status_code = status_code
        self._lines = lines or []
        self.text = text
        self.encoding = None

    def json(self):
        return self._payload

    def iter_lines(self, decode_unicode=False):
        assert decode_unicode is True
        return iter(self._lines)

    def close(self):
        return None


class FakeSession:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def request(self, method, url, **kwargs):
        self.calls.append((method, url, kwargs))
        return self.responses.pop(0)


def test_parse_sse_lines_handles_multiple_events_and_multiline_data():
    lines = [
        "id: 1",
        "event: token_delta",
        'data: {"delta":"你"}',
        "",
        "id: 2",
        "event: run_completed",
        'data: {"status":"completed",',
        'data: "answer":"你好"}',
        "",
    ]

    assert list(parse_sse_lines(lines)) == [
        {"id": "1", "event": "token_delta", "data": {"delta": "你"}},
        {
            "id": "2",
            "event": "run_completed",
            "data": {"status": "completed", "answer": "你好"},
        },
    ]


def test_client_sends_stable_identity_headers_and_streams_chat_events():
    session = FakeSession(
        [
            FakeResponse(
                lines=[
                    "id: 1",
                    "event: token_delta",
                    'data: {"delta":"A"}',
                    "",
                ]
            )
        ]
    )
    client = AgentApiClient(
        "http://127.0.0.1:8000/",
        "secret",
        "tenant-a",
        "user-1",
        session=session,
    )

    events = list(
        client.chat_events(
            "hello",
            "session-1",
            request_id="request-1",
            approval_id="approval-1",
        )
    )

    method, url, kwargs = session.calls[0]
    assert events[0]["data"] == {"delta": "A"}
    assert method == "POST" and url == "http://127.0.0.1:8000/chat/stream"
    assert kwargs["headers"] == {"X-API-Key": "secret"}
    assert kwargs["json"] == {
        "message": "hello",
        "session_id": "session-1",
        "request_id": "request-1",
        "approval_id": "approval-1",
    }
    assert kwargs["stream"] is True


def test_client_lists_pending_approvals():
    session = FakeSession([FakeResponse([{"approval_id": "approval-1"}])])
    client = AgentApiClient(
        "http://api", "operator-key", "tenant-a", "operator-1", session=session
    )

    assert client.list_approvals(status="pending") == [{"approval_id": "approval-1"}]
    assert session.calls[0][1] == "http://api/approvals"
    assert session.calls[0][2]["params"] == {"status": "pending", "limit": 100}


def test_client_can_send_verified_bearer_token():
    session = FakeSession([FakeResponse({"tenant_id": "tenant-a"})])
    client = AgentApiClient(
        "http://api",
        "",
        "tenant-a",
        "user-1",
        bearer_token="signed-token",
        session=session,
    )

    client.identity()

    assert session.calls[0][2]["headers"]["Authorization"] == "Bearer signed-token"


def test_client_memory_operations_use_backend_endpoints():
    session = FakeSession(
        [
            FakeResponse([{"key": "device.model", "value": "S10"}]),
            FakeResponse({"key": "device.model", "value": "S20"}),
            FakeResponse({"deleted": 1}),
            FakeResponse(
                {
                    "memory_id": "pending-1",
                    "status": "active",
                    "value": "S30",
                }
            ),
        ]
    )
    client = AgentApiClient(
        "http://api", "secret", "tenant-a", "user-1", session=session
    )

    assert client.list_memories()[0]["value"] == "S10"
    assert client.remember("device.model", "S20", "device_identity")["value"] == "S20"
    assert client.forget("device.model") == {"deleted": 1}
    assert client.review_memory("pending-1", "accept")["status"] == "active"

    assert [call[1] for call in session.calls] == [
        "http://api/memory",
        "http://api/memory",
        "http://api/memory",
        "http://api/memory/pending-1/review",
    ]
    assert [call[0] for call in session.calls] == [
        "GET",
        "POST",
        "DELETE",
        "POST",
    ]


def test_client_reads_all_memory_layers_and_session_messages():
    session = FakeSession(
        [
            FakeResponse([{"session_id": "session-1"}]),
            FakeResponse({"session_id": "session-1", "messages": []}),
            FakeResponse([{"event_id": "event-1"}]),
            FakeResponse([{"session_id": "session-1", "summary": "摘要"}]),
            FakeResponse([{"procedure_id": "procedure-1"}]),
        ]
    )
    client = AgentApiClient(
        "http://api", "secret", "tenant-a", "user-1", session=session
    )

    assert client.list_sessions()[0]["session_id"] == "session-1"
    assert client.session_messages("session-1")["session_id"] == "session-1"
    assert client.memory_events()[0]["event_id"] == "event-1"
    assert client.memory_summaries()[0]["summary"] == "摘要"
    assert client.memory_procedures()[0]["procedure_id"] == "procedure-1"
    assert [call[1] for call in session.calls] == [
        "http://api/sessions",
        "http://api/sessions/session-1/messages",
        "http://api/memory/events",
        "http://api/memory/summaries",
        "http://api/memory/procedures",
    ]


def test_client_surfaces_backend_error_detail():
    session = FakeSession(
        [FakeResponse({"detail": "invalid api key"}, status_code=401, text="unauthorized")]
    )
    client = AgentApiClient(
        "http://api", "bad", "tenant-a", "user-1", session=session
    )

    with pytest.raises(AgentApiError, match="invalid api key"):
        client.health()
