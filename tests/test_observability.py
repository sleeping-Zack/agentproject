from observability.tracing import TraceRecorder


def test_trace_recorder_exports_tool_span_with_redaction():
    recorder = TraceRecorder()
    trace = recorder.start_trace(request_id="req-1", session_id="session-1")

    with recorder.span("req-1", "tool", "get_weather", {"city": "深圳", "token": "secret-value"}):
        pass

    exported = recorder.export_trace(trace.request_id)

    assert exported["request_id"] == "req-1"
    assert exported["session_id"] == "session-1"
    assert exported["events"][0]["category"] == "tool"
    assert exported["events"][0]["name"] == "get_weather"
    assert exported["events"][0]["duration_ms"] >= 0
    assert exported["events"][0]["metadata"]["token"] == "<redacted>"
