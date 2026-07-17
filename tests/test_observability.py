from observability.tracing import TraceRecorder, otel_spans_from_trace_payload


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


def test_trace_recorder_exports_opentelemetry_style_spans():
    recorder = TraceRecorder()
    recorder.start_trace(request_id="req-otel", session_id="session-1")

    with recorder.span("req-otel", "model", "qwen", {"tokens": 12}):
        pass

    spans = recorder.export_otel_spans("req-otel")

    assert spans[0]["trace_id"] == "req-otel"
    assert spans[0]["name"] == "model.qwen"
    assert spans[0]["attributes"]["tokens"] == 12


def test_otel_span_export_can_use_persisted_trace_payload():
    recorder = TraceRecorder()
    recorder.start_trace(request_id="req-persisted", session_id="session-1")
    with recorder.span("req-persisted", "tool", "first"):
        pass
    with recorder.span("req-persisted", "model", "second"):
        pass

    spans = otel_spans_from_trace_payload(recorder.export_trace("req-persisted"))

    assert [span["name"] for span in spans] == ["tool.first", "model.second"]
