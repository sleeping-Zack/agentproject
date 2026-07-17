import pytest


pytest.importorskip("opentelemetry.sdk")


def test_trace_recorder_emits_real_opentelemetry_spans(monkeypatch):
    from opentelemetry.sdk.trace import TracerProvider
    from opentelemetry.sdk.trace.export import SimpleSpanProcessor
    from opentelemetry.sdk.trace.export.in_memory_span_exporter import InMemorySpanExporter

    import observability.otel as otel
    from observability.tracing import TraceRecorder

    exporter = InMemorySpanExporter()
    provider = TracerProvider()
    provider.add_span_processor(SimpleSpanProcessor(exporter))
    monkeypatch.setattr(otel, "_tracer", provider.get_tracer("test"))

    recorder = TraceRecorder()
    recorder.start_trace("req-real-otel", "session-a")
    with recorder.span(
        "req-real-otel",
        "tool",
        "get_weather",
        {"city": "Hefei", "token": "secret"},
    ):
        pass

    spans = exporter.get_finished_spans()
    assert len(spans) == 1
    assert spans[0].name == "tool.get_weather"
    assert spans[0].attributes["agent.request_id"] == "req-real-otel"
    assert spans[0].attributes["token"] == "<redacted>"
    provider.shutdown()
