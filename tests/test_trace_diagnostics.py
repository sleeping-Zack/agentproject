from observability.tracing import TraceRecorder


def test_trace_recorder_exports_structured_diagnostic_event():
    recorder = TraceRecorder()
    recorder.start_trace(request_id="req-diagnostic", session_id="session-1")

    recorder.record_diagnostic_event(
        request_id="req-diagnostic",
        step_id="step-1",
        event_type="tool_call",
        status="ok",
        latency_ms=12.5,
        tool="rag_summarize",
        args_hash="abc123",
        tokens_in=10,
        tokens_out=20,
        cost=0.001,
        evidence_ids=["doc-1"],
        verifier={"passed": True},
        retry=0,
        prompt_version="main:v1",
        model_name="mock:qwen",
    )

    event = recorder.export_trace("req-diagnostic")["events"][0]

    assert event["category"] == "diagnostic"
    assert event["name"] == "tool_call"
    assert event["duration_ms"] == 12.5
    assert event["metadata"]["step_id"] == "step-1"
    assert event["metadata"]["tool"] == "rag_summarize"
    assert event["metadata"]["evidence_ids"] == ["doc-1"]
    assert event["metadata"]["verifier"]["passed"] is True
