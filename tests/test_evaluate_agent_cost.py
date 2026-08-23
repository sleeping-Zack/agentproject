from observability.tracing import trace_recorder
from scripts.evaluate_agent import CaseResult, _summarize_cost


def test_cost_summary_does_not_recount_usage_from_verifier_diagnostic():
    request_id = "req-cost-dedup"
    trace_recorder.start_trace(request_id, "session-cost-dedup")
    for event_type in ("model_usage", "verifier"):
        trace_recorder.record_diagnostic_event(
            request_id=request_id,
            step_id=event_type,
            event_type=event_type,
            status="ok",
            latency_ms=0.0,
            tokens_in=100,
            tokens_out=20,
            cost=0.012,
        )
    result = CaseResult(
        id="cost-dedup",
        passed=True,
        tool_recall=1.0,
        keyword_recall=1.0,
        rejected=False,
        detail={"request_id": request_id},
    )

    summary = _summarize_cost([result])

    assert summary["avg"] == 0.012
    assert summary["tokens_avg"] == 120
