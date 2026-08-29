from scripts.benchmark_api import (
    _call,
    evaluate_performance_gate,
    summarize_latency,
    summarize_stage_latency,
)


class _StreamResponse:
    def __init__(self, lines):
        self.lines = lines

    def raise_for_status(self):
        return None

    def iter_lines(self, decode_unicode=True):
        assert decode_unicode is True
        return iter(self.lines)


def test_summarize_latency_outputs_response_ttft_throughput_and_error_rates():
    summary = summarize_latency(
        latencies_ms=[100, 200, 300],
        success_count=3,
        failure_count=1,
        elapsed_seconds=2,
        ttft_ms=[20, 30, 40],
        timeout_count=1,
        concurrency=2,
    )

    assert summary["response_latency_ms"] == {
        "sample_count": 3,
        "p50": 200,
        "p95": 300,
        "p99": 300,
    }
    assert summary["ttft_ms"]["p95"] == 40
    assert summary["qps"] == 1.5
    assert summary["attempted_qps"] == 2.0
    assert summary["failure_rate"] == 0.25
    assert summary["timeout_rate"] == 0.25


def test_performance_gate_requires_ttft_and_checks_p99_and_failures():
    summary = summarize_latency(
        latencies_ms=[1000, 2000],
        success_count=2,
        failure_count=1,
        elapsed_seconds=2,
        timeout_count=1,
    )

    gate = evaluate_performance_gate(
        summary,
        {
            "minimum_success_count": 3,
            "minimum_qps": 2,
            "maximum_ttft_p95_ms": 1000,
            "maximum_response_p99_ms": 1500,
            "maximum_timeout_rate": 0.1,
            "maximum_failure_rate": 0.1,
        },
    )

    assert gate["passed"] is False
    assert "metric_unavailable:ttft_p95" in gate["failures"]
    assert "response_p99_above_threshold" in gate["failures"]
    assert "timeout_rate_above_threshold" in gate["failures"]


def test_stage_latency_uses_only_histogram_delta():
    key = 'agent_model_latency_ms{provider="mock",scene="rag",status="success"}'
    before = {
        "histograms": {
            key: {
                "count": 2,
                "sum": 150,
                "buckets": [[50, 1], [100, 2], [200, 2]],
            }
        }
    }
    after = {
        "histograms": {
            key: {"count": 4, "sum": 370, "buckets": [[50, 1], [100, 3], [200, 4]]}
        }
    }

    stages = summarize_stage_latency(before, after)

    assert stages["model"] == {
        "sample_count": 2,
        "average": 110.0,
        "p95_upper_bound": 200.0,
        "p95_overflow": False,
    }
    assert stages["tool"]["sample_count"] == 0


def test_deployed_performance_gate_is_separate_from_pull_request_ci():
    workflow = open(".github/workflows/performance.yml", encoding="utf-8").read()

    assert "workflow_dispatch:" in workflow
    assert "pull_request:" not in workflow
    assert "config/api_performance_gates.yml" in workflow
    assert "PERFORMANCE_TARGET_URL" in workflow
    assert "PERFORMANCE_API_KEY" in workflow


def test_stream_request_counts_run_failed_as_failure(monkeypatch):
    monkeypatch.setattr(
        "scripts.benchmark_api.requests.post",
        lambda *args, **kwargs: _StreamResponse(
            ["event: run_failed", 'data: {"status":"failed"}', ""]
        ),
    )

    sample = _call("http://api/chat/stream", "key", "hello", 1, stream=True)

    assert sample == {"status": "error", "error": "run_failed"}
