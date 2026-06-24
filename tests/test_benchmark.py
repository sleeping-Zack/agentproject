from scripts.benchmark_api import summarize_latency


def test_summarize_latency_outputs_p50_p95_qps_and_failure_rate():
    summary = summarize_latency(
        latencies_ms=[100, 200, 300, 400],
        success_count=3,
        failure_count=1,
        elapsed_seconds=2,
    )

    assert summary["p50_ms"] == 250
    assert summary["p95_ms"] == 400
    assert summary["qps"] == 2.0
    assert summary["failure_rate"] == 0.25
