from observability.metrics import MetricsRegistry


def test_counter_and_render_prometheus():
    reg = MetricsRegistry()

    reg.inc_request("success")
    reg.inc_request("success")
    reg.inc_request("error")

    text = reg.render_prometheus()
    assert "# TYPE agent_request_total counter" in text
    assert 'agent_request_total{status="success"} 2' in text
    assert 'agent_request_total{status="error"} 1' in text


def test_histogram_buckets_and_export():
    reg = MetricsRegistry()

    for value in (10, 50, 150, 800, 5000):
        reg.observe_request_latency(value)

    snapshot = reg.snapshot()
    histograms = snapshot["histograms"]
    key = "agent_request_latency_ms"
    assert histograms[key]["count"] == 5
    text = reg.render_prometheus()
    assert "agent_request_latency_ms_bucket" in text
    assert "agent_request_latency_ms_count 5" in text


def test_tool_call_metrics():
    reg = MetricsRegistry()

    reg.inc_tool_call("get_weather", status="success")
    reg.inc_tool_call("get_weather", status="success")
    reg.inc_tool_call("get_weather", status="failure")
    reg.observe_tool_latency("get_weather", 12.5)

    text = reg.render_prometheus()
    assert 'agent_tool_call_total{status="success",tool="get_weather"} 2' in text
    assert 'agent_tool_call_total{status="failure",tool="get_weather"} 1' in text
    assert "agent_tool_latency_ms_count" in text


def test_gauge_set_and_overwrite():
    reg = MetricsRegistry()

    reg.set_rag_score("recall_at_3", 0.8)
    reg.set_rag_score("recall_at_3", 0.95)

    snapshot = reg.snapshot()
    key = 'agent_rag_eval_score{metric="recall_at_3"}'
    assert snapshot["gauges"][key] == 0.95
