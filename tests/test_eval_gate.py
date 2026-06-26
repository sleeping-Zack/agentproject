from rag.eval_gate import EvalGate, EvalThresholds


def test_eval_gate_fails_with_bucket_and_latency_breakdown():
    gate = EvalGate(
        EvalThresholds(
            min_pass_rate=0.9,
            min_tool_recall=0.8,
            max_p95_latency_ms=800,
            max_avg_cost=0.05,
        )
    )
    report = {
        "aggregate": {"pass_rate": 0.75, "tool_recall": 0.7},
        "latency": {"p95_ms": 1200},
        "cost": {"avg": 0.03},
        "cases": [
            {"id": "rag-1", "bucket": "rag", "passed": False, "error_type": "citation_missing"},
            {"id": "tool-1", "bucket": "tool", "passed": False, "error_type": "tool_miss"},
        ],
    }

    result = gate.evaluate(report)

    assert result.passed is False
    assert "pass_rate_below_threshold" in result.failures
    assert result.failure_breakdown["rag"]["citation_missing"] == 1
    assert result.failure_breakdown["tool"]["tool_miss"] == 1


def test_eval_gate_passes_when_metrics_meet_thresholds():
    gate = EvalGate(EvalThresholds(min_pass_rate=0.8, min_tool_recall=0.7))

    result = gate.evaluate(
        {
            "aggregate": {"pass_rate": 0.9, "tool_recall": 0.85},
            "latency": {"p95_ms": 200},
            "cost": {"avg": 0.01},
            "cases": [],
        }
    )

    assert result.passed is True
    assert result.failures == []
