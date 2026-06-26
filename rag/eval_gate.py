from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List


@dataclass
class EvalThresholds:
    min_pass_rate: float = 0.85
    min_tool_recall: float = 0.75
    min_keyword_recall: float = 0.75
    max_p95_latency_ms: float = 5000.0
    max_avg_cost: float = 0.2


@dataclass
class EvalGateResult:
    passed: bool
    failures: List[str] = field(default_factory=list)
    failure_breakdown: Dict[str, Dict[str, int]] = field(default_factory=dict)


class EvalGate:
    def __init__(self, thresholds: EvalThresholds | None = None) -> None:
        self.thresholds = thresholds or EvalThresholds()

    def evaluate(self, report: Dict) -> EvalGateResult:
        aggregate = report.get("aggregate", {})
        latency = report.get("latency", {})
        cost = report.get("cost", {})
        failures: List[str] = []

        if aggregate.get("pass_rate", 0.0) < self.thresholds.min_pass_rate:
            failures.append("pass_rate_below_threshold")
        if aggregate.get("tool_recall", 0.0) < self.thresholds.min_tool_recall:
            failures.append("tool_recall_below_threshold")
        if aggregate.get("keyword_recall", 1.0) < self.thresholds.min_keyword_recall:
            failures.append("keyword_recall_below_threshold")
        if latency.get("p95_ms", 0.0) > self.thresholds.max_p95_latency_ms:
            failures.append("p95_latency_above_threshold")
        if cost.get("avg", 0.0) > self.thresholds.max_avg_cost:
            failures.append("avg_cost_above_threshold")

        breakdown: Dict[str, Dict[str, int]] = {}
        for case in report.get("cases", []):
            if case.get("passed", True):
                continue
            bucket = case.get("bucket") or "unknown"
            error_type = case.get("error_type") or case.get("error") or "failed"
            breakdown.setdefault(bucket, {})
            breakdown[bucket][error_type] = breakdown[bucket].get(error_type, 0) + 1

        return EvalGateResult(
            passed=not failures,
            failures=failures,
            failure_breakdown=breakdown,
        )
