from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, List, Optional


@dataclass
class EvalThresholds:
    min_pass_rate: float = 0.85
    min_tool_recall: float = 0.75
    min_keyword_recall: float = 0.75
    min_parameter_accuracy: Optional[float] = None
    min_citation_validity: Optional[float] = None
    min_standard_tool_recall: Optional[float] = None
    min_high_risk_pass_rate: Optional[float] = None
    min_high_risk_tool_recall: Optional[float] = None
    min_high_risk_parameter_accuracy: Optional[float] = None
    min_high_risk_citation_validity: Optional[float] = None
    min_case_count: int = 1
    minimum_bucket_case_counts: Dict[str, int] = field(default_factory=dict)
    minimum_risk_case_counts: Dict[str, int] = field(default_factory=dict)
    minimum_applicable_case_counts: Dict[str, int] = field(default_factory=dict)
    max_offline_harness_p95_ms: Optional[float] = None
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
        offline_latency = report.get("offline_harness_latency") or {}
        cost = report.get("cost", {})
        risk_tiers = report.get("risk_tiers") or {}
        buckets = report.get("buckets") or {}
        applicable_counts = report.get("applicable_case_counts") or {}
        failures: List[str] = []

        self._check_min(failures, aggregate, "pass_rate", self.thresholds.min_pass_rate)
        self._check_min(failures, aggregate, "tool_recall", self.thresholds.min_tool_recall)
        self._check_min(
            failures, aggregate, "keyword_recall", self.thresholds.min_keyword_recall
        )
        self._check_min(
            failures,
            aggregate,
            "parameter_accuracy",
            self.thresholds.min_parameter_accuracy,
        )
        self._check_min(
            failures,
            aggregate,
            "citation_validity",
            self.thresholds.min_citation_validity,
        )

        standard = risk_tiers.get("standard") or {}
        high = risk_tiers.get("high") or {}
        self._check_min(
            failures,
            standard,
            "tool_recall",
            self.thresholds.min_standard_tool_recall,
            "standard_tool_recall_below_threshold",
        )
        self._check_min(
            failures,
            high,
            "pass_rate",
            self.thresholds.min_high_risk_pass_rate,
            "high_risk_pass_rate_below_threshold",
        )
        self._check_min(
            failures,
            high,
            "tool_recall",
            self.thresholds.min_high_risk_tool_recall,
            "high_risk_tool_recall_below_threshold",
        )
        self._check_min(
            failures,
            high,
            "parameter_accuracy",
            self.thresholds.min_high_risk_parameter_accuracy,
            "high_risk_parameter_accuracy_below_threshold",
        )
        self._check_min(
            failures,
            high,
            "citation_validity",
            self.thresholds.min_high_risk_citation_validity,
            "high_risk_citation_validity_below_threshold",
        )

        if int(aggregate.get("case_count", 0)) < self.thresholds.min_case_count:
            failures.append("case_count_below_threshold")
        for bucket, minimum in self.thresholds.minimum_bucket_case_counts.items():
            if int((buckets.get(bucket) or {}).get("case_count", 0)) < int(minimum):
                failures.append(f"bucket_case_count_below_threshold:{bucket}")
        for tier, minimum in self.thresholds.minimum_risk_case_counts.items():
            if int((risk_tiers.get(tier) or {}).get("case_count", 0)) < int(minimum):
                failures.append(f"risk_case_count_below_threshold:{tier}")
        for dimension, minimum in self.thresholds.minimum_applicable_case_counts.items():
            if int(applicable_counts.get(dimension, 0)) < int(minimum):
                failures.append(f"applicable_case_count_below_threshold:{dimension}")

        latency_limit = self.thresholds.max_offline_harness_p95_ms
        if latency_limit is not None:
            if offline_latency.get("p95_ms") is None:
                failures.append("offline_harness_latency_missing")
            elif float(offline_latency["p95_ms"]) > latency_limit:
                failures.append("offline_harness_p95_latency_above_threshold")
        if cost.get("mode") == "disabled":
            failures.append("cost_disabled")
        elif float(cost.get("avg", 0.0)) > self.thresholds.max_avg_cost:
            failures.append("avg_cost_above_threshold")

        breakdown: Dict[str, Dict[str, int]] = {}
        for case in report.get("cases", []):
            if case.get("passed", True):
                continue
            bucket = case.get("bucket") or "unknown"
            error_type = case.get("error_type") or case.get("error") or "failed"
            breakdown.setdefault(bucket, {})
            breakdown[bucket][error_type] = breakdown[bucket].get(error_type, 0) + 1

        failures = list(dict.fromkeys(failures))
        return EvalGateResult(
            passed=not failures,
            failures=failures,
            failure_breakdown=breakdown,
        )

    @staticmethod
    def _check_min(
        failures: List[str],
        metrics: Dict,
        metric: str,
        minimum: Optional[float],
        failure_name: Optional[str] = None,
    ) -> None:
        if minimum is None:
            return
        value = metrics.get(metric)
        if value is None or float(value) < minimum:
            failures.append(failure_name or f"{metric}_below_threshold")
