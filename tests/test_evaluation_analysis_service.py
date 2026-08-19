from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from typing import Any, Callable

import jsonschema
import pytest

from evaluation_analysis.service import EvaluationAnalysisService


GENERATED_AT = "2026-08-12T10:00:00+08:00"
DIMENSIONS = (
    "task_completion",
    "factual_correctness",
    "tool_use",
    "instruction_following",
    "groundedness",
    "safety",
    "response_quality",
)


def _file_sha256(path: str) -> str:
    import hashlib

    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


MACHINE_CONFIG_SHA256 = _file_sha256("config/machine_evaluation.yml")
RUBRIC_SHA256 = _file_sha256("config/evaluation_rubric.yml")


def _case(
    number: int,
    *,
    split: str,
    passed: bool = True,
    overall_score: float = 3.0,
    risk_level: str = "L1",
) -> dict[str, Any]:
    return {
        "case_id": f"{split}-case-{number:03d}",
        "family_id": f"family-{number:03d}",
        "dataset_version": "agent-eval-v1",
        "split": split,
        "scene": "tool_execution",
        "category": "tool_call",
        "risk_level": risk_level,
        "capability_tags": ["tool_selection"],
        "model_metadata": {"model": "fixture-model", "snapshot": "2026-08-01"},
        "deterministic": {"passed": True, "failures": []},
        "judge": {"status": "ok"},
        "hybrid": {
            "passed": passed,
            "overall_score": overall_score,
            "scores": {dimension: 3 for dimension in DIMENSIONS},
            "vetoes": [],
        },
        "latency_ms": 100.0 + number,
        "estimated_cost": 0.01,
        "cost_mode": "estimated",
        "tokens_in": 20,
        "tokens_out": 10,
    }


def _machine_report(
    *,
    split: str = "dev",
    count: int = 30,
    production_ready: bool = True,
) -> dict[str, Any]:
    return {
        "schema_version": 1,
        "pipeline_version": "agent-machine-eval-v1",
        "rubric_version": "agent-rubric-v1",
        "prompt_version": "agent-judge-v1",
        "judge_id": "qwen-plus-2026-08-01",
        "config_sha256": MACHINE_CONFIG_SHA256,
        "rubric_sha256": RUBRIC_SHA256,
        "run_metadata": {
            "run_id": f"run-{split}",
            "variant": "baseline",
            "dataset_sha256": "3" * 64,
            "dataset_versions": ["agent-eval-v1"],
            "splits": [split],
        },
        "production_gate": {
            "status": "evaluated" if production_ready else "blocked",
            "passed": production_ready if production_ready else None,
            "failures": [] if production_ready else ["fixture_not_production_ready"],
        },
        "review_queue": {"cases": []},
        "cases": [_case(number, split=split) for number in range(1, count + 1)],
    }


def _experiment(mode: str) -> dict[str, Any]:
    return {
        "experiment_id": f"exp-{mode}",
        "mode": mode,
        "hypothesis": "candidate does not reduce paired pass rate",
        "change": "routing prompt v2",
        "primary_metric": "pass_rate",
        "predeclared_slices": ["scene"],
    }


def _approval(
    service: EvaluationAnalysisService, baseline: dict[str, Any]
) -> dict[str, Any]:
    return {
        "status": "approved",
        "approver": "evaluation-owner@example.com",
        "approved_at": GENERATED_AT,
        "report_sha256": service.report_sha256(baseline),
    }


def _analyze(
    service: EvaluationAnalysisService,
    baseline: dict[str, Any],
    candidate: dict[str, Any],
    *,
    mode: str,
) -> dict[str, Any]:
    return service.analyze(
        baseline,
        candidate,
        experiment=_experiment(mode),
        baseline_approval=_approval(service, baseline) if mode == "promotion" else None,
        report_id=f"report-{mode}",
        generated_at=GENERATED_AT,
    )


def test_diagnostic_dev_computes_paired_results_but_cannot_authorize_release() -> None:
    service = EvaluationAnalysisService()
    baseline = _machine_report(split="dev", count=5)
    candidate = deepcopy(baseline)

    report = _analyze(service, baseline, candidate, mode="diagnostic")

    assert report["comparability"]["status"] == "comparable"
    assert report["evidence"]["pair_count"] == 5
    assert report["quality_comparison"]["pass_rate"]["delta"] == 0.0
    assert report["release_decision"] == {
        "status": "diagnostic_only",
        "reasons": [
            "diagnostic_data_cannot_authorize_release",
            "paired_case_count_below_threshold",
        ],
        "requires_human_approval": False,
    }


def test_equal_approved_test_pair_is_eligible_for_human_approval_only() -> None:
    service = EvaluationAnalysisService()
    baseline = _machine_report(split="test")
    candidate = deepcopy(baseline)
    candidate["run_metadata"]["variant"] = "candidate"

    report = _analyze(service, baseline, candidate, mode="promotion")

    assert report["evaluator_gate"]["status"] == "passed"
    assert report["evidence"]["pair_count"] == 30
    assert report["evidence"]["non_inferior"] is True
    assert report["performance"]["status"] == "available"
    assert report["release_decision"]["status"] == "eligible_for_human_approval"
    assert report["release_decision"]["requires_human_approval"] is True
    assert report["bad_cases"] == []
    assert report["recommendations"] == []
    assert report["regression_candidates"] == {"proposed": [], "excluded": []}


def test_safety_veto_p0_and_l3_regression_keeps_baseline() -> None:
    service = EvaluationAnalysisService()
    baseline = _machine_report(split="test")
    candidate = deepcopy(baseline)
    baseline["cases"][0]["risk_level"] = "L3"
    candidate["cases"][0]["risk_level"] = "L3"
    degraded = candidate["cases"][0]
    degraded["hybrid"].update(
        {
            "passed": False,
            "vetoes": ["unsafe_action"],
        }
    )
    degraded["hybrid"]["scores"]["safety"] = 0
    degraded["hybrid"]["overall_score"] = service.rubric.overall_score(
        degraded["hybrid"]["scores"]
    )
    candidate["review_queue"]["cases"] = [
        {"case_id": degraded["case_id"], "priority": "P0"}
    ]

    report = _analyze(service, baseline, candidate, mode="promotion")

    assert report["release_decision"]["status"] == "keep_baseline"
    assert {
        "new_safety_failure",
        "new_veto",
        "new_p0_regression",
        "new_l3_pass_regression",
    } <= set(report["release_decision"]["reasons"])
    assert report["safety"]["new_failure_count"] == 1
    assert report["safety"]["new_veto_count"] == 1
    assert report["safety"]["new_p0_count"] == 1
    assert report["safety"]["new_l3_failure_count"] == 1
    assert all("case_id" not in key and "vetoes" not in key for key in report["safety"])


def test_twenty_nine_promotion_pairs_are_insufficient_and_keep_baseline() -> None:
    service = EvaluationAnalysisService()
    baseline = _machine_report(split="test", count=29)
    candidate = deepcopy(baseline)

    report = _analyze(service, baseline, candidate, mode="promotion")

    assert report["evidence"]["status"] == "insufficient_sample"
    assert report["evidence"]["minimum_pair_count"] == 30
    assert report["release_decision"]["status"] == "keep_baseline"
    assert "paired_case_count_below_threshold" in report["release_decision"]["reasons"]


def _remove_case(report: dict[str, Any]) -> None:
    report["cases"].pop()


def _change_dataset_version(report: dict[str, Any]) -> None:
    for row in report["cases"]:
        row["dataset_version"] = "agent-eval-v2"
    report["run_metadata"]["dataset_versions"] = ["agent-eval-v2"]


def _change_judge(report: dict[str, Any]) -> None:
    report["judge_id"] = "different-pinned-judge"


def _change_split(report: dict[str, Any]) -> None:
    for number, row in enumerate(report["cases"], start=1):
        row["split"] = "regression"
        row["case_id"] = f"regression-case-{number:03d}"
    report["run_metadata"]["splits"] = ["regression"]


def _change_machine_config_hash(report: dict[str, Any]) -> None:
    report["config_sha256"] = "4" * 64


def _change_rubric_hash(report: dict[str, Any]) -> None:
    report["rubric_sha256"] = "5" * 64


def _change_dataset_hash(report: dict[str, Any]) -> None:
    report["run_metadata"]["dataset_sha256"] = "6" * 64


@pytest.mark.parametrize(
    ("mutation", "expected_reason"),
    [
        (_remove_case, "case_set_mismatch"),
        (_change_dataset_version, "dataset_version_mismatch"),
        (_change_judge, "judge_id_mismatch"),
        (_change_split, "split_mismatch"),
        (_change_machine_config_hash, "machine_config_fingerprint_mismatch"),
        (_change_rubric_hash, "rubric_fingerprint_mismatch"),
        (_change_dataset_hash, "dataset_fingerprint_mismatch"),
    ],
)
def test_noncomparable_identity_or_fingerprint_is_blocked(
    mutation: Callable[[dict[str, Any]], None], expected_reason: str
) -> None:
    service = EvaluationAnalysisService()
    baseline = _machine_report(split="test")
    candidate = deepcopy(baseline)
    mutation(candidate)

    report = _analyze(service, baseline, candidate, mode="promotion")

    assert report["comparability"]["status"] == "not_comparable"
    assert expected_reason in report["comparability"]["reasons"]
    assert report["release_decision"]["status"] == "blocked"
    assert report["release_decision"]["requires_human_approval"] is False


def test_diagnostic_cannot_read_frozen_test_split() -> None:
    service = EvaluationAnalysisService()
    baseline = _machine_report(split="test")
    candidate = deepcopy(baseline)

    report = _analyze(service, baseline, candidate, mode="diagnostic")

    assert report["comparability"]["status"] == "not_comparable"
    assert "test_split_forbidden_for_diagnostic" in report["comparability"]["reasons"]
    assert report["release_decision"]["status"] == "blocked"
    assert report["bad_cases"] == []


def test_missing_performance_evidence_keeps_baseline_without_zero_imputation() -> None:
    service = EvaluationAnalysisService()
    baseline = _machine_report(split="test")
    candidate = deepcopy(baseline)
    for row in candidate["cases"]:
        row["latency_ms"] = None
        row["estimated_cost"] = None

    report = _analyze(service, baseline, candidate, mode="promotion")

    assert report["performance"]["status"] == "not_available"
    assert report["performance"]["p95_latency_ms"] == {"status": "not_available"}
    assert report["performance"]["average_cost"] == {
        "status": "not_available",
        "reason": "cost_mode_missing_or_mismatch",
    }
    assert report["release_decision"]["status"] == "keep_baseline"
    assert "performance_evidence_missing" in report["release_decision"]["reasons"]


def test_unresolved_promotion_pair_is_blocked_not_silently_excluded() -> None:
    service = EvaluationAnalysisService()
    baseline = _machine_report(split="test")
    candidate = deepcopy(baseline)
    candidate["cases"][0]["hybrid"] = None

    report = _analyze(service, baseline, candidate, mode="promotion")

    assert report["evidence"]["unresolved_pair_count"] == 1
    assert report["evaluator_gate"]["status"] == "failed"
    assert "incomplete_paired_outcomes" in report["evaluator_gate"]["reasons"]
    assert report["release_decision"]["status"] == "blocked"


def test_dev_regression_emits_suspected_rca_recommendation_and_regression_proposal() -> None:
    service = EvaluationAnalysisService()
    baseline = _machine_report(split="dev")
    candidate = deepcopy(baseline)
    degraded = candidate["cases"][0]
    degraded["deterministic"] = {
        "passed": False,
        "failures": ["required_tool_missing"],
    }
    degraded["hybrid"]["passed"] = False
    degraded["hybrid"]["scores"]["tool_use"] = 0
    degraded["hybrid"]["overall_score"] = service.rubric.overall_score(
        degraded["hybrid"]["scores"]
    )

    report = _analyze(service, baseline, candidate, mode="diagnostic")

    assert report["release_decision"]["status"] == "diagnostic_only"
    assert report["bad_cases"][0]["case_id"] == degraded["case_id"]
    assert report["bad_cases"][0]["status"] == "open"
    assert "required_tool_missing" in report["bad_cases"][0]["suspected_root_causes"]
    assert any(
        row["classification"] == "suspected"
        and row["root_cause"] == "required_tool_missing"
        for row in report["root_cause_summary"]
    )
    assert any(
        row["suspected_root_cause"] == "required_tool_missing"
        and row["status"] == "proposed"
        for row in report["recommendations"]
    )
    assert report["regression_candidates"]["proposed"][0]["source_case_id"] == degraded[
        "case_id"
    ]
    assert report["regression_candidates"]["proposed"][0]["requires_human_approval"] is True


def test_generated_report_validates_against_public_schema() -> None:
    service = EvaluationAnalysisService()
    baseline = _machine_report(split="test")
    candidate = deepcopy(baseline)
    report = _analyze(service, baseline, candidate, mode="promotion")
    schema = json.loads(
        Path("evals/evaluation_analysis/report_schema_v1.json").read_text(
            encoding="utf-8"
        )
    )

    jsonschema.Draft202012Validator(schema).validate(report)


def test_case_input_order_does_not_change_metrics_decision_or_case_order() -> None:
    service = EvaluationAnalysisService()
    baseline = _machine_report(split="dev")
    candidate = deepcopy(baseline)
    candidate["cases"][2]["hybrid"]["passed"] = False
    candidate["cases"][2]["hybrid"]["scores"]["task_completion"] = 1
    candidate["cases"][2]["hybrid"]["overall_score"] = service.rubric.overall_score(
        candidate["cases"][2]["hybrid"]["scores"]
    )

    forward = _analyze(service, baseline, candidate, mode="diagnostic")
    reversed_report = _analyze(
        service,
        {**baseline, "cases": list(reversed(baseline["cases"]))},
        {**candidate, "cases": list(reversed(candidate["cases"]))},
        mode="diagnostic",
    )

    for field in (
        "quality_comparison",
        "performance",
        "safety",
        "slices",
        "case_transitions",
        "bad_cases",
        "root_cause_summary",
        "recommendations",
        "regression_candidates",
        "release_decision",
    ):
        assert forward[field] == reversed_report[field]


def test_promotion_uses_case_split_and_dataset_version_not_claimed_metadata() -> None:
    service = EvaluationAnalysisService()
    baseline = _machine_report(split="test")
    candidate = deepcopy(baseline)
    for report in (baseline, candidate):
        for row in report["cases"]:
            row["split"] = "dev"
            row["dataset_version"] = "spoofed-v9"

    report = _analyze(service, baseline, candidate, mode="promotion")

    assert report["comparability"]["status"] == "not_comparable"
    assert {
        "baseline_split_metadata_invalid",
        "candidate_split_metadata_invalid",
        "baseline_dataset_version_metadata_invalid",
        "candidate_dataset_version_metadata_invalid",
        "promotion_requires_test_only",
    } <= set(report["comparability"]["reasons"])
    assert report["release_decision"]["status"] == "blocked"


@pytest.mark.parametrize(
    ("field", "value", "message"),
    [
        ("overall_score", 99, "overall_score must be between 0 and 3"),
        ("scores", {**{dimension: 3 for dimension in DIMENSIONS}, "safety": 99}, "scores.safety"),
        ("vetoes", ["not-a-rubric-veto"], "unknown vetoes"),
    ],
)
def test_report_semantics_reject_invalid_hybrid_results(
    field: str, value: Any, message: str
) -> None:
    service = EvaluationAnalysisService()
    baseline = _machine_report(split="dev")
    candidate = deepcopy(baseline)
    candidate["cases"][0]["hybrid"][field] = value

    with pytest.raises(ValueError, match=message):
        _analyze(service, baseline, candidate, mode="diagnostic")


def test_report_semantics_rejects_inconsistent_pass_and_overall_score() -> None:
    service = EvaluationAnalysisService()
    baseline = _machine_report(split="dev")
    candidate = deepcopy(baseline)
    candidate["cases"][0]["hybrid"]["passed"] = False

    with pytest.raises(ValueError, match="passed does not match"):
        _analyze(service, baseline, candidate, mode="diagnostic")

    candidate = deepcopy(baseline)
    candidate["cases"][0]["hybrid"]["overall_score"] = 2.9
    with pytest.raises(ValueError, match="overall_score does not match"):
        _analyze(service, baseline, candidate, mode="diagnostic")


def test_untrusted_evaluator_lineage_and_inconsistent_source_gate_are_blocked() -> None:
    service = EvaluationAnalysisService()
    baseline = _machine_report(split="test")
    candidate = deepcopy(baseline)
    for report in (baseline, candidate):
        report["config_sha256"] = "a" * 64
        report["rubric_sha256"] = "b" * 64
        report["production_gate"] = {
            "status": "blocked",
            "passed": True,
            "failures": ["safety_false_negative_detected"],
        }

    report = _analyze(service, baseline, candidate, mode="promotion")

    assert report["evaluator_gate"]["status"] == "failed"
    assert "source_report_not_production_ready" in report["evaluator_gate"]["reasons"]
    assert {
        "baseline_machine_config_fingerprint_untrusted",
        "candidate_machine_config_fingerprint_untrusted",
        "baseline_rubric_fingerprint_untrusted",
        "candidate_rubric_fingerprint_untrusted",
    } <= set(report["comparability"]["reasons"])
    assert report["release_decision"]["status"] == "blocked"


def test_promotion_report_contains_only_aggregate_frozen_test_transitions() -> None:
    service = EvaluationAnalysisService()
    baseline = _machine_report(split="test")
    candidate = deepcopy(baseline)
    degraded = candidate["cases"][0]["hybrid"]
    degraded["scores"]["task_completion"] = 1
    degraded["overall_score"] = service.rubric.overall_score(degraded["scores"])
    degraded["passed"] = False

    report = _analyze(service, baseline, candidate, mode="promotion")
    serialized = json.dumps(
        {
            "case_transitions": report["case_transitions"],
            "safety": report["safety"],
        }
    )

    assert report["case_transitions"] == {
        "improved": 0,
        "regressed": 1,
        "unchanged": 29,
    }
    assert "test-case-001" not in serialized


def test_gated_slice_score_regression_keeps_baseline_when_global_score_is_flat() -> None:
    service = EvaluationAnalysisService()
    baseline = _machine_report(split="test", count=40)
    candidate = deepcopy(baseline)
    for number, (before, after) in enumerate(
        zip(baseline["cases"], candidate["cases"]), start=1
    ):
        category = "regressed" if number <= 20 else "improved"
        before["category"] = category
        after["category"] = category
        dimension = "response_quality"
        if category == "regressed":
            after["hybrid"]["scores"][dimension] = 2
        else:
            before["hybrid"]["scores"][dimension] = 2
        before["hybrid"]["overall_score"] = service.rubric.overall_score(
            before["hybrid"]["scores"]
        )
        after["hybrid"]["overall_score"] = service.rubric.overall_score(
            after["hybrid"]["scores"]
        )

    report = service.analyze(
        baseline,
        candidate,
        experiment={**_experiment("promotion"), "predeclared_slices": ["category"]},
        baseline_approval=_approval(service, baseline),
        generated_at=GENERATED_AT,
    )

    assert report["quality_comparison"]["overall_score"]["delta"] == 0
    assert report["release_decision"]["status"] == "keep_baseline"
    assert "gated_slice_score_regressed:category=regressed" in report[
        "release_decision"
    ]["reasons"]
