import json

from machine_eval.judge import AgentRubricJudge
from machine_eval.pipeline import MachineEvalPipeline


DIMENSIONS = (
    "task_completion",
    "factual_correctness",
    "tool_use",
    "instruction_following",
    "groundedness",
    "safety",
    "response_quality",
)


def _judge_payload(default=3):
    return {
        "scores": {dimension: default for dimension in DIMENSIONS},
        "vetoes": [],
        "rationales": {dimension: f"{dimension} 对应证据" for dimension in DIMENSIONS},
    }


def _item(index, *, category="tool_use", risk_level="L1"):
    return {
        "case_id": f"case-{index:03d}",
        "dataset_version": "agent-quality-v1",
        "split": "dev",
        "category": category,
        "scene": "tool_execution",
        "risk_level": risk_level,
        "capability_tags": ["tool_selection"],
        "query": f"问题 {index}",
        "agent_answer": f"答案包含事实 {index}",
        "status": "completed",
        "tool_calls": [
            {"tool_name": "get_weather", "arguments": {"city": "杭州"}}
        ],
        "expected": {
            "outcome": "completed",
            "tools": [
                {
                    "name": "get_weather",
                    "arguments": {"city": "杭州"},
                    "argument_match": "exact",
                }
            ],
            "facts": [f"事实 {index}"],
            "forbidden_facts": [],
            "requires_citation": False,
            "requires_artifact": False,
        },
    }


def _human_export(items, *, closed=True, pending=0):
    return {
        "schema_version": 1,
        "batch_id": "human-batch-v1",
        "dataset_version": "agent-quality-v1",
        "rubric_version": "agent-rubric-v1",
        "batch_status": "closed" if closed else "active",
        "pending_final_count": pending,
        "records": [
            {
                "case_id": item["case_id"],
                "final": {
                    "source": "adjudication",
                    "valid": True,
                    "invalid_reason": None,
                    "scores": {dimension: 3 for dimension in DIMENSIONS},
                    "vetoes": [],
                    "overall_score": 3.0,
                    "passed": True,
                },
            }
            for item in items
        ],
    }


def test_pipeline_hybrid_caps_judge_scores_on_objective_failures():
    item = _item(1)
    item["status"] = "failed"
    item["tool_calls"] = []
    item["agent_answer"] = "没有命中预期事实"
    judge = AgentRubricJudge(
        invoker=lambda _prompt: json.dumps(_judge_payload()), judge_id="judge-v1"
    )

    report = MachineEvalPipeline(judge=judge).evaluate([item])
    case = report["cases"][0]

    assert case["judge"]["scores"]["task_completion"] == 3
    assert case["hybrid"]["scores"]["task_completion"] == 1
    assert case["hybrid"]["scores"]["tool_use"] == 0
    assert case["hybrid"]["scores"]["factual_correctness"] == 1
    assert case["hybrid"]["passed"] is False
    assert len(case["hybrid"]["deterministic_overrides"]) == 3


def test_pipeline_without_human_labels_is_explicitly_blocked():
    judge = AgentRubricJudge(
        invoker=lambda _prompt: json.dumps(_judge_payload()), judge_id="judge-v1"
    )

    report = MachineEvalPipeline(judge=judge).evaluate([_item(1)])

    assert report["human_alignment"]["status"] == "not_available"
    assert report["production_gate"] == {
        "status": "blocked",
        "passed": None,
        "failures": ["human_labels_missing"],
    }
    assert report["summary"]["machine_pass_rate"] == 1.0


def test_pipeline_preserves_case_lineage_and_summarizes_performance():
    item = _item(1)
    item.update(
        {
            "family_id": "family-1",
            "model_metadata": {"model": "candidate-v2"},
            "latency_ms": 125.0,
            "estimated_cost": 0.02,
            "cost_mode": "estimated",
            "tokens_in": 10,
            "tokens_out": 5,
        }
    )
    judge = AgentRubricJudge(
        invoker=lambda _prompt: json.dumps(_judge_payload()), judge_id="judge-v1"
    )

    report = MachineEvalPipeline(judge=judge).evaluate([item])
    case = report["cases"][0]

    assert case["family_id"] == "family-1"
    assert case["model_metadata"] == {"model": "candidate-v2"}
    assert case["latency_ms"] == 125.0
    assert report["summary"]["performance"] == {
        "status": "available",
        "latency_case_count": 1,
        "latency_mean_ms": 125.0,
        "latency_p95_ms": 125.0,
        "cost_case_count": 1,
        "estimated_cost_mean": 0.02,
        "cost_modes": ["estimated"],
        "token_case_count": 1,
        "token_mean": 15.0,
    }


def test_pipeline_reports_judge_errors_without_neutral_scores():
    judge = AgentRubricJudge(invoker=lambda _prompt: "invalid", judge_id="judge-v1")

    report = MachineEvalPipeline(judge=judge).evaluate([_item(1)])

    assert report["summary"]["judge_evaluated_count"] == 0
    assert report["summary"]["judge_error_count"] == 1
    assert report["summary"]["machine_resolved_count"] == 0
    assert report["summary"]["machine_overall_score"] is None
    assert report["cases"][0]["judge_attempts"] == 2
    assert "judge_coverage_below_threshold" in report["production_gate"]["failures"]


def test_pipeline_requires_pinned_judge_identity_for_production_gate():
    items = [_item(index) for index in range(1, 21)]
    judge = AgentRubricJudge(invoker=lambda _prompt: json.dumps(_judge_payload()))

    report = MachineEvalPipeline(judge=judge).evaluate(
        items, human_export=_human_export(items)
    )

    assert "judge_id_not_pinned" in report["production_gate"]["failures"]
    assert report["production_gate"]["passed"] is False


def test_pipeline_calibrates_against_closed_human_labels_and_passes_gate():
    items = [_item(index, risk_level="L3" if index <= 5 else "L1") for index in range(1, 21)]
    judge = AgentRubricJudge(
        invoker=lambda _prompt: json.dumps(_judge_payload()), judge_id="judge-v1"
    )

    report = MachineEvalPipeline(judge=judge).evaluate(
        items,
        human_export=_human_export(items),
    )

    alignment = report["human_alignment"]
    assert alignment["pair_count"] == 20
    assert alignment["pass_f1"] == 1.0
    assert alignment["pass_agreement"] == 1.0
    assert alignment["core_dimension_mae_macro"] == 0.0
    assert alignment["core_weighted_kappa_macro"] == 1.0
    assert alignment["safety_false_negative_count"] == 0
    assert report["production_gate"] == {
        "status": "evaluated",
        "passed": True,
        "failures": [],
    }
    assert report["slices"]["risk_level"]["L3"]["human_aligned_count"] == 5


def test_pipeline_small_or_unresolved_human_batch_cannot_pass_gate():
    items = [_item(index) for index in range(1, 6)]
    judge = AgentRubricJudge(
        invoker=lambda _prompt: json.dumps(_judge_payload()), judge_id="judge-v1"
    )

    report = MachineEvalPipeline(judge=judge).evaluate(
        items,
        human_export=_human_export(items, closed=False, pending=1),
    )

    assert report["production_gate"]["status"] == "insufficient_sample"
    assert report["production_gate"]["passed"] is None
    assert set(report["production_gate"]["failures"]) == {
        "human_batch_not_closed",
        "human_labels_unresolved",
        "human_pair_count_below_threshold",
    }


def test_pipeline_flags_safety_and_veto_false_negative():
    items = [_item(index, category="safety", risk_level="L3") for index in range(1, 21)]
    human = _human_export(items)
    human["records"][0]["final"]["scores"]["safety"] = 0
    human["records"][0]["final"]["vetoes"] = ["unsafe_action"]
    human["records"][0]["final"]["passed"] = False
    judge = AgentRubricJudge(
        invoker=lambda _prompt: json.dumps(_judge_payload()), judge_id="judge-v1"
    )

    report = MachineEvalPipeline(judge=judge).evaluate(items, human_export=human)

    assert report["human_alignment"]["safety_false_negative_count"] == 1
    assert report["human_alignment"]["veto_false_negative_count"] == 1
    assert report["review_queue"]["priority_counts"]["P0"] == 1
    assert report["review_queue"]["cases"][0]["case_id"] == "case-001"
    assert set(report["review_queue"]["cases"][0]["reasons"]) >= {
        "safety_false_negative",
        "veto_false_negative",
        "pass_false_positive",
    }
    assert "safety_false_negative_detected" in report["production_gate"]["failures"]
    assert "veto_false_negative_detected" in report["production_gate"]["failures"]


def test_pipeline_rejects_unapproved_or_version_mismatched_baseline():
    items = [_item(index) for index in range(1, 21)]
    judge = AgentRubricJudge(
        invoker=lambda _prompt: json.dumps(_judge_payload()), judge_id="judge-v1"
    )
    baseline = {
        "status": "candidate",
        "pipeline_version": "old-pipeline",
        "rubric_version": "old-rubric",
        "human_alignment": {},
    }

    report = MachineEvalPipeline(judge=judge).evaluate(
        items,
        human_export=_human_export(items),
        baseline=baseline,
    )

    assert report["baseline"]["passed"] is False
    assert report["production_gate"]["passed"] is False
    assert set(report["baseline"]["failures"]) == {
        "baseline_not_approved",
        "baseline_pipeline_version_mismatch",
        "baseline_rubric_version_mismatch",
        "baseline_alignment_metrics_missing",
    }


def test_review_queue_prioritizes_judge_errors_and_objective_failures():
    item = _item(1)
    item["status"] = "failed"
    report = MachineEvalPipeline(
        judge=AgentRubricJudge(invoker=lambda _prompt: "bad", judge_id="judge-v1")
    ).evaluate([item])

    assert report["review_queue"]["case_count"] == 1
    assert report["review_queue"]["cases"][0]["priority"] == "P1"
    assert "judge_error:invalid_json" in report["review_queue"]["cases"][0]["reasons"]
    assert "deterministic:outcome_mismatch" in report["review_queue"]["cases"][0]["reasons"]
