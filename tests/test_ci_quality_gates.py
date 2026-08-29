import json

from rag.eval_gate import EvalGate, EvalThresholds
from scripts.evaluate_agent import CaseResult, _summarize_results
from scripts.evaluate_generation import evaluate_generation_gate
from scripts.evaluate_retrieval import evaluate_retrieval_gate
from utils.evaluation_gate_config import load_gate_profile


def test_retrieval_fixture_policy_has_meaningful_floor_and_zero_regression():
    policy = load_gate_profile(
        "config/ci_quality_gates.yml", "retrieval", "offline_fixture"
    )
    baseline = json.loads(
        open("evals/baselines/retrieval_baseline_v1.json", encoding="utf-8").read()
    )

    assert policy["metrics"]["precision_at_k"]["minimum"] == 0.30
    assert policy["minimum_case_count"] == 30
    assert set(baseline["allowed_regression"].values()) == {0.0}


def test_retrieval_gate_checks_precision_floor_and_required_baseline():
    policy = load_gate_profile(
        "config/ci_quality_gates.yml", "retrieval", "offline_fixture"
    )
    report = {
        "hybrid": {
            "case_count": 30,
            "recall_at_k": 0.95,
            "precision_at_k": 0.29,
            "mrr": 0.95,
            "ndcg_at_k": 0.95,
            "hit_rate": 0.95,
        }
    }

    gate = evaluate_retrieval_gate(
        report,
        strategy_name="hybrid",
        policy=policy,
        baseline_result=None,
    )

    assert gate["passed"] is False
    assert "precision_at_k_below_threshold" in gate["failures"]
    assert "baseline_required" in gate["failures"]


def test_generation_online_policy_allows_soft_failures_but_not_critical_failure():
    policy = load_gate_profile(
        "config/ci_quality_gates.yml", "generation", "online"
    )
    summary = {
        "case_count": 100,
        "pass_rate": 0.90,
        "fact_coverage": 0.90,
        "citation_validity": 0.98,
        "unsupported_claim_rate": 0.05,
        "critical_case_pass_rate": 0.99,
        "critical_citation_validity": 1.0,
        "refusal_accuracy": 0.95,
        "forbidden_hit_rate": 0.0,
        "harmful_instruction_rate": 0.0,
        "judge_error_rate": 0.02,
        "gate_classes": {
            "quality": {"case_count": 50},
            "safety": {"case_count": 15},
            "grounding": {"case_count": 20},
            "refusal": {"case_count": 15},
        },
    }

    gate = evaluate_generation_gate(summary, policy=policy, baseline=None)

    assert gate["passed"] is False
    assert gate["minimums"]["pass_rate"] == 0.90
    assert "critical_case_pass_rate_below_threshold" in gate["failures"]


def test_agent_averages_only_tool_applicable_cases():
    results = [
        CaseResult(
            id="tool-case",
            passed=False,
            tool_recall=0.0,
            keyword_recall=1.0,
            rejected=False,
            tool_applicable=True,
        ),
        *[
            CaseResult(
                id=f"non-tool-{index}",
                passed=True,
                tool_recall=1.0,
                keyword_recall=1.0,
                rejected=False,
            )
            for index in range(9)
        ],
    ]

    summary = _summarize_results(results)

    assert summary["tool_case_count"] == 1
    assert summary["tool_recall"] == 0.0


def test_agent_high_risk_tool_miss_cannot_hide_in_overall_average():
    gate = EvalGate(
        EvalThresholds(
            min_pass_rate=0.90,
            min_tool_recall=0.90,
            min_keyword_recall=0.80,
            min_high_risk_tool_recall=1.0,
        )
    )
    report = {
        "aggregate": {
            "case_count": 100,
            "pass_rate": 0.99,
            "tool_recall": 0.95,
            "keyword_recall": 0.90,
        },
        "risk_tiers": {
            "high": {"case_count": 20, "pass_rate": 0.95, "tool_recall": 0.90},
            "standard": {"case_count": 80, "pass_rate": 1.0, "tool_recall": 1.0},
        },
        "cost": {"avg": 0.01},
        "cases": [],
    }

    result = gate.evaluate(report)

    assert result.passed is False
    assert "high_risk_tool_recall_below_threshold" in result.failures
