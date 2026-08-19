import json
from pathlib import Path

from evaluation_analysis.reporting import render_markdown, write_markdown


def _report():
    metric = {
        "baseline": 0.7,
        "candidate": 0.8,
        "delta": 0.1,
        "confidence_interval": {"level": 0.95, "lower": 0.02, "upper": 0.18},
        "p_value": 0.03,
    }
    return {
        "schema_version": 1,
        "analysis_version": "agent-eval-analysis-v1",
        "report_id": "report-1",
        "generated_at": "2026-08-12T10:00:00+08:00",
        "analysis_config_sha256": "a" * 64,
        "input_reports": {
            "baseline": {"sha256": "b" * 64, "identity": {"name": "baseline"}},
            "candidate": {"sha256": "c" * 64, "identity": {"name": "candidate"}},
        },
        "experiment": {
            "experiment_id": "exp-1",
            "mode": "diagnostic",
            "hypothesis": "tool routing change improves completion rate",
            "change": "routing-v2",
            "primary_metric": "pass_rate",
        },
        "comparability": {"status": "comparable", "reasons": [], "warnings": []},
        "evaluator_gate": {"status": "passed", "reasons": [], "warnings": []},
        "evidence": {
            "status": "evaluated",
            "reasons": [],
            "warnings": [],
            "pair_count": 40,
        },
        "quality_comparison": {
            "pass_rate": metric,
            "overall_score": metric,
            "dimensions": {"tool_correctness": metric},
        },
        "performance": {
            "p95_latency_ms": metric,
            "average_cost": metric,
            "average_tokens": metric,
        },
        "safety": {
            "new_failure_count": 0,
            "new_veto_count": 0,
            "new_failure_case_ids": [],
        },
        "slices": [
            {
                "name": "scene=tool_call",
                "pair_count": 20,
                "pass_rate_delta": 0.1,
                "overall_score_delta": 0.2,
                "gated": True,
            }
        ],
        "case_transitions": {"improved": 5, "regressed": 1},
        "bad_cases": [
            {
                "case_id": "dev-tool-001",
                "priority": "P1",
                "error_types": ["required_tool_missing"],
                "root_cause": "tool_routing",
                "owner_module": "agent_orchestration",
                "query": "SECRET_QUERY_SENTINEL",
                "answer": "SECRET_ANSWER_SENTINEL",
                "trace": "SECRET_TRACE_SENTINEL",
            }
        ],
        "root_cause_summary": [
            {
                "root_cause": "tool_routing",
                "count": 1,
                "owner_module": "agent_orchestration",
                "case_ids": ["dev-tool-001"],
            }
        ],
        "recommendations": [
            {
                "recommendation_id": "rec-1",
                "priority": "P1",
                "action": "tighten routing instruction",
                "owner_module": "agent_orchestration",
                "affected_case_count": 1,
                "raw_answer": "SECRET_RECOMMENDATION_SENTINEL",
            }
        ],
        "regression_candidates": {
            "proposed": [{"source_case_id": "dev-tool-001"}],
            "excluded": [],
        },
        "release_decision": {
            "status": "eligible_for_human_approval",
            "reasons": ["quality_gate_passed"],
            "requires_human_approval": True,
        },
        "limitations": ["offline evaluation does not measure production drift"],
    }


def test_render_markdown_has_complete_auditable_sections():
    rendered = render_markdown(_report())

    for heading in (
        "## 1. 实验目标",
        "## 2. 可比性",
        "## 3. 三层门禁",
        "## 4. KPI 与置信区间",
        "## 5. 安全与切片",
        "## 6. Top Bad Cases",
        "## 7. 根因归因",
        "## 8. 迭代建议与回归候选",
        "## 9. 发布决策",
        "## 10. 限制",
    ):
        assert heading in rendered
    assert "dev-tool-001" in rendered
    assert "[0.02, 0.18]" in rendered
    assert "可提交人工审批" in rendered


def test_render_markdown_never_emits_prompt_answer_or_trace_payloads():
    rendered = render_markdown(_report())

    for secret in (
        "SECRET_QUERY_SENTINEL",
        "SECRET_ANSWER_SENTINEL",
        "SECRET_TRACE_SENTINEL",
        "SECRET_RECOMMENDATION_SENTINEL",
    ):
        assert secret not in rendered


def test_write_markdown_creates_utf8_report(tmp_path):
    output = write_markdown(_report(), tmp_path / "nested" / "report.md")

    assert output == tmp_path / "nested" / "report.md"
    assert output.read_text(encoding="utf-8") == render_markdown(_report())


def test_report_schema_accepts_minimal_valid_report():
    import jsonschema

    schema = json.loads(
        Path("evals/evaluation_analysis/report_schema_v1.json").read_text(encoding="utf-8")
    )

    jsonschema.Draft202012Validator(schema).validate(_report())


def test_report_schema_rejects_unknown_top_level_fields():
    import jsonschema
    import pytest

    schema = json.loads(
        Path("evals/evaluation_analysis/report_schema_v1.json").read_text(encoding="utf-8")
    )
    report = _report()
    report["raw_cases"] = []

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(report)


def test_report_schema_rejects_promotion_case_details_and_tuning_advice():
    import jsonschema
    import pytest

    schema = json.loads(
        Path("evals/evaluation_analysis/report_schema_v1.json").read_text(encoding="utf-8")
    )
    report = _report()
    report["experiment"]["mode"] = "promotion"

    with pytest.raises(jsonschema.ValidationError):
        jsonschema.Draft202012Validator(schema).validate(report)


def _aggregate_only_promotion_report():
    report = _report()
    report["experiment"]["mode"] = "promotion"
    report["safety"] = {
        "new_failure_count": 0,
        "resolved_failure_count": 1,
        "new_veto_count": 0,
        "resolved_veto_count": 1,
        "new_l3_failure_count": 0,
        "new_p0_count": 0,
        "resolved_p0_count": 0,
    }
    report["case_transitions"] = {"improved": 5, "regressed": 1, "unchanged": 34}
    report["bad_cases"] = []
    report["root_cause_summary"] = []
    report["recommendations"] = []
    report["regression_candidates"] = {"proposed": [], "excluded": []}
    return report


def test_report_schema_accepts_aggregate_only_promotion_report():
    import jsonschema

    schema = json.loads(
        Path("evals/evaluation_analysis/report_schema_v1.json").read_text(encoding="utf-8")
    )

    jsonschema.Draft202012Validator(schema).validate(_aggregate_only_promotion_report())


def test_report_schema_rejects_promotion_case_transition_ids():
    import jsonschema
    import pytest

    schema = json.loads(
        Path("evals/evaluation_analysis/report_schema_v1.json").read_text(encoding="utf-8")
    )
    for forbidden_field in ("improved_case_ids", "regressed_case_ids"):
        report = _aggregate_only_promotion_report()
        report["case_transitions"][forbidden_field] = ["test-secret-case"]
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(schema).validate(report)


def test_report_schema_rejects_all_promotion_safety_case_details():
    import jsonschema
    import pytest

    schema = json.loads(
        Path("evals/evaluation_analysis/report_schema_v1.json").read_text(encoding="utf-8")
    )
    forbidden_details = {
        "new_failure_case_ids": ["test-secret-case"],
        "resolved_failure_case_ids": ["test-secret-case"],
        "new_l3_failure_case_ids": ["test-secret-case"],
        "new_veto_case_ids": ["test-secret-case"],
        "new_vetoes": {"test-secret-case": ["unsafe_action"]},
        "resolved_vetoes": {"test-secret-case": ["unsafe_action"]},
        "new_p0_case_ids": ["test-secret-case"],
        "resolved_p0_case_ids": ["test-secret-case"],
    }
    for forbidden_field, value in forbidden_details.items():
        report = _aggregate_only_promotion_report()
        report["safety"][forbidden_field] = value
        with pytest.raises(jsonschema.ValidationError):
            jsonschema.Draft202012Validator(schema).validate(report)


def test_render_markdown_suppresses_promotion_safety_case_ids_even_without_validation():
    report = _aggregate_only_promotion_report()
    report["safety"]["new_failure_case_ids"] = ["TEST_CASE_SECRET_SENTINEL"]

    rendered = render_markdown(report)

    assert "TEST_CASE_SECRET_SENTINEL" not in rendered
    assert "冻结 test 安全结果仅展示聚合计数" in rendered
