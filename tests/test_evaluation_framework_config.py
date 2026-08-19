import json
from pathlib import Path

import yaml


RUBRIC_PATH = Path("config/evaluation_rubric.yml")
METRICS_PATH = Path("config/evaluation_metrics.yml")
CALIBRATION_PATH = Path("evals/calibration/agent_rubric_v1.jsonl")
ANALYSIS_CONFIG_PATH = Path("config/evaluation_analysis.yml")


def _load_yaml(path: Path):
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def test_evaluation_rubric_is_complete_and_consistent():
    rubric = _load_yaml(RUBRIC_PATH)
    dimensions = rubric["dimensions"]
    dimension_ids = [item["id"] for item in dimensions]

    assert rubric["schema_version"] == 1
    assert len(dimension_ids) == len(set(dimension_ids)) == 7
    assert sum(item["weight"] for item in dimensions) == 100
    assert rubric["scale"]["allowed_scores"] == [0, 1, 2, 3]
    assert all(set(item["anchors"]) == {0, 1, 2, 3} for item in dimensions)
    assert all(item["weight"] > 0 for item in dimensions)

    minimum_scores = rubric["overall"]["pass_conditions"]["minimum_dimension_scores"]
    assert set(minimum_scores).issubset(dimension_ids)
    assert all(score in rubric["scale"]["allowed_scores"] for score in minimum_scores.values())

    veto_ids = [item["id"] for item in rubric["veto_rules"]]
    assert len(veto_ids) == len(set(veto_ids))
    for veto in rubric["veto_rules"]:
        assert veto["description"]
        assert veto["required_evidence"]
        assert set(veto["forces_scores"]).issubset(dimension_ids)

    scenario_ids = [item["id"] for item in rubric["scenarios"]]
    assert len(scenario_ids) == len(set(scenario_ids)) == 7
    assert all(set(item["default_dimensions"]).issubset(dimension_ids) for item in rubric["scenarios"])


def test_evaluation_metrics_reference_the_current_rubric_and_known_dimensions():
    rubric = _load_yaml(RUBRIC_PATH)
    metrics = _load_yaml(METRICS_PATH)
    dimension_ids = {item["id"] for item in rubric["dimensions"]}
    metric_ids = [item["id"] for item in metrics["metrics"]]

    assert metrics["schema_version"] == 1
    assert metrics["rubric_version"] == rubric["rubric_version"]
    assert len(metric_ids) == len(set(metric_ids))
    assert set(next(item for item in metrics["metrics"] if item["id"] == "dimension_mean_score")["dimensions"]) == dimension_ids
    kappa = next(item for item in metrics["metrics"] if item["id"] == "weighted_kappa")
    assert set(kappa["core_dimensions"]).issubset(dimension_ids)

    valid_directions = {"higher_is_better", "lower_is_better", "diagnostic"}
    valid_operators = {"gt", "gte", "lt", "lte", "eq"}
    for metric in metrics["metrics"]:
        assert metric["definition"]
        assert metric["formula"]
        assert metric["direction"] in valid_directions
        if "gate" in metric:
            assert metric["gate"]["operator"] in valid_operators
            assert isinstance(metric["gate"]["threshold"], (int, float))


def test_phase_one_documents_reference_versioned_machine_readable_sources():
    spec = Path("docs/evaluation_spec.md").read_text(encoding="utf-8")
    guideline = Path("docs/annotation_guideline.md").read_text(encoding="utf-8")

    assert "config/evaluation_rubric.yml" in spec
    assert "config/evaluation_metrics.yml" in spec
    assert "agent-rubric-v1" in spec
    assert "agent-rubric-v1" in guideline
    assert "观察—规则—结论" in guideline
    assert "感觉不安全" in guideline


def test_calibration_cases_cover_scenarios_and_match_rubric_scoring():
    rubric = _load_yaml(RUBRIC_PATH)
    dimensions = {item["id"]: item for item in rubric["dimensions"]}
    veto_ids = {item["id"] for item in rubric["veto_rules"]}
    scenario_ids = {item["id"] for item in rubric["scenarios"]}
    capability_tags = set(rubric["capability_tags"])
    risk_levels = set(rubric["risk_levels"])
    required_scores = rubric["overall"]["pass_conditions"]["minimum_dimension_scores"]
    minimum_overall = rubric["overall"]["pass_conditions"]["minimum_overall_score"]
    cases = [
        json.loads(line)
        for line in CALIBRATION_PATH.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]

    assert len(cases) >= 10
    assert len({case["case_id"] for case in cases}) == len(cases)
    assert {case["scene"] for case in cases} == scenario_ids
    assert sum("tool_use" in case["adjudicated"]["scores"] and case["adjudicated"]["scores"]["tool_use"] is not None for case in cases) >= 2
    assert sum(bool(case["adjudicated"]["vetoes"]) for case in cases) >= 2
    assert sum("groundedness" in case["adjudicated"]["scores"] and case["adjudicated"]["scores"]["groundedness"] is not None for case in cases) >= 2

    for case in cases:
        result = case["adjudicated"]
        scores = result["scores"]
        assert case["rubric_version"] == rubric["rubric_version"]
        assert case["risk_level"] in risk_levels
        assert set(case["capability_tags"]).issubset(capability_tags)
        assert set(scores) == set(dimensions)
        assert set(result["vetoes"]).issubset(veto_ids)

        applicable = {key: value for key, value in scores.items() if value is not None}
        assert all(value in rubric["scale"]["allowed_scores"] for value in applicable.values())
        weighted_sum = sum(value * dimensions[key]["weight"] for key, value in applicable.items())
        weight_sum = sum(dimensions[key]["weight"] for key in applicable)
        calculated_overall = round(weighted_sum / weight_sum, rubric["overall"]["precision"])
        assert calculated_overall == result["overall_score"], case["case_id"]

        minimums_met = all(
            scores[dimension_id] is None or scores[dimension_id] >= minimum_score
            for dimension_id, minimum_score in required_scores.items()
        )
        calculated_pass = (
            not result["vetoes"]
            and calculated_overall >= minimum_overall
            and minimums_met
        )
        assert calculated_pass is result["passed"], case["case_id"]


def test_evaluation_analysis_config_separates_diagnostic_and_promotion_gates():
    config = _load_yaml(ANALYSIS_CONFIG_PATH)
    statistics = config["statistics"]
    decision = config["decision_policy"]
    experiment = config["experiment_policy"]

    assert config["schema_version"] == 1
    assert config["rubric_version"] == "agent-rubric-v1"
    assert statistics["minimum_paired_cases"] >= 30
    assert statistics["promotion_bootstrap_iterations"] >= 10000
    assert statistics["diagnostic_bootstrap_iterations"] >= 2000
    assert 0 < statistics["alpha"] < 1
    assert decision["primary_metric"] == "pass_rate"
    assert decision["pass_rate_noninferiority_margin"] == 0.02
    assert decision["allow_new_veto"] is False
    assert decision["allow_new_safety_failure"] is False
    assert decision["allow_p0_regression"] is False
    assert decision["require_performance_evidence_for_promotion"] is True
    assert set(experiment["diagnostic_allowed_splits"]) == {"dev", "regression"}
    assert experiment["promotion_allowed_splits"] == ["test"]
