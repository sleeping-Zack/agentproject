import json
from pathlib import Path

import yaml

from machine_eval.pipeline import MachineEvalPipeline


def test_machine_eval_config_versions_and_thresholds_are_coherent():
    config = yaml.safe_load(
        Path("config/machine_evaluation.yml").read_text(encoding="utf-8")
    )

    assert config["pipeline_version"] == "agent-machine-eval-v1"
    assert config["rubric_version"] == "agent-rubric-v1"
    assert config["judge"]["maximum_attempts"] >= 1
    assert config["judge"]["maximum_workers"] >= 1
    assert 0 <= config["judge"]["maximum_error_rate"] <= 1
    gate = config["calibration_gate"]
    assert gate["minimum_human_pairs"] >= 20
    assert 0 <= gate["minimum_judge_coverage"] <= 1
    assert 0 <= gate["minimum_pass_f1"] <= 1
    assert gate["maximum_safety_false_negatives"] == 0
    assert gate["maximum_veto_false_negatives"] == 0


def test_machine_eval_smoke_fixture_matches_dataset_and_passes_deterministic_checks():
    dataset = {}
    for path in Path("evals/agent_quality/v1").glob("*.jsonl"):
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                case = json.loads(line)
                dataset[case["case_id"]] = case
    results = [
        json.loads(line)
        for line in Path(
            "evals/fixtures/agent_quality_run_results_smoke_v1.jsonl"
        ).read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    from scripts.evaluate_agent_quality import build_evaluation_items

    items = build_evaluation_items(dataset, results, splits=["dev"])
    report = MachineEvalPipeline().evaluate(items)

    assert report["summary"]["case_count"] == 2
    assert report["summary"]["deterministic_pass_rate"] == 1.0
    assert report["production_gate"]["failures"] == ["judge_not_run"]


def test_machine_eval_report_schema_accepts_a_real_candidate_report():
    import jsonschema

    schema = json.loads(
        Path("evals/machine_eval/report_schema_v1.json").read_text(encoding="utf-8")
    )
    item = {
        "case_id": "schema-case",
        "agent_answer": "测试事实",
        "status": "completed",
        "expected": {
            "outcome": "completed",
            "tools": [],
            "facts": ["测试事实"],
            "forbidden_facts": [],
            "requires_citation": False,
            "requires_artifact": False,
        },
    }
    report = MachineEvalPipeline().evaluate([item])

    jsonschema.Draft202012Validator(schema).validate(report)
