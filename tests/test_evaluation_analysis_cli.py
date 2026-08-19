from copy import deepcopy
import json

import pytest

from evaluation_analysis.service import EvaluationAnalysisService
from scripts.analyze_evaluation_experiment import main
from tests.test_evaluation_analysis_service import _approval, _machine_report


def _write(path, payload):
    path.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")


def test_cli_writes_json_markdown_and_immutable_store(monkeypatch, tmp_path):
    service = EvaluationAnalysisService()
    baseline = _machine_report(split="test")
    candidate = deepcopy(baseline)
    candidate["run_metadata"]["variant"] = "candidate"
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    approval_path = tmp_path / "approval.json"
    report_path = tmp_path / "report.json"
    markdown_path = tmp_path / "report.md"
    store_path = tmp_path / "analysis.db"
    _write(baseline_path, baseline)
    _write(candidate_path, candidate)
    _write(approval_path, _approval(service, baseline))
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_evaluation_experiment.py",
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--experiment-id",
            "cli-exp",
            "--mode",
            "promotion",
            "--hypothesis",
            "candidate is non-inferior",
            "--change",
            "router v2",
            "--baseline-approval",
            str(approval_path),
            "--report-id",
            "cli-report",
            "--output",
            str(report_path),
            "--markdown",
            str(markdown_path),
            "--store-db",
            str(store_path),
            "--tenant",
            "cli-tenant",
            "--gate",
        ],
    )

    main()

    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["release_decision"]["status"] == "eligible_for_human_approval"
    assert "## 9. 发布决策" in markdown_path.read_text(encoding="utf-8")
    assert store_path.exists()


def test_cli_gate_fails_for_diagnostic_only(monkeypatch, tmp_path):
    baseline = _machine_report(split="dev")
    baseline_path = tmp_path / "baseline.json"
    candidate_path = tmp_path / "candidate.json"
    _write(baseline_path, baseline)
    _write(candidate_path, deepcopy(baseline))
    monkeypatch.setattr(
        "sys.argv",
        [
            "analyze_evaluation_experiment.py",
            "--baseline",
            str(baseline_path),
            "--candidate",
            str(candidate_path),
            "--experiment-id",
            "cli-diagnostic",
            "--mode",
            "diagnostic",
            "--hypothesis",
            "locate regressions",
            "--change",
            "router v2",
            "--output",
            str(tmp_path / "report.json"),
            "--gate",
        ],
    )

    with pytest.raises(SystemExit) as exc:
        main()

    assert exc.value.code == 1
