from copy import deepcopy

from fastapi.testclient import TestClient

import api.server as server
from api.server import app
from evaluation_analysis.service import EvaluationAnalysisService
from services.approval_store import SQLiteApprovalStore
from services.evaluation_analysis_store import SQLiteEvaluationAnalysisStore
from tests.test_evaluation_analysis_service import _machine_report


def _headers(tenant="analysis-api", role="operator"):
    return {
        "X-Tenant-ID": tenant,
        "X-User-Role": role,
        "X-Principal-ID": "evaluation-operator",
    }


def test_evaluation_analysis_api_compares_persists_lists_and_reads_trend(
    monkeypatch, tmp_path
):
    service = EvaluationAnalysisService()
    store = SQLiteEvaluationAnalysisStore(tmp_path / "analysis.db")
    approvals = SQLiteApprovalStore(str(tmp_path / "approvals.db"))
    monkeypatch.setattr(server, "evaluation_analysis_service", service)
    monkeypatch.setattr(server, "evaluation_analysis_store", store)
    monkeypatch.setattr(server, "approval_store", approvals)
    client = TestClient(app)
    baseline = _machine_report(split="test")
    candidate = deepcopy(baseline)
    candidate["run_metadata"]["variant"] = "candidate"
    body = {
        "baseline_report": baseline,
        "candidate_report": candidate,
        "experiment": {
            "experiment_id": "api-exp-1",
            "mode": "promotion",
            "hypothesis": "candidate is non-inferior",
            "change": "router v2",
        },
        "report_id": "api-report-1",
    }

    forbidden = client.post(
        "/evaluation-analysis/compare", headers=_headers(role="user"), json=body
    )
    approval_request = client.post(
        "/evaluation-analysis/baseline-approvals",
        headers=_headers(),
        json={"baseline_report": baseline, "request_id": "approve-api-baseline"},
    )
    assert approval_request.status_code == 200
    approval_id = approval_request.json()["approval_id"]
    approvals.approve(approval_id, decided_by="evaluation-owner")
    body["baseline_approval_id"] = approval_id
    created = client.post(
        "/evaluation-analysis/compare", headers=_headers(), json=body
    )

    assert forbidden.status_code == 403
    assert created.status_code == 200
    assert created.json()["release_decision"]["status"] == "eligible_for_human_approval"
    listed = client.get("/evaluation-analysis/reports", headers=_headers())
    assert [row["report_id"] for row in listed.json()["reports"]] == ["api-report-1"]
    fetched = client.get(
        "/evaluation-analysis/reports/api-report-1", headers=_headers()
    )
    assert fetched.json()["report_id"] == "api-report-1"
    trend = client.get(
        "/evaluation-analysis/experiments/api-exp-1/trend", headers=_headers()
    )
    assert trend.json()["runs"][0]["decision"] == "eligible_for_human_approval"


def test_evaluation_analysis_api_rejects_untrusted_or_cross_tenant_approval(
    monkeypatch, tmp_path
):
    service = EvaluationAnalysisService()
    approvals = SQLiteApprovalStore(str(tmp_path / "approvals.db"))
    monkeypatch.setattr(server, "evaluation_analysis_service", service)
    monkeypatch.setattr(server, "approval_store", approvals)
    monkeypatch.setattr(
        server,
        "evaluation_analysis_store",
        SQLiteEvaluationAnalysisStore(tmp_path / "analysis.db"),
    )
    baseline = _machine_report(split="test")
    candidate = deepcopy(baseline)
    candidate["run_metadata"]["variant"] = "candidate"
    pending = approvals.create_pending(
        request_id="wrong-tenant-baseline",
        tenant_id="tenant-b",
        user_role="operator",
        tool_name="evaluation_analysis_baseline",
        args={"report_sha256": service.report_sha256(baseline)},
        reason="tenant scoped approval",
    )
    approvals.approve(pending.approval_id, decided_by="operator-b")
    body = {
        "baseline_report": baseline,
        "candidate_report": candidate,
        "experiment": {
            "experiment_id": "api-untrusted-approval",
            "mode": "promotion",
            "hypothesis": "candidate is non-inferior",
            "change": "router v2",
        },
        "baseline_approval_id": pending.approval_id,
    }

    response = TestClient(app).post(
        "/evaluation-analysis/compare", headers=_headers("analysis-api"), json=body
    )

    assert response.status_code == 404


def test_evaluation_baseline_approval_request_rejects_untrusted_report(
    monkeypatch, tmp_path
):
    service = EvaluationAnalysisService()
    baseline = _machine_report(split="test")
    baseline["config_sha256"] = "a" * 64
    monkeypatch.setattr(server, "evaluation_analysis_service", service)
    monkeypatch.setattr(
        server, "approval_store", SQLiteApprovalStore(str(tmp_path / "approvals.db"))
    )

    response = TestClient(app).post(
        "/evaluation-analysis/baseline-approvals",
        headers=_headers(),
        json={"baseline_report": baseline},
    )

    assert response.status_code == 400
    assert "lineage is untrusted" in response.json()["detail"]


def test_evaluation_analysis_api_is_tenant_isolated(monkeypatch, tmp_path):
    store = SQLiteEvaluationAnalysisStore(tmp_path / "analysis.db")
    monkeypatch.setattr(server, "evaluation_analysis_store", store)
    report = {
        "report_id": "tenant-report",
        "experiment": {"experiment_id": "tenant-exp"},
        "generated_at": "2026-08-12T10:00:00+08:00",
        "release_decision": {"status": "diagnostic_only"},
        "quality_comparison": {
            "pass_rate": {"delta": 0.0},
            "overall_score": {"delta": 0.0},
        },
        "safety": {"new_failure_count": 0},
    }
    store.save_report("tenant-a", report)
    client = TestClient(app)

    assert (
        client.get(
            "/evaluation-analysis/reports/tenant-report", headers=_headers("tenant-b")
        ).status_code
        == 404
    )
    assert client.get(
        "/evaluation-analysis/reports", headers=_headers("tenant-b")
    ).json() == {"reports": []}


def test_evaluation_analysis_api_rejects_oversized_case_payload(monkeypatch, tmp_path):
    monkeypatch.setattr(
        server,
        "evaluation_analysis_store",
        SQLiteEvaluationAnalysisStore(tmp_path / "analysis.db"),
    )
    report = _machine_report(split="dev", count=1)
    report["cases"] = report["cases"] * 501
    client = TestClient(app)

    response = client.post(
        "/evaluation-analysis/compare",
        headers=_headers(),
        json={
            "baseline_report": report,
            "candidate_report": report,
            "experiment": {
                "experiment_id": "oversized",
                "mode": "diagnostic",
                "hypothesis": "test request bound",
                "change": "none",
            },
        },
    )

    assert response.status_code == 400
    assert "between 1 and 500" in response.json()["detail"]
