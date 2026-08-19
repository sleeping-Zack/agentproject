from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy

import pytest

from services.evaluation_analysis_store import SQLiteEvaluationAnalysisStore


def _report(
    report_id: str = "report-1",
    experiment_id: str = "exp-1",
    generated_at: str = "2026-08-12T10:00:00+08:00",
    decision: str = "eligible_for_human_approval",
    pass_delta: float = 0.1,
    score_delta: float = 0.2,
    new_safety_failures: int = 0,
):
    return {
        "report_id": report_id,
        "experiment": {"experiment_id": experiment_id},
        "generated_at": generated_at,
        "release_decision": {"status": decision},
        "quality_comparison": {
            "pass_rate": {"delta": pass_delta},
            "overall_score": {"delta": score_delta},
        },
        "safety": {"new_failure_count": new_safety_failures},
    }


def test_save_is_canonical_idempotent_and_immutable(tmp_path):
    store = SQLiteEvaluationAnalysisStore(tmp_path / "analysis.db")
    report = _report()

    assert store.save_report("tenant-a", report) == "report-1"
    reordered = {key: report[key] for key in reversed(report)}
    assert store.save_report("tenant-a", reordered) == "report-1"
    assert store.get_report("tenant-a", "report-1") == report

    changed = deepcopy(report)
    changed["quality_comparison"]["pass_rate"]["delta"] = 0.3
    with pytest.raises(ValueError, match="different content"):
        store.save_report("tenant-a", changed)


def test_reports_are_strictly_tenant_scoped(tmp_path):
    store = SQLiteEvaluationAnalysisStore(tmp_path / "analysis.db")
    report = _report()
    store.save_report("tenant-a", report)
    store.save_report("tenant-b", report)

    assert store.get_report("tenant-a", "report-1") == report
    assert len(store.list_reports("tenant-a")) == 1
    assert len(store.list_reports("tenant-b")) == 1
    with pytest.raises(KeyError):
        store.get_report("tenant-c", "report-1")
    assert store.trend("tenant-c", "exp-1")["runs"] == []


def test_list_reports_filters_limits_and_orders_newest_first(tmp_path):
    store = SQLiteEvaluationAnalysisStore(tmp_path / "analysis.db")
    store.save_report("tenant-a", _report("r1", generated_at="2026-08-12T01:00:00Z"))
    store.save_report("tenant-a", _report("r2", generated_at="2026-08-12T03:00:00Z"))
    store.save_report(
        "tenant-a",
        _report("other", experiment_id="exp-2", generated_at="2026-08-12T04:00:00Z"),
    )
    store.save_report("tenant-a", _report("r3", generated_at="2026-08-12T02:00:00Z"))

    rows = store.list_reports("tenant-a", experiment_id="exp-1", limit=2)

    assert [row["report_id"] for row in rows] == ["r2", "r3"]
    assert rows[0]["decision"] == "eligible_for_human_approval"
    assert rows[0]["pass_rate_delta"] == 0.1
    assert len(rows[0]["content_hash"]) == 64


def test_trend_uses_latest_window_in_chronological_order(tmp_path):
    store = SQLiteEvaluationAnalysisStore(tmp_path / "analysis.db")
    reports = [
        _report("r1", generated_at="2026-08-12T01:00:00Z"),
        _report(
            "r2",
            generated_at="2026-08-12T02:00:00Z",
            decision="keep_baseline",
            pass_delta=-0.1,
            score_delta=-0.2,
            new_safety_failures=3,
        ),
        _report(
            "r3",
            generated_at="2026-08-12T03:00:00Z",
            decision="keep_baseline",
            pass_delta=-0.2,
            score_delta=-0.1,
            new_safety_failures=1,
        ),
    ]
    for report in reports:
        store.save_report("tenant-a", report)

    trend = store.trend("tenant-a", "exp-1", limit=2)

    assert [run["report_id"] for run in trend["runs"]] == ["r2", "r3"]
    assert trend["runs"][0]["pass_rate_delta"] == -0.1
    assert trend["runs"][1]["overall_score_delta"] == -0.1
    assert trend["consecutive_regressions"] == 2
    assert trend["safety_alert_count"] == 2


def test_trend_stops_counting_regressions_at_latest_non_rejection(tmp_path):
    store = SQLiteEvaluationAnalysisStore(tmp_path / "analysis.db")
    store.save_report(
        "tenant-a",
        _report("r1", generated_at="2026-08-12T01:00:00Z", decision="keep_baseline"),
    )
    store.save_report(
        "tenant-a",
        _report("r2", generated_at="2026-08-12T02:00:00Z", decision="diagnostic_only"),
    )
    store.save_report(
        "tenant-a",
        _report("r3", generated_at="2026-08-12T03:00:00Z", decision="keep_baseline"),
    )

    assert store.trend("tenant-a", "exp-1")["consecutive_regressions"] == 1


@pytest.mark.parametrize(
    ("tenant_id", "report", "error"),
    [
        ("", _report(), "tenant_id"),
        ("tenant-a", {}, "report.report_id"),
        ("tenant-a", _report(generated_at="2026-08-12T10:00:00"), "timezone"),
        ("tenant-a", _report(decision="promote"), "status"),
        ("tenant-a", _report(pass_delta=float("nan")), "finite number"),
        ("tenant-a", _report(new_safety_failures=-1), "non-negative integer"),
    ],
)
def test_save_rejects_invalid_reports(tmp_path, tenant_id, report, error):
    store = SQLiteEvaluationAnalysisStore(tmp_path / "analysis.db")

    with pytest.raises(ValueError, match=error):
        store.save_report(tenant_id, report)


def test_blocked_report_can_persist_null_comparison_deltas(tmp_path):
    store = SQLiteEvaluationAnalysisStore(tmp_path / "analysis.db")
    report = _report(decision="blocked")
    report["quality_comparison"]["pass_rate"]["delta"] = None
    report["quality_comparison"]["overall_score"]["delta"] = None

    store.save_report("tenant-a", report)

    summary = store.list_reports("tenant-a")[0]
    assert summary["pass_rate_delta"] is None
    assert summary["overall_score_delta"] is None


@pytest.mark.parametrize("limit", [0, -1, True, 1.5, 1001])
def test_list_and_trend_reject_invalid_limits(tmp_path, limit):
    store = SQLiteEvaluationAnalysisStore(tmp_path / "analysis.db")

    with pytest.raises(ValueError, match="limit"):
        store.list_reports("tenant-a", limit=limit)
    with pytest.raises(ValueError, match="limit"):
        store.trend("tenant-a", "exp-1", limit=limit)


def test_concurrent_identical_saves_are_safe(tmp_path):
    store = SQLiteEvaluationAnalysisStore(tmp_path / "analysis.db")
    report = _report()

    with ThreadPoolExecutor(max_workers=8) as pool:
        results = list(pool.map(lambda _: store.save_report("tenant-a", report), range(16)))

    assert results == ["report-1"] * 16
    assert len(store.list_reports("tenant-a")) == 1
