from __future__ import annotations

import hashlib
import json
import math
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Mapping


_DECISIONS = {
    "blocked",
    "diagnostic_only",
    "keep_baseline",
    "eligible_for_human_approval",
}


class SQLiteEvaluationAnalysisStore:
    """Immutable, tenant-scoped storage for evaluation analysis reports."""

    def __init__(self, db_path: str | Path) -> None:
        self.db_path = str(db_path)
        directory = os.path.dirname(self.db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._init_db()

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=30.0)
        connection.row_factory = sqlite3.Row
        return connection

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("PRAGMA journal_mode = WAL")
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS evaluation_analysis_reports (
                    tenant_id TEXT NOT NULL,
                    report_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, report_id)
                );

                CREATE INDEX IF NOT EXISTS idx_evaluation_analysis_experiment
                ON evaluation_analysis_reports(tenant_id, experiment_id, generated_at DESC);
                """
            )

    def save_report(self, tenant_id: str, report: Mapping[str, Any]) -> str:
        tenant_id = self._required_string(tenant_id, "tenant_id")
        if not isinstance(report, Mapping):
            raise TypeError("report must be a mapping")

        report_id = self._required_string(report.get("report_id"), "report.report_id")
        experiment = report.get("experiment")
        if not isinstance(experiment, Mapping):
            raise ValueError("report.experiment must be a mapping")
        experiment_id = self._required_string(
            experiment.get("experiment_id"), "report.experiment.experiment_id"
        )
        generated_at = self._normalized_timestamp(report.get("generated_at"))
        self._trend_values(report)

        try:
            payload = json.dumps(
                report,
                ensure_ascii=False,
                sort_keys=True,
                separators=(",", ":"),
                allow_nan=False,
            )
        except (TypeError, ValueError) as exc:
            raise ValueError("report must be finite, JSON-serializable data") from exc
        content_hash = hashlib.sha256(payload.encode("utf-8")).hexdigest()

        with self._connect() as conn:
            conn.execute(
                """
                INSERT OR IGNORE INTO evaluation_analysis_reports(
                    tenant_id, report_id, experiment_id, generated_at,
                    report_json, content_hash
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    tenant_id,
                    report_id,
                    experiment_id,
                    generated_at,
                    payload,
                    content_hash,
                ),
            )
            row = conn.execute(
                """
                SELECT content_hash FROM evaluation_analysis_reports
                WHERE tenant_id = ? AND report_id = ?
                """,
                (tenant_id, report_id),
            ).fetchone()
            if row is None or row["content_hash"] != content_hash:
                raise ValueError(
                    f"report {report_id!r} already exists with different content"
                )
        return report_id

    def get_report(self, tenant_id: str, report_id: str) -> dict[str, Any]:
        tenant_id = self._required_string(tenant_id, "tenant_id")
        report_id = self._required_string(report_id, "report_id")
        with self._connect() as conn:
            row = conn.execute(
                """
                SELECT report_json FROM evaluation_analysis_reports
                WHERE tenant_id = ? AND report_id = ?
                """,
                (tenant_id, report_id),
            ).fetchone()
        if row is None:
            raise KeyError(report_id)
        return json.loads(row["report_json"])

    def list_reports(
        self,
        tenant_id: str,
        experiment_id: str | None = None,
        limit: int = 50,
    ) -> list[dict[str, Any]]:
        tenant_id = self._required_string(tenant_id, "tenant_id")
        limit = self._limit(limit)
        parameters: list[Any] = [tenant_id]
        where = "tenant_id = ?"
        if experiment_id is not None:
            experiment_id = self._required_string(experiment_id, "experiment_id")
            where += " AND experiment_id = ?"
            parameters.append(experiment_id)
        parameters.append(limit)
        with self._connect() as conn:
            rows = conn.execute(
                f"""
                SELECT report_json, content_hash
                FROM evaluation_analysis_reports
                WHERE {where}
                ORDER BY generated_at DESC, report_id DESC
                LIMIT ?
                """,  # noqa: S608 -- only a fixed optional predicate is interpolated
                parameters,
            ).fetchall()
        return [self._summary(row) for row in rows]

    def trend(
        self,
        tenant_id: str,
        experiment_id: str,
        limit: int = 20,
    ) -> dict[str, Any]:
        tenant_id = self._required_string(tenant_id, "tenant_id")
        experiment_id = self._required_string(experiment_id, "experiment_id")
        limit = self._limit(limit)
        with self._connect() as conn:
            rows = conn.execute(
                """
                SELECT report_json, content_hash
                FROM (
                    SELECT report_id, generated_at, report_json, content_hash
                    FROM evaluation_analysis_reports
                    WHERE tenant_id = ? AND experiment_id = ?
                    ORDER BY generated_at DESC, report_id DESC
                    LIMIT ?
                )
                ORDER BY generated_at ASC, report_id ASC
                """,
                (tenant_id, experiment_id, limit),
            ).fetchall()

        runs = [self._summary(row) for row in rows]
        consecutive_regressions = 0
        for run in reversed(runs):
            if run["decision"] != "keep_baseline":
                break
            consecutive_regressions += 1
        return {
            "experiment_id": experiment_id,
            "runs": runs,
            "consecutive_regressions": consecutive_regressions,
            "safety_alert_count": sum(
                run["new_safety_failures"] > 0 for run in runs
            ),
        }

    @classmethod
    def _summary(cls, row: sqlite3.Row) -> dict[str, Any]:
        report = json.loads(row["report_json"])
        decision, pass_delta, score_delta, safety_count = cls._trend_values(report)
        return {
            "report_id": report["report_id"],
            "experiment_id": report["experiment"]["experiment_id"],
            "generated_at": report["generated_at"],
            "decision": decision,
            "pass_rate_delta": pass_delta,
            "overall_score_delta": score_delta,
            "new_safety_failures": safety_count,
            "content_hash": row["content_hash"],
        }

    @staticmethod
    def _trend_values(
        report: Mapping[str, Any]
    ) -> tuple[str, float | None, float | None, int]:
        try:
            decision = report["release_decision"]["status"]
            pass_delta = report["quality_comparison"]["pass_rate"]["delta"]
            score_delta = report["quality_comparison"]["overall_score"]["delta"]
            safety_count = report["safety"]["new_failure_count"]
        except (KeyError, TypeError) as exc:
            raise ValueError("report is missing required trend fields") from exc
        if decision not in _DECISIONS:
            raise ValueError("report.release_decision.status is invalid")
        pass_delta = SQLiteEvaluationAnalysisStore._optional_finite_number(
            pass_delta, "report.quality_comparison.pass_rate.delta"
        )
        score_delta = SQLiteEvaluationAnalysisStore._optional_finite_number(
            score_delta, "report.quality_comparison.overall_score.delta"
        )
        if isinstance(safety_count, bool) or not isinstance(safety_count, int):
            raise ValueError("report.safety.new_failure_count must be a non-negative integer")
        if safety_count < 0:
            raise ValueError("report.safety.new_failure_count must be a non-negative integer")
        return decision, pass_delta, score_delta, safety_count

    @staticmethod
    def _required_string(value: Any, name: str) -> str:
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"{name} must be a non-empty string")
        return value

    @staticmethod
    def _normalized_timestamp(value: Any) -> str:
        value = SQLiteEvaluationAnalysisStore._required_string(
            value, "report.generated_at"
        )
        try:
            timestamp = datetime.fromisoformat(value.replace("Z", "+00:00"))
        except ValueError as exc:
            raise ValueError("report.generated_at must be an ISO-8601 timestamp") from exc
        if timestamp.tzinfo is None:
            raise ValueError("report.generated_at must include a timezone")
        return timestamp.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _finite_number(value: Any, name: str) -> float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{name} must be a finite number")
        value = float(value)
        if not math.isfinite(value):
            raise ValueError(f"{name} must be a finite number")
        return value

    @staticmethod
    def _optional_finite_number(value: Any, name: str) -> float | None:
        if value is None:
            return None
        return SQLiteEvaluationAnalysisStore._finite_number(value, name)

    @staticmethod
    def _limit(value: Any) -> int:
        if isinstance(value, bool) or not isinstance(value, int) or not 1 <= value <= 1000:
            raise ValueError("limit must be an integer between 1 and 1000")
        return value
