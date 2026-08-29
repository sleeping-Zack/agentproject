from __future__ import annotations

import re
from collections.abc import Iterator, Mapping
from typing import Any

from services.evaluation_analysis_store import SQLiteEvaluationAnalysisStore
from services.human_eval_store import SQLiteHumanEvalStore


class _CompatRow(Mapping[str, Any]):
    """Expose psycopg dict rows with sqlite3.Row-style integer access."""

    def __init__(self, values: Mapping[str, Any]) -> None:
        self._values = dict(values)

    def __getitem__(self, key: str | int) -> Any:
        if isinstance(key, int):
            return tuple(self._values.values())[key]
        return self._values[key]

    def __iter__(self) -> Iterator[str]:
        return iter(self._values)

    def __len__(self) -> int:
        return len(self._values)


class _CompatCursor:
    def __init__(self, cursor) -> None:
        self._cursor = cursor

    @property
    def rowcount(self) -> int:
        return self._cursor.rowcount

    def fetchone(self) -> _CompatRow | None:
        row = self._cursor.fetchone()
        return None if row is None else _CompatRow(row)

    def fetchall(self) -> list[_CompatRow]:
        return [_CompatRow(row) for row in self._cursor.fetchall()]

    def __iter__(self) -> Iterator[_CompatRow]:
        return (_CompatRow(row) for row in self._cursor)


class _PostgresCompatConnection:
    """Small DB-API adapter for the ANSI SQL used by the evaluation stores."""

    def __init__(self, database_url: str) -> None:
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - production dependency
            raise RuntimeError(
                "Postgres evaluation backends require the 'production' dependency extra"
            ) from exc
        self._connection = psycopg.connect(database_url, row_factory=dict_row)

    def __enter__(self) -> _PostgresCompatConnection:
        self._connection.__enter__()
        return self

    def __exit__(self, exc_type, exc_value, traceback) -> bool | None:
        return self._connection.__exit__(exc_type, exc_value, traceback)

    def execute(self, statement: str, parameters=None) -> _CompatCursor:
        sql = self._translate(statement)
        cursor = self._connection.execute(sql, parameters or ())
        return _CompatCursor(cursor)

    @staticmethod
    def _translate(statement: str) -> str:
        if statement.strip().upper() == "BEGIN IMMEDIATE":
            return "SELECT pg_advisory_xact_lock(hashtext('sweeper_human_eval_write'))"
        translated, replacements = re.subn(
            r"\bINSERT\s+OR\s+IGNORE\s+INTO\b",
            "INSERT INTO",
            statement,
            flags=re.IGNORECASE,
        )
        translated = translated.replace("?", "%s")
        if replacements:
            translated = translated.rstrip().rstrip(";") + " ON CONFLICT DO NOTHING"
        return translated


class _PostgresEvaluationBackend:
    shared = True

    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise ValueError("database_url is required")
        self.database_url = database_url
        self._init_db()

    def _connect(self) -> _PostgresCompatConnection:
        return _PostgresCompatConnection(self.database_url)


class PostgresHumanEvalStore(_PostgresEvaluationBackend, SQLiteHumanEvalStore):
    def _init_db(self) -> None:
        statements = (
            """
            CREATE TABLE IF NOT EXISTS human_eval_batches (
                batch_id TEXT PRIMARY KEY,
                tenant_id TEXT NOT NULL,
                name TEXT NOT NULL,
                dataset_version TEXT NOT NULL,
                rubric_version TEXT NOT NULL,
                status TEXT NOT NULL,
                assignments_per_item INTEGER NOT NULL,
                qc_rate DOUBLE PRECISION NOT NULL,
                seed INTEGER NOT NULL,
                created_by TEXT NOT NULL,
                created_at TEXT NOT NULL,
                closed_at TEXT
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS human_eval_items (
                item_id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL REFERENCES human_eval_batches(batch_id),
                case_id TEXT NOT NULL,
                blind_payload TEXT NOT NULL,
                oracle_payload TEXT NOT NULL,
                ordinal INTEGER NOT NULL,
                qc_selected INTEGER NOT NULL,
                qc_status TEXT NOT NULL,
                UNIQUE(batch_id, case_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS human_eval_assignments (
                assignment_id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL REFERENCES human_eval_batches(batch_id),
                item_id TEXT NOT NULL REFERENCES human_eval_items(item_id),
                reviewer_id TEXT NOT NULL,
                reviewer_slot INTEGER NOT NULL,
                blind_key TEXT NOT NULL UNIQUE,
                status TEXT NOT NULL,
                revision INTEGER NOT NULL DEFAULT 0,
                assigned_at TEXT NOT NULL,
                claimed_at TEXT,
                submitted_at TEXT,
                UNIQUE(item_id, reviewer_id)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS human_eval_annotations (
                annotation_id TEXT PRIMARY KEY,
                assignment_id TEXT NOT NULL REFERENCES human_eval_assignments(assignment_id),
                revision INTEGER NOT NULL,
                payload TEXT NOT NULL,
                overall_score DOUBLE PRECISION,
                passed INTEGER,
                is_current INTEGER NOT NULL,
                submitted_at TEXT NOT NULL,
                UNIQUE(assignment_id, revision)
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS human_eval_qc_reviews (
                qc_review_id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL REFERENCES human_eval_batches(batch_id),
                item_id TEXT NOT NULL REFERENCES human_eval_items(item_id),
                reviewer_id TEXT NOT NULL,
                decision TEXT NOT NULL,
                note TEXT NOT NULL,
                returned_assignments TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS human_eval_adjudications (
                adjudication_id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL REFERENCES human_eval_batches(batch_id),
                item_id TEXT NOT NULL UNIQUE REFERENCES human_eval_items(item_id),
                adjudicator_id TEXT NOT NULL,
                payload TEXT NOT NULL,
                overall_score DOUBLE PRECISION,
                passed INTEGER,
                triggers TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE TABLE IF NOT EXISTS human_eval_audit_events (
                event_id TEXT PRIMARY KEY,
                batch_id TEXT NOT NULL REFERENCES human_eval_batches(batch_id),
                item_id TEXT REFERENCES human_eval_items(item_id),
                actor_id TEXT NOT NULL,
                event_type TEXT NOT NULL,
                payload TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_human_eval_assignment_queue
            ON human_eval_assignments(batch_id, reviewer_id, status, assigned_at)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_human_eval_annotation_current
            ON human_eval_annotations(assignment_id, is_current)
            """,
            """
            CREATE INDEX IF NOT EXISTS idx_human_eval_items_qc
            ON human_eval_items(batch_id, qc_selected, qc_status)
            """,
        )
        with self._connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(hashtext('sweeper_human_eval_schema'))")
            for statement in statements:
                conn.execute(statement)


class PostgresEvaluationAnalysisStore(
    _PostgresEvaluationBackend,
    SQLiteEvaluationAnalysisStore,
):
    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(hashtext('sweeper_evaluation_schema'))")
            conn.execute(
                """
                CREATE TABLE IF NOT EXISTS evaluation_analysis_reports (
                    tenant_id TEXT NOT NULL,
                    report_id TEXT NOT NULL,
                    experiment_id TEXT NOT NULL,
                    generated_at TEXT NOT NULL,
                    report_json TEXT NOT NULL,
                    content_hash TEXT NOT NULL,
                    PRIMARY KEY (tenant_id, report_id)
                )
                """
            )
            conn.execute(
                """
                CREATE INDEX IF NOT EXISTS idx_evaluation_analysis_experiment
                ON evaluation_analysis_reports(
                    tenant_id, experiment_id, generated_at DESC
                )
                """
            )
