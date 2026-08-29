from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

from dotenv import load_dotenv
from psycopg import sql
from psycopg.rows import dict_row
from psycopg.types.json import Jsonb

from services.memory_store import PostgresMemoryStore
from services.postgres import PostgresApprovalStore, PostgresArtifactStore, PostgresStore
from services.postgres_evaluation import (
    PostgresEvaluationAnalysisStore,
    PostgresHumanEvalStore,
)


AGENT_TABLES = (
    "session_messages",
    "traces",
    "memory_events",
    "memory_facts",
    "memory_tombstones",
    "memory_summaries",
    "memory_access_log",
    "procedural_memories",
)
HUMAN_EVAL_TABLES = (
    "human_eval_batches",
    "human_eval_items",
    "human_eval_assignments",
    "human_eval_annotations",
    "human_eval_qc_reviews",
    "human_eval_adjudications",
    "human_eval_audit_events",
)

JSON_COLUMNS = {
    ("traces", "payload"),
    ("approvals", "args"),
    ("artifacts", "payload"),
    ("artifacts", "metadata"),
    ("memory_events", "metadata"),
    ("memory_facts", "metadata"),
    ("memory_summaries", "source_message_ids"),
    ("procedural_memories", "evidence"),
}
BOOLEAN_COLUMNS = {
    ("memory_facts", "explicit"),
    ("memory_access_log", "adopted"),
}
TIMESTAMP_COLUMNS = {
    ("session_messages", "created_at"),
    ("traces", "created_at"),
    ("memory_events", "created_at"),
    ("memory_facts", "created_at"),
    ("memory_facts", "updated_at"),
    ("memory_facts", "last_confirmed_at"),
    ("memory_facts", "valid_from"),
    ("memory_facts", "valid_to"),
    ("memory_tombstones", "created_at"),
    ("memory_summaries", "updated_at"),
    ("memory_access_log", "created_at"),
    ("procedural_memories", "created_at"),
    ("procedural_memories", "approved_at"),
}
SEQUENCES = {
    "session_messages": "id",
    "memory_access_log": "id",
}


def _source_groups() -> tuple[tuple[Path, tuple[str, ...]], ...]:
    return (
        (Path(os.getenv("AGENT_DB_PATH", "storage/agent.db")), AGENT_TABLES),
        (
            Path(os.getenv("AGENT_APPROVAL_DB_PATH", "storage/approvals.db")),
            ("approvals",),
        ),
        (
            Path(os.getenv("AGENT_ARTIFACT_DB_PATH", "storage/artifacts.db")),
            ("artifacts",),
        ),
        (
            Path(os.getenv("AGENT_HUMAN_EVAL_DB_PATH", "storage/human_eval.db")),
            HUMAN_EVAL_TABLES,
        ),
        (
            Path(
                os.getenv(
                    "AGENT_EVALUATION_ANALYSIS_DB_PATH",
                    "storage/evaluation_analysis.db",
                )
            ),
            ("evaluation_analysis_reports",),
        ),
    )


def _initialize_postgres(database_url: str) -> None:
    PostgresStore(database_url)
    PostgresMemoryStore(database_url)
    PostgresApprovalStore(database_url)
    PostgresArtifactStore(database_url)
    PostgresHumanEvalStore(database_url)
    PostgresEvaluationAnalysisStore(database_url)


def _sqlite_tables(connection: sqlite3.Connection) -> set[str]:
    return {
        row[0] for row in connection.execute("SELECT name FROM sqlite_master WHERE type = 'table'")
    }


def _source_columns(connection: sqlite3.Connection, table: str) -> tuple[list[str], list[str]]:
    rows = connection.execute(f'PRAGMA table_info("{table}")').fetchall()
    columns = [row[1] for row in rows]
    primary_key = [
        row[1] for row in sorted((row for row in rows if row[5]), key=lambda row: row[5])
    ]
    if not primary_key:
        raise RuntimeError(f"SQLite table {table} has no primary key")
    return columns, primary_key


def _target_columns(connection, table: str) -> set[str]:
    rows = connection.execute(
        "SELECT column_name FROM information_schema.columns "
        "WHERE table_schema = current_schema() AND table_name = %s",
        (table,),
    ).fetchall()
    return {row["column_name"] for row in rows}


def _adapt_value(table: str, column: str, value: Any) -> Any:
    if value is None:
        return None
    if (table, column) in JSON_COLUMNS:
        return Jsonb(json.loads(value) if isinstance(value, str) else value)
    if (table, column) in BOOLEAN_COLUMNS:
        return bool(value)
    return value


def _normalized_value(table: str, column: str, value: Any) -> Any:
    if value is None:
        return None
    if (table, column) in JSON_COLUMNS:
        decoded = json.loads(value) if isinstance(value, str) else value
        return json.dumps(decoded, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if (table, column) in BOOLEAN_COLUMNS:
        return bool(value)
    if (table, column) in TIMESTAMP_COLUMNS:
        if isinstance(value, datetime):
            timestamp = value
        else:
            timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
        if timestamp.tzinfo is None:
            timestamp = timestamp.replace(tzinfo=timezone.utc)
        return timestamp.astimezone(timezone.utc).isoformat()
    return value


def _read_rows(
    connection: sqlite3.Connection,
    table: str,
    columns: Iterable[str],
) -> list[dict[str, Any]]:
    identifiers = ", ".join(f'"{column}"' for column in columns)
    rows = connection.execute(f'SELECT {identifiers} FROM "{table}"').fetchall()
    return [dict(row) for row in rows]


def _prepare_rows(table: str, rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if table != "approvals":
        return rows
    prepared = []
    for source_row in rows:
        row = dict(source_row)
        if row.get("principal_id", "") == "" and row.get("status") == "pending":
            row["status"] = "denied"
            row["decided_at"] = row.get("decided_at") or row["created_at"]
            row["decided_by"] = "system:legacy-scope-migration"
        prepared.append(row)
    return prepared


def _insert_rows(connection, table: str, columns: list[str], rows: list[dict]) -> None:
    if not rows:
        return
    statement = sql.SQL("INSERT INTO {} ({}) VALUES ({}) ON CONFLICT DO NOTHING").format(
        sql.Identifier(table),
        sql.SQL(", ").join(map(sql.Identifier, columns)),
        sql.SQL(", ").join(sql.Placeholder() for _ in columns),
    )
    values = [tuple(_adapt_value(table, column, row[column]) for column in columns) for row in rows]
    with connection.cursor() as cursor:
        cursor.executemany(statement, values)


def _normalize_legacy_approvals(
    connection,
    source_rows: list[dict[str, Any]],
) -> None:
    legacy_rows = [
        row
        for row in source_rows
        if row.get("principal_id", "") == "" and row.get("status") == "pending"
    ]
    if not legacy_rows:
        return
    with connection.cursor() as cursor:
        cursor.executemany(
            "UPDATE approvals SET status = 'denied', decided_at = %s, "
            "decided_by = 'system:legacy-scope-migration' "
            "WHERE approval_id = %s AND principal_id = ''",
            [
                (row.get("decided_at") or row["created_at"], row["approval_id"])
                for row in legacy_rows
            ],
        )


def _verify_rows(
    connection,
    table: str,
    columns: list[str],
    primary_key: list[str],
    source_rows: list[dict[str, Any]],
) -> None:
    statement = sql.SQL("SELECT {} FROM {}").format(
        sql.SQL(", ").join(map(sql.Identifier, columns)),
        sql.Identifier(table),
    )
    target_rows = connection.execute(statement).fetchall()
    target_by_key = {
        tuple(_normalized_value(table, key, row[key]) for key in primary_key): row
        for row in target_rows
    }
    for source_row in source_rows:
        key = tuple(_normalized_value(table, column, source_row[column]) for column in primary_key)
        target_row = target_by_key.get(key)
        if target_row is None:
            raise RuntimeError(f"migration verification failed: missing {table} row {key}")
        for column in columns:
            source_value = _normalized_value(table, column, source_row[column])
            target_value = _normalized_value(table, column, target_row[column])
            if source_value != target_value:
                raise RuntimeError(f"migration verification failed: {table}{key}.{column} differs")


def _reset_sequences(connection) -> None:
    for table, column in SEQUENCES.items():
        statement = sql.SQL(
            "SELECT setval(pg_get_serial_sequence({}, {}), "
            "COALESCE(MAX({}), 1), MAX({}) IS NOT NULL) FROM {}"
        ).format(
            sql.Literal(table),
            sql.Literal(column),
            sql.Identifier(column),
            sql.Identifier(column),
            sql.Identifier(table),
        )
        connection.execute(statement)


def migrate(database_url: str) -> dict[str, int]:
    try:
        import psycopg
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError("migration requires the 'production' dependency extra") from exc

    _initialize_postgres(database_url)
    summary: dict[str, int] = {}
    with psycopg.connect(database_url, row_factory=dict_row) as target:
        target.execute("SET TIME ZONE 'UTC'")
        for path, requested_tables in _source_groups():
            if not path.is_file():
                continue
            with sqlite3.connect(path) as source:
                source.row_factory = sqlite3.Row
                available_tables = _sqlite_tables(source)
                for table in requested_tables:
                    if table not in available_tables:
                        continue
                    source_columns, primary_key = _source_columns(source, table)
                    columns = [
                        column
                        for column in source_columns
                        if column in _target_columns(target, table)
                    ]
                    if not set(primary_key) <= set(columns):
                        raise RuntimeError(f"target table {table} is missing its source key")
                    source_rows = _read_rows(source, table, columns)
                    rows = _prepare_rows(table, source_rows)
                    _insert_rows(target, table, columns, rows)
                    if table == "approvals":
                        _normalize_legacy_approvals(target, source_rows)
                    _verify_rows(target, table, columns, primary_key, rows)
                    summary[table] = len(rows)
        _reset_sequences(target)
    return summary


def main() -> None:
    load_dotenv()
    database_url = os.getenv(
        "AGENT_DATABASE_URL",
        "postgresql://agent:agent@127.0.0.1:55432/agent",
    )
    summary = migrate(database_url)
    print(json.dumps({"verified_rows": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
