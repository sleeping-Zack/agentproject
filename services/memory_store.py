from __future__ import annotations

import json
import os
import sqlite3
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional
from uuid import uuid4

from agent.long_term_memory import (
    MemoryCategory,
    MemoryRecord,
    ProcedureMemory,
    stable_value_hash,
)


def _iso(value: datetime) -> str:
    return value.astimezone(timezone.utc).isoformat()


def _dt(value: Optional[str]) -> Optional[datetime]:
    return datetime.fromisoformat(value) if value else None


class SQLiteMemoryStore:
    shared = True

    _FACT_COLUMNS = (
        "memory_id, tenant_id, user_id, memory_key, value, category, status, version, "
        "importance, confidence, reinforcement, explicit, created_at, updated_at, "
        "last_confirmed_at, valid_from, valid_to, supersedes_id, source_event_id, metadata"
    )

    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._init_db()

    def _connect(self):
        conn = sqlite3.connect(self.db_path)
        conn.row_factory = sqlite3.Row
        return conn

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.executescript(
                """
                CREATE TABLE IF NOT EXISTS memory_events (
                    event_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    request_id TEXT NOT NULL,
                    kind TEXT NOT NULL,
                    content TEXT NOT NULL,
                    metadata TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    UNIQUE(tenant_id, user_id, request_id, kind)
                );
                CREATE TABLE IF NOT EXISTS memory_facts (
                    memory_id TEXT PRIMARY KEY,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    value TEXT NOT NULL,
                    category TEXT NOT NULL,
                    status TEXT NOT NULL,
                    version INTEGER NOT NULL,
                    importance REAL NOT NULL,
                    confidence REAL NOT NULL,
                    reinforcement REAL NOT NULL,
                    explicit INTEGER NOT NULL,
                    created_at TEXT NOT NULL,
                    updated_at TEXT NOT NULL,
                    last_confirmed_at TEXT NOT NULL,
                    valid_from TEXT NOT NULL,
                    valid_to TEXT,
                    supersedes_id TEXT,
                    source_event_id TEXT,
                    metadata TEXT NOT NULL DEFAULT '{}'
                );
                CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_fact_active
                    ON memory_facts(tenant_id, user_id, memory_key)
                    WHERE status = 'active';
                CREATE INDEX IF NOT EXISTS idx_memory_fact_owner
                    ON memory_facts(tenant_id, user_id, status);
                CREATE TABLE IF NOT EXISTS memory_tombstones (
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    memory_key TEXT NOT NULL,
                    value_hash TEXT NOT NULL,
                    created_at TEXT NOT NULL,
                    PRIMARY KEY(tenant_id, user_id, memory_key, value_hash)
                );
                CREATE TABLE IF NOT EXISTS memory_summaries (
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    session_id TEXT NOT NULL,
                    summary TEXT NOT NULL,
                    covered_message_count INTEGER NOT NULL,
                    version TEXT NOT NULL,
                    source_message_ids TEXT NOT NULL DEFAULT '[]',
                    source_digest TEXT NOT NULL DEFAULT '',
                    updated_at TEXT NOT NULL,
                    PRIMARY KEY(tenant_id, user_id, session_id)
                );
                CREATE TABLE IF NOT EXISTS memory_access_log (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    memory_id TEXT NOT NULL,
                    tenant_id TEXT NOT NULL,
                    user_id TEXT NOT NULL,
                    score REAL NOT NULL,
                    adopted INTEGER NOT NULL DEFAULT 0,
                    created_at TEXT NOT NULL
                );
                CREATE TABLE IF NOT EXISTS procedural_memories (
                    procedure_id TEXT PRIMARY KEY,
                    tenant_id TEXT,
                    agent_version TEXT NOT NULL,
                    status TEXT NOT NULL,
                    title TEXT NOT NULL,
                    content TEXT NOT NULL,
                    evidence TEXT NOT NULL DEFAULT '{}',
                    created_at TEXT NOT NULL,
                    approved_at TEXT
                );
                """
            )
            self._migrate_summaries(conn)

    @staticmethod
    def _migrate_summaries(conn) -> None:
        columns = {
            row["name"] for row in conn.execute("PRAGMA table_info(memory_summaries)")
        }
        if "user_id" not in columns:
            conn.execute("ALTER TABLE memory_summaries RENAME TO memory_summaries_legacy")
            conn.execute(
                "CREATE TABLE memory_summaries ("
                "tenant_id TEXT NOT NULL, user_id TEXT NOT NULL, session_id TEXT NOT NULL, "
                "summary TEXT NOT NULL, covered_message_count INTEGER NOT NULL, "
                "version TEXT NOT NULL, source_message_ids TEXT NOT NULL DEFAULT '[]', "
                "source_digest TEXT NOT NULL DEFAULT '', updated_at TEXT NOT NULL, "
                "PRIMARY KEY(tenant_id, user_id, session_id))"
            )
            conn.execute(
                "INSERT INTO memory_summaries("
                "tenant_id, user_id, session_id, summary, covered_message_count, version, "
                "source_message_ids, source_digest, updated_at) "
                "SELECT legacy.tenant_id, COALESCE(("
                "SELECT MIN(events.user_id) FROM memory_events events "
                "WHERE events.tenant_id = legacy.tenant_id "
                "AND events.session_id = legacy.session_id "
                "GROUP BY events.tenant_id, events.session_id "
                "HAVING COUNT(DISTINCT events.user_id) = 1), ''), "
                "legacy.session_id, legacy.summary, legacy.covered_message_count, "
                "legacy.version, '[]', '', legacy.updated_at "
                "FROM memory_summaries_legacy legacy"
            )
            conn.execute("DROP TABLE memory_summaries_legacy")
            return
        if "source_message_ids" not in columns:
            conn.execute(
                "ALTER TABLE memory_summaries ADD COLUMN "
                "source_message_ids TEXT NOT NULL DEFAULT '[]'"
            )
        if "source_digest" not in columns:
            conn.execute(
                "ALTER TABLE memory_summaries ADD COLUMN "
                "source_digest TEXT NOT NULL DEFAULT ''"
            )

    def get_active_fact(
        self, tenant_id: str, user_id: str, key: str
    ) -> Optional[MemoryRecord]:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {self._FACT_COLUMNS} FROM memory_facts "
                "WHERE tenant_id = ? AND user_id = ? AND memory_key = ? "
                "AND status IN ('active', 'stale') ORDER BY version DESC LIMIT 1",
                (tenant_id, user_id, key),
            ).fetchone()
        return self._record(row) if row else None

    def write_fact(
        self,
        tenant_id: str,
        user_id: str,
        key: str,
        value: str,
        category: MemoryCategory,
        importance: float,
        confidence: float,
        explicit: bool,
        source_event_id: Optional[str],
        metadata: Dict[str, Any],
        current_time: datetime,
    ) -> MemoryRecord:
        with self._connect() as conn:
            conn.execute("BEGIN IMMEDIATE")
            row = conn.execute(
                f"SELECT {self._FACT_COLUMNS} FROM memory_facts "
                "WHERE tenant_id = ? AND user_id = ? AND memory_key = ? "
                "AND status IN ('active', 'stale') ORDER BY version DESC LIMIT 1",
                (tenant_id, user_id, key),
            ).fetchone()
            existing = self._record(row) if row else None
            if existing and existing.value == value:
                if source_event_id and existing.source_event_id == source_event_id:
                    return existing
                timestamp = _iso(current_time)
                conn.execute(
                    "UPDATE memory_facts SET last_confirmed_at = ?, updated_at = ?, "
                    "confidence = MIN(1.0, confidence + 0.05), "
                    "reinforcement = MIN(2.0, reinforcement + 0.1), "
                    "status = 'active', valid_to = NULL WHERE memory_id = ?",
                    (timestamp, timestamp, existing.memory_id),
                )
                updated = conn.execute(
                    f"SELECT {self._FACT_COLUMNS} FROM memory_facts WHERE memory_id = ?",
                    (existing.memory_id,),
                ).fetchone()
                return self._record(updated)
            if existing and not explicit:
                raise ValueError(
                    "automatically extracted fact conflicts with active memory"
                )
            record = MemoryRecord(
                memory_id=str(uuid4()),
                tenant_id=tenant_id,
                user_id=user_id,
                key=key,
                value=value,
                category=category,
                status="active",
                version=(existing.version + 1) if existing else 1,
                importance=importance,
                confidence=confidence,
                reinforcement=1.0,
                explicit=explicit,
                created_at=current_time,
                updated_at=current_time,
                last_confirmed_at=current_time,
                valid_from=current_time,
                valid_to=None,
                supersedes_id=existing.memory_id if existing else None,
                source_event_id=source_event_id,
                metadata=metadata,
            )
            if existing:
                conn.execute(
                    "UPDATE memory_facts SET status = 'superseded', valid_to = ?, updated_at = ? "
                    "WHERE memory_id = ? AND status IN ('active', 'stale')",
                    (
                        _iso(current_time),
                        _iso(current_time),
                        existing.memory_id,
                    ),
                )
            conn.execute(
                "INSERT INTO memory_facts(" + self._FACT_COLUMNS + ") "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._values(record),
            )
            return record

    def save_fact(self, memory: MemoryRecord, supersede_id: Optional[str] = None) -> None:
        with self._connect() as conn:
            if supersede_id:
                conn.execute(
                    "UPDATE memory_facts SET status = 'superseded', valid_to = ?, updated_at = ? "
                    "WHERE memory_id = ? AND status IN ('active', 'stale')",
                    (_iso(memory.valid_from), _iso(memory.updated_at), supersede_id),
                )
            conn.execute(
                "INSERT INTO memory_facts(" + self._FACT_COLUMNS + ") "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                self._values(memory),
            )

    def confirm_fact(self, memory_id: str, confirmed_at: datetime) -> MemoryRecord:
        timestamp = _iso(confirmed_at)
        with self._connect() as conn:
            conn.execute(
                "UPDATE memory_facts SET last_confirmed_at = ?, updated_at = ?, "
                "confidence = MIN(1.0, confidence + 0.05), "
                "reinforcement = MIN(2.0, reinforcement + 0.1), status = 'active', "
                "valid_to = NULL WHERE memory_id = ? AND status IN ('active', 'stale')",
                (timestamp, timestamp, memory_id),
            )
            row = conn.execute(
                f"SELECT {self._FACT_COLUMNS} FROM memory_facts WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        if row is None:
            raise KeyError(memory_id)
        return self._record(row)

    def mark_fact_status(
        self, memory_id: str, status: str, changed_at: datetime
    ) -> MemoryRecord:
        if status not in {
            "active",
            "stale",
            "resolved",
            "accepted",
            "rejected",
        }:
            raise ValueError("unsupported memory status")
        timestamp = _iso(changed_at)
        valid_to = timestamp if status in {"resolved", "accepted", "rejected"} else None
        with self._connect() as conn:
            conn.execute(
                "UPDATE memory_facts SET status = ?, updated_at = ?, valid_to = ? "
                "WHERE memory_id = ?",
                (status, timestamp, valid_to, memory_id),
            )
            row = conn.execute(
                f"SELECT {self._FACT_COLUMNS} FROM memory_facts WHERE memory_id = ?",
                (memory_id,),
            ).fetchone()
        if row is None:
            raise KeyError(memory_id)
        return self._record(row)

    def list_facts(
        self, tenant_id: str, user_id: str, include_inactive: bool = False
    ) -> List[MemoryRecord]:
        status_clause = "" if include_inactive else " AND status = 'active'"
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {self._FACT_COLUMNS} FROM memory_facts "
                f"WHERE tenant_id = ? AND user_id = ?{status_clause} ORDER BY created_at, version",
                (tenant_id, user_id),
            ).fetchall()
        return [self._record(row) for row in rows]

    def forget_facts(
        self, tenant_id: str, user_id: str, key: Optional[str] = None
    ) -> int:
        conditions = "tenant_id = ? AND user_id = ?"
        params: List[Any] = [tenant_id, user_id]
        if key is not None:
            conditions += " AND memory_key = ?"
            params.append(key)
        timestamp = datetime.now(timezone.utc).isoformat()
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT memory_id, memory_key, value, status, source_event_id "
                f"FROM memory_facts WHERE {conditions}",
                params,
            ).fetchall()
            for row in rows:
                conn.execute(
                    "INSERT OR IGNORE INTO memory_tombstones("
                    "tenant_id, user_id, memory_key, value_hash, created_at) VALUES (?, ?, ?, ?, ?)",
                    (
                        tenant_id,
                        user_id,
                        row["memory_key"],
                        stable_value_hash(row["memory_key"], row["value"]),
                        timestamp,
                    ),
                )
            memory_ids = [row["memory_id"] for row in rows]
            event_ids = [row["source_event_id"] for row in rows if row["source_event_id"]]
            sessions: List[str] = []
            requests: List[tuple[str, str]] = []
            if key is None:
                owner_events = conn.execute(
                    "SELECT event_id FROM memory_events WHERE tenant_id = ? AND user_id = ?",
                    (tenant_id, user_id),
                ).fetchall()
                event_ids = list({*event_ids, *[row["event_id"] for row in owner_events]})
            if event_ids:
                marks = ",".join(["?"] * len(event_ids))
                events = conn.execute(
                    f"SELECT session_id, request_id FROM memory_events WHERE event_id IN ({marks})",
                    event_ids,
                ).fetchall()
                sessions = list({row["session_id"] for row in events})
                requests = [(row["session_id"], row["request_id"]) for row in events]
                conn.execute(f"DELETE FROM memory_events WHERE event_id IN ({marks})", event_ids)
            if memory_ids:
                marks = ",".join(["?"] * len(memory_ids))
                conn.execute(
                    f"DELETE FROM memory_access_log WHERE memory_id IN ({marks})", memory_ids
                )
            conn.execute(f"DELETE FROM memory_facts WHERE {conditions}", params)
            for session_id in sessions:
                conn.execute(
                    "DELETE FROM memory_summaries "
                    "WHERE tenant_id = ? AND user_id = ? AND session_id = ?",
                    (tenant_id, user_id, session_id),
                )
            has_messages = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'session_messages'"
            ).fetchone()
            if has_messages:
                for session_id, request_id in requests:
                    conn.execute(
                        "DELETE FROM session_messages WHERE tenant_id = ? AND session_id = ? "
                        "AND request_id = ?",
                        (tenant_id, session_id, request_id),
                    )
        active_count = sum(1 for row in rows if row["status"] == "active")
        return active_count + (len(event_ids) if key is None else 0)

    def has_tombstone(
        self, tenant_id: str, user_id: str, key: str, value: str
    ) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM memory_tombstones WHERE tenant_id = ? AND user_id = ? "
                "AND memory_key = ? AND value_hash = ?",
                (tenant_id, user_id, key, stable_value_hash(key, value)),
            ).fetchone()
        return row is not None

    def clear_tombstone(self, tenant_id: str, user_id: str, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM memory_tombstones WHERE tenant_id = ? AND user_id = ? "
                "AND memory_key = ? AND value_hash = ?",
                (tenant_id, user_id, key, stable_value_hash(key, value)),
            )

    def append_event(self, event: Dict[str, Any]) -> str:
        event_id = str(uuid4())
        with self._connect() as conn:
            existing = conn.execute(
                "SELECT event_id FROM memory_events WHERE tenant_id = ? AND user_id = ? "
                "AND request_id = ? AND kind = ?",
                (
                    event["tenant_id"], event["user_id"], event["request_id"], event["kind"]
                ),
            ).fetchone()
            if existing:
                conn.execute(
                    "UPDATE memory_events SET session_id = ?, content = ?, metadata = ? "
                    "WHERE event_id = ?",
                    (
                        event["session_id"],
                        event["content"],
                        json.dumps(event.get("metadata", {}), ensure_ascii=False),
                        existing["event_id"],
                    ),
                )
                return str(existing["event_id"])
            conn.execute(
                "INSERT INTO memory_events(event_id, tenant_id, user_id, session_id, "
                "request_id, kind, content, metadata, created_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    event_id,
                    event["tenant_id"],
                    event["user_id"],
                    event["session_id"],
                    event["request_id"],
                    event["kind"],
                    event["content"],
                    json.dumps(event.get("metadata", {}), ensure_ascii=False),
                    datetime.now(timezone.utc).isoformat(),
                ),
            )
        return event_id

    def list_events(
        self, tenant_id: str, user_id: str, limit: int = 100
    ) -> List[MemoryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT event_id, content, metadata, created_at FROM memory_events "
                "WHERE tenant_id = ? AND user_id = ? ORDER BY created_at DESC LIMIT ?",
                (tenant_id, user_id, limit),
            ).fetchall()
        return [self._event_record(row, tenant_id, user_id) for row in rows]

    def list_event_details(
        self, tenant_id: str, user_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT event_id, session_id, request_id, kind, content, metadata, created_at "
                "FROM memory_events WHERE tenant_id = ? AND user_id = ? "
                "ORDER BY created_at DESC LIMIT ?",
                (tenant_id, user_id, limit),
            ).fetchall()
        return [
            {
                "event_id": row["event_id"], "session_id": row["session_id"],
                "request_id": row["request_id"], "kind": row["kind"],
                "content": row["content"],
                "metadata": json.loads(row["metadata"] or "{}"),
                "created_at": row["created_at"],
            }
            for row in rows
        ]

    def log_access(
        self, memory_id: str, tenant_id: str, user_id: str, score: float
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memory_access_log(memory_id, tenant_id, user_id, score, created_at) "
                "VALUES (?, ?, ?, ?, ?)",
                (memory_id, tenant_id, user_id, score, datetime.now(timezone.utc).isoformat()),
            )

    def save_summary(
        self,
        tenant_id: str,
        user_id: str,
        session_id: str,
        summary: str,
        covered_message_count: int,
        version: str,
        source_message_ids: Optional[List[str]] = None,
        source_digest: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memory_summaries(tenant_id, user_id, session_id, summary, "
                "covered_message_count, version, source_message_ids, source_digest, updated_at) "
                "VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(tenant_id, user_id, session_id) "
                "DO UPDATE SET summary = excluded.summary, "
                "covered_message_count = excluded.covered_message_count, version = excluded.version, "
                "source_message_ids = excluded.source_message_ids, "
                "source_digest = excluded.source_digest, "
                "updated_at = excluded.updated_at",
                (
                    tenant_id,
                    user_id,
                    session_id,
                    summary,
                    covered_message_count,
                    version,
                    json.dumps(source_message_ids or []),
                    source_digest,
                    datetime.now(timezone.utc).isoformat(),
                ),
            )

    def load_summary(
        self, tenant_id: str, user_id: str, session_id: str
    ) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary, covered_message_count, version, "
                "source_message_ids, source_digest FROM memory_summaries "
                "WHERE tenant_id = ? AND user_id = ? AND session_id = ?",
                (tenant_id, user_id, session_id),
            ).fetchone()
        if row is None:
            return None
        return {
            "summary": row["summary"],
            "covered_message_count": row["covered_message_count"],
            "version": row["version"],
            "source_message_ids": json.loads(row["source_message_ids"] or "[]"),
            "source_digest": row["source_digest"],
        }

    def list_summaries(
        self, tenant_id: str, user_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, summary, covered_message_count, version, "
                "source_message_ids, source_digest, updated_at "
                "FROM memory_summaries WHERE tenant_id = ? AND user_id = ? "
                "ORDER BY updated_at DESC LIMIT ?",
                (tenant_id, user_id, limit),
            ).fetchall()
        return [
            {
                "session_id": row["session_id"], "summary": row["summary"],
                "covered_message_count": row["covered_message_count"],
                "version": row["version"],
                "source_message_ids": json.loads(row["source_message_ids"] or "[]"),
                "source_digest": row["source_digest"],
                "updated_at": row["updated_at"],
            }
            for row in rows
        ]

    def prune_retention(
        self,
        raw_message_days: int,
        episodic_days: int,
        superseded_fact_days: int,
        access_log_days: int,
        procedure_candidate_days: int,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        with self._connect() as conn:
            old_facts = conn.execute(
                "SELECT memory_id FROM memory_facts WHERE status = 'superseded' "
                "AND datetime(updated_at) < datetime('now', ?)",
                (f"-{superseded_fact_days} days",),
            ).fetchall()
            deleted_ids = [row["memory_id"] for row in old_facts]
            cursor = conn.execute(
                "DELETE FROM memory_facts WHERE status = 'superseded' "
                "AND datetime(updated_at) < datetime('now', ?)",
                (f"-{superseded_fact_days} days",),
            )
            result["superseded_facts"] = cursor.rowcount
            cursor = conn.execute(
                "DELETE FROM memory_events WHERE datetime(created_at) < datetime('now', ?)",
                (f"-{episodic_days} days",),
            )
            result["events"] = cursor.rowcount
            cursor = conn.execute(
                "DELETE FROM memory_access_log WHERE datetime(created_at) < datetime('now', ?)",
                (f"-{access_log_days} days",),
            )
            result["access_logs"] = cursor.rowcount
            cursor = conn.execute(
                "DELETE FROM procedural_memories WHERE status = 'candidate' "
                "AND datetime(created_at) < datetime('now', ?)",
                (f"-{procedure_candidate_days} days",),
            )
            result["procedure_candidates"] = cursor.rowcount
            cursor = conn.execute(
                "DELETE FROM memory_summaries WHERE datetime(updated_at) < datetime('now', ?)",
                (f"-{raw_message_days} days",),
            )
            result["summaries"] = cursor.rowcount
            has_messages = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'session_messages'"
            ).fetchone()
            if has_messages:
                cursor = conn.execute(
                    "DELETE FROM session_messages WHERE datetime(created_at) < datetime('now', ?)",
                    (f"-{raw_message_days} days",),
                )
                result["raw_messages"] = cursor.rowcount
            else:
                result["raw_messages"] = 0
        result["deleted_memory_ids"] = deleted_ids
        return result

    def save_procedure(self, procedure: ProcedureMemory) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO procedural_memories(procedure_id, tenant_id, agent_version, status, "
                "title, content, evidence, created_at, approved_at) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    procedure.procedure_id, procedure.tenant_id, procedure.agent_version,
                    procedure.status, procedure.title, procedure.content,
                    json.dumps(procedure.evidence, ensure_ascii=False),
                    _iso(procedure.created_at),
                    _iso(procedure.approved_at) if procedure.approved_at else None,
                ),
            )

    def approve_procedure(
        self, procedure_id: str, approved_at: datetime
    ) -> ProcedureMemory:
        with self._connect() as conn:
            conn.execute(
                "UPDATE procedural_memories SET status = 'approved', approved_at = ? "
                "WHERE procedure_id = ? AND status = 'candidate'",
                (_iso(approved_at), procedure_id),
            )
            row = conn.execute(
                "SELECT * FROM procedural_memories WHERE procedure_id = ?", (procedure_id,)
            ).fetchone()
        if row is None:
            raise KeyError(procedure_id)
        return self._procedure(row)

    def list_procedures(
        self, tenant_id: Optional[str], status: str = "approved"
    ) -> List[ProcedureMemory]:
        with self._connect() as conn:
            if tenant_id is None:
                rows = conn.execute(
                    "SELECT * FROM procedural_memories WHERE tenant_id IS NULL AND status = ? "
                    "ORDER BY created_at",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM procedural_memories WHERE (tenant_id IS NULL OR tenant_id = ?) "
                    "AND status = ? ORDER BY created_at",
                    (tenant_id, status),
                ).fetchall()
        return [self._procedure(row) for row in rows]

    @staticmethod
    def _values(memory: MemoryRecord) -> tuple[Any, ...]:
        return (
            memory.memory_id,
            memory.tenant_id,
            memory.user_id,
            memory.key,
            memory.value,
            memory.category.value,
            memory.status,
            memory.version,
            memory.importance,
            memory.confidence,
            memory.reinforcement,
            int(memory.explicit),
            _iso(memory.created_at),
            _iso(memory.updated_at),
            _iso(memory.last_confirmed_at),
            _iso(memory.valid_from),
            _iso(memory.valid_to) if memory.valid_to else None,
            memory.supersedes_id,
            memory.source_event_id,
            json.dumps(memory.metadata, ensure_ascii=False),
        )

    @staticmethod
    def _record(row: sqlite3.Row) -> MemoryRecord:
        return MemoryRecord(
            memory_id=row["memory_id"],
            tenant_id=row["tenant_id"],
            user_id=row["user_id"],
            key=row["memory_key"],
            value=row["value"],
            category=MemoryCategory(row["category"]),
            status=row["status"],
            version=int(row["version"]),
            importance=float(row["importance"]),
            confidence=float(row["confidence"]),
            reinforcement=float(row["reinforcement"]),
            explicit=bool(row["explicit"]),
            created_at=_dt(row["created_at"]),
            updated_at=_dt(row["updated_at"]),
            last_confirmed_at=_dt(row["last_confirmed_at"]),
            valid_from=_dt(row["valid_from"]),
            valid_to=_dt(row["valid_to"]),
            supersedes_id=row["supersedes_id"],
            source_event_id=row["source_event_id"],
            metadata=json.loads(row["metadata"] or "{}"),
        )

    @staticmethod
    def _event_record(row: sqlite3.Row, tenant_id: str, user_id: str) -> MemoryRecord:
        created_at = _dt(row["created_at"])
        return MemoryRecord(
            memory_id=row["event_id"], tenant_id=tenant_id, user_id=user_id,
            key="episode", value=row["content"], category=MemoryCategory.EPISODIC,
            status="active", version=1, importance=0.4, confidence=0.8,
            reinforcement=1.0, explicit=False, created_at=created_at,
            updated_at=created_at, last_confirmed_at=created_at, valid_from=created_at,
            metadata=json.loads(row["metadata"] or "{}"),
        )

    @staticmethod
    def _procedure(row: sqlite3.Row) -> ProcedureMemory:
        return ProcedureMemory(
            procedure_id=row["procedure_id"], tenant_id=row["tenant_id"],
            agent_version=row["agent_version"], status=row["status"], title=row["title"],
            content=row["content"], evidence=json.loads(row["evidence"] or "{}"),
            created_at=_dt(row["created_at"]), approved_at=_dt(row["approved_at"]),
        )


class PostgresMemoryStore:
    """Postgres production backend; the vector index remains a derived layer."""

    shared = True
    _FACT_COLUMNS = SQLiteMemoryStore._FACT_COLUMNS

    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise ValueError("database_url is required")
        self.database_url = database_url
        self._init_db()

    def _connect(self):
        try:
            import psycopg
            from psycopg.rows import dict_row
        except ImportError as exc:  # pragma: no cover - production dependency
            raise RuntimeError(
                "Postgres memory backend requires the 'production' dependency extra"
            ) from exc
        return psycopg.connect(self.database_url, row_factory=dict_row)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(hashtext('sweeper_memory_schema'))")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memory_events ("
                "event_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, user_id TEXT NOT NULL, "
                "session_id TEXT NOT NULL, request_id TEXT NOT NULL, kind TEXT NOT NULL, "
                "content TEXT NOT NULL, metadata JSONB NOT NULL DEFAULT '{}'::jsonb, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "UNIQUE(tenant_id, user_id, request_id, kind))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memory_facts ("
                "memory_id TEXT PRIMARY KEY, tenant_id TEXT NOT NULL, user_id TEXT NOT NULL, "
                "memory_key TEXT NOT NULL, value TEXT NOT NULL, category TEXT NOT NULL, "
                "status TEXT NOT NULL, version INTEGER NOT NULL, importance DOUBLE PRECISION NOT NULL, "
                "confidence DOUBLE PRECISION NOT NULL, reinforcement DOUBLE PRECISION NOT NULL, "
                "explicit BOOLEAN NOT NULL, created_at TIMESTAMPTZ NOT NULL, "
                "updated_at TIMESTAMPTZ NOT NULL, last_confirmed_at TIMESTAMPTZ NOT NULL, "
                "valid_from TIMESTAMPTZ NOT NULL, valid_to TIMESTAMPTZ, supersedes_id TEXT, "
                "source_event_id TEXT, metadata JSONB NOT NULL DEFAULT '{}'::jsonb)"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_memory_fact_active "
                "ON memory_facts(tenant_id, user_id, memory_key) WHERE status = 'active'"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_memory_fact_owner "
                "ON memory_facts(tenant_id, user_id, status)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memory_tombstones ("
                "tenant_id TEXT NOT NULL, user_id TEXT NOT NULL, memory_key TEXT NOT NULL, "
                "value_hash TEXT NOT NULL, created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "PRIMARY KEY(tenant_id, user_id, memory_key, value_hash))"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memory_summaries ("
                "tenant_id TEXT NOT NULL, user_id TEXT NOT NULL, "
                "session_id TEXT NOT NULL, summary TEXT NOT NULL, "
                "covered_message_count INTEGER NOT NULL, version TEXT NOT NULL, "
                "source_message_ids JSONB NOT NULL DEFAULT '[]'::jsonb, "
                "source_digest TEXT NOT NULL DEFAULT '', "
                "updated_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, "
                "PRIMARY KEY(tenant_id, user_id, session_id))"
            )
            conn.execute(
                "ALTER TABLE memory_summaries ADD COLUMN IF NOT EXISTS "
                "user_id TEXT NOT NULL DEFAULT ''"
            )
            conn.execute(
                "UPDATE memory_summaries summaries SET user_id = owners.user_id FROM ("
                "SELECT tenant_id, session_id, MIN(user_id) AS user_id FROM memory_events "
                "GROUP BY tenant_id, session_id HAVING COUNT(DISTINCT user_id) = 1) owners "
                "WHERE summaries.tenant_id = owners.tenant_id "
                "AND summaries.session_id = owners.session_id AND summaries.user_id = ''"
            )
            conn.execute(
                "ALTER TABLE memory_summaries ADD COLUMN IF NOT EXISTS "
                "source_message_ids JSONB NOT NULL DEFAULT '[]'::jsonb"
            )
            conn.execute(
                "ALTER TABLE memory_summaries ADD COLUMN IF NOT EXISTS "
                "source_digest TEXT NOT NULL DEFAULT ''"
            )
            conn.execute(
                "ALTER TABLE memory_summaries DROP CONSTRAINT IF EXISTS memory_summaries_pkey"
            )
            conn.execute(
                "ALTER TABLE memory_summaries ADD CONSTRAINT memory_summaries_pkey "
                "PRIMARY KEY(tenant_id, user_id, session_id)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS memory_access_log ("
                "id BIGSERIAL PRIMARY KEY, memory_id TEXT NOT NULL, tenant_id TEXT NOT NULL, "
                "user_id TEXT NOT NULL, score DOUBLE PRECISION NOT NULL, "
                "adopted BOOLEAN NOT NULL DEFAULT FALSE, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS procedural_memories ("
                "procedure_id TEXT PRIMARY KEY, tenant_id TEXT, agent_version TEXT NOT NULL, "
                "status TEXT NOT NULL, title TEXT NOT NULL, content TEXT NOT NULL, "
                "evidence JSONB NOT NULL DEFAULT '{}'::jsonb, "
                "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP, approved_at TIMESTAMPTZ)"
            )

    def get_active_fact(
        self, tenant_id: str, user_id: str, key: str
    ) -> Optional[MemoryRecord]:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {self._FACT_COLUMNS} FROM memory_facts WHERE tenant_id = %s "
                "AND user_id = %s AND memory_key = %s "
                "AND status IN ('active', 'stale') ORDER BY version DESC LIMIT 1",
                (tenant_id, user_id, key),
            ).fetchone()
        return self._record(row) if row else None

    def write_fact(
        self,
        tenant_id: str,
        user_id: str,
        key: str,
        value: str,
        category: MemoryCategory,
        importance: float,
        confidence: float,
        explicit: bool,
        source_event_id: Optional[str],
        metadata: Dict[str, Any],
        current_time: datetime,
    ) -> MemoryRecord:
        lock_key = json.dumps(
            [tenant_id, user_id, key],
            ensure_ascii=False,
            separators=(",", ":"),
        )
        with self._connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(hashtext(%s))", (lock_key,))
            row = conn.execute(
                f"SELECT {self._FACT_COLUMNS} FROM memory_facts WHERE tenant_id = %s "
                "AND user_id = %s AND memory_key = %s "
                "AND status IN ('active', 'stale') ORDER BY version DESC LIMIT 1 FOR UPDATE",
                (tenant_id, user_id, key),
            ).fetchone()
            existing = self._record(row) if row else None
            if existing and existing.value == value:
                if source_event_id and existing.source_event_id == source_event_id:
                    return existing
                updated = conn.execute(
                    "UPDATE memory_facts SET last_confirmed_at = %s, updated_at = %s, "
                    "confidence = LEAST(1.0, confidence + 0.05), "
                    "reinforcement = LEAST(2.0, reinforcement + 0.1), "
                    "status = 'active', valid_to = NULL WHERE memory_id = %s "
                    "RETURNING " + self._FACT_COLUMNS,
                    (current_time, current_time, existing.memory_id),
                ).fetchone()
                return self._record(updated)
            if existing and not explicit:
                raise ValueError(
                    "automatically extracted fact conflicts with active memory"
                )
            record = MemoryRecord(
                memory_id=str(uuid4()),
                tenant_id=tenant_id,
                user_id=user_id,
                key=key,
                value=value,
                category=category,
                status="active",
                version=(existing.version + 1) if existing else 1,
                importance=importance,
                confidence=confidence,
                reinforcement=1.0,
                explicit=explicit,
                created_at=current_time,
                updated_at=current_time,
                last_confirmed_at=current_time,
                valid_from=current_time,
                valid_to=None,
                supersedes_id=existing.memory_id if existing else None,
                source_event_id=source_event_id,
                metadata=metadata,
            )
            if existing:
                conn.execute(
                    "UPDATE memory_facts SET status = 'superseded', valid_to = %s, updated_at = %s "
                    "WHERE memory_id = %s AND status IN ('active', 'stale')",
                    (current_time, current_time, existing.memory_id),
                )
            conn.execute(
                "INSERT INTO memory_facts(" + self._FACT_COLUMNS + ") VALUES ("
                + ", ".join(["%s"] * 19)
                + ", %s::jsonb)",
                self._values(record),
            )
            return record

    def save_fact(self, memory: MemoryRecord, supersede_id: Optional[str] = None) -> None:
        with self._connect() as conn:
            if supersede_id:
                conn.execute(
                    "UPDATE memory_facts SET status = 'superseded', valid_to = %s, updated_at = %s "
                    "WHERE memory_id = %s AND status IN ('active', 'stale')",
                    (memory.valid_from, memory.updated_at, supersede_id),
                )
            conn.execute(
                "INSERT INTO memory_facts(" + self._FACT_COLUMNS + ") VALUES ("
                + ", ".join(["%s"] * 19)
                + ", %s::jsonb)",
                self._values(memory),
            )

    def confirm_fact(self, memory_id: str, confirmed_at: datetime) -> MemoryRecord:
        with self._connect() as conn:
            row = conn.execute(
                "UPDATE memory_facts SET last_confirmed_at = %s, updated_at = %s, "
                "confidence = LEAST(1.0, confidence + 0.05), "
                "reinforcement = LEAST(2.0, reinforcement + 0.1), status = 'active', "
                "valid_to = NULL WHERE memory_id = %s AND status IN ('active', 'stale') "
                "RETURNING " + self._FACT_COLUMNS,
                (confirmed_at, confirmed_at, memory_id),
            ).fetchone()
        if row is None:
            raise KeyError(memory_id)
        return self._record(row)

    def mark_fact_status(
        self, memory_id: str, status: str, changed_at: datetime
    ) -> MemoryRecord:
        if status not in {
            "active",
            "stale",
            "resolved",
            "accepted",
            "rejected",
        }:
            raise ValueError("unsupported memory status")
        valid_to = changed_at if status in {"resolved", "accepted", "rejected"} else None
        with self._connect() as conn:
            row = conn.execute(
                "UPDATE memory_facts SET status = %s, updated_at = %s, valid_to = %s "
                "WHERE memory_id = %s RETURNING " + self._FACT_COLUMNS,
                (status, changed_at, valid_to, memory_id),
            ).fetchone()
        if row is None:
            raise KeyError(memory_id)
        return self._record(row)

    def list_facts(
        self, tenant_id: str, user_id: str, include_inactive: bool = False
    ) -> List[MemoryRecord]:
        status_clause = "" if include_inactive else " AND status = 'active'"
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT {self._FACT_COLUMNS} FROM memory_facts WHERE tenant_id = %s "
                f"AND user_id = %s{status_clause} ORDER BY created_at, version",
                (tenant_id, user_id),
            ).fetchall()
        return [self._record(row) for row in rows]

    def forget_facts(
        self, tenant_id: str, user_id: str, key: Optional[str] = None
    ) -> int:
        conditions = "tenant_id = %s AND user_id = %s"
        params: List[Any] = [tenant_id, user_id]
        if key is not None:
            conditions += " AND memory_key = %s"
            params.append(key)
        now = datetime.now(timezone.utc)
        with self._connect() as conn:
            rows = conn.execute(
                f"SELECT memory_id, memory_key, value, status, source_event_id "
                f"FROM memory_facts WHERE {conditions}",
                params,
            ).fetchall()
            for row in rows:
                conn.execute(
                    "INSERT INTO memory_tombstones(tenant_id, user_id, memory_key, value_hash, created_at) "
                    "VALUES (%s, %s, %s, %s, %s) ON CONFLICT DO NOTHING",
                    (
                        tenant_id,
                        user_id,
                        row["memory_key"],
                        stable_value_hash(row["memory_key"], row["value"]),
                        now,
                    ),
                )
            memory_ids = [row["memory_id"] for row in rows]
            event_ids = [row["source_event_id"] for row in rows if row["source_event_id"]]
            sessions: List[str] = []
            requests: List[tuple[str, str]] = []
            if key is None:
                owner_events = conn.execute(
                    "SELECT event_id FROM memory_events WHERE tenant_id = %s AND user_id = %s",
                    (tenant_id, user_id),
                ).fetchall()
                event_ids = list({*event_ids, *[row["event_id"] for row in owner_events]})
            if event_ids:
                events = conn.execute(
                    "SELECT session_id, request_id FROM memory_events WHERE event_id = ANY(%s)",
                    (event_ids,),
                ).fetchall()
                sessions = list({row["session_id"] for row in events})
                requests = [(row["session_id"], row["request_id"]) for row in events]
                conn.execute("DELETE FROM memory_events WHERE event_id = ANY(%s)", (event_ids,))
            if memory_ids:
                conn.execute(
                    "DELETE FROM memory_access_log WHERE memory_id = ANY(%s)", (memory_ids,)
                )
            conn.execute(f"DELETE FROM memory_facts WHERE {conditions}", params)
            for session_id in sessions:
                conn.execute(
                    "DELETE FROM memory_summaries "
                    "WHERE tenant_id = %s AND user_id = %s AND session_id = %s",
                    (tenant_id, user_id, session_id),
                )
            messages_table = conn.execute("SELECT to_regclass('session_messages') AS name").fetchone()
            if messages_table and messages_table["name"]:
                for session_id, request_id in requests:
                    conn.execute(
                        "DELETE FROM session_messages WHERE tenant_id = %s AND session_id = %s "
                        "AND request_id = %s",
                        (tenant_id, session_id, request_id),
                    )
        active_count = sum(1 for row in rows if row["status"] == "active")
        return active_count + (len(event_ids) if key is None else 0)

    def has_tombstone(
        self, tenant_id: str, user_id: str, key: str, value: str
    ) -> bool:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT 1 FROM memory_tombstones WHERE tenant_id = %s AND user_id = %s "
                "AND memory_key = %s AND value_hash = %s",
                (tenant_id, user_id, key, stable_value_hash(key, value)),
            ).fetchone()
        return row is not None

    def clear_tombstone(self, tenant_id: str, user_id: str, key: str, value: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "DELETE FROM memory_tombstones WHERE tenant_id = %s AND user_id = %s "
                "AND memory_key = %s AND value_hash = %s",
                (tenant_id, user_id, key, stable_value_hash(key, value)),
            )

    def append_event(self, event: Dict[str, Any]) -> str:
        with self._connect() as conn:
            row = conn.execute(
                "INSERT INTO memory_events(event_id, tenant_id, user_id, session_id, request_id, "
                "kind, content, metadata) VALUES (%s, %s, %s, %s, %s, %s, %s, %s::jsonb) "
                "ON CONFLICT(tenant_id, user_id, request_id, kind) DO UPDATE SET "
                "session_id = EXCLUDED.session_id, content = EXCLUDED.content, "
                "metadata = EXCLUDED.metadata RETURNING event_id",
                (
                    str(uuid4()),
                    event["tenant_id"],
                    event["user_id"],
                    event["session_id"],
                    event["request_id"],
                    event["kind"],
                    event["content"],
                    json.dumps(event.get("metadata", {}), ensure_ascii=False),
                ),
            ).fetchone()
        return str(row["event_id"])

    def list_events(
        self, tenant_id: str, user_id: str, limit: int = 100
    ) -> List[MemoryRecord]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT event_id, content, metadata, created_at FROM memory_events "
                "WHERE tenant_id = %s AND user_id = %s ORDER BY created_at DESC LIMIT %s",
                (tenant_id, user_id, limit),
            ).fetchall()
        return [self._event_record(row, tenant_id, user_id) for row in rows]

    def list_event_details(
        self, tenant_id: str, user_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT event_id, session_id, request_id, kind, content, metadata, created_at "
                "FROM memory_events WHERE tenant_id = %s AND user_id = %s "
                "ORDER BY created_at DESC LIMIT %s",
                (tenant_id, user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def log_access(
        self, memory_id: str, tenant_id: str, user_id: str, score: float
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memory_access_log(memory_id, tenant_id, user_id, score) "
                "VALUES (%s, %s, %s, %s)",
                (memory_id, tenant_id, user_id, score),
            )

    def save_summary(
        self,
        tenant_id: str,
        user_id: str,
        session_id: str,
        summary: str,
        covered_message_count: int,
        version: str,
        source_message_ids: Optional[List[str]] = None,
        source_digest: str = "",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO memory_summaries("
                "tenant_id, user_id, session_id, summary, covered_message_count, "
                "version, source_message_ids, source_digest) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s) "
                "ON CONFLICT(tenant_id, user_id, session_id) "
                "DO UPDATE SET summary = EXCLUDED.summary, "
                "covered_message_count = EXCLUDED.covered_message_count, "
                "version = EXCLUDED.version, "
                "source_message_ids = EXCLUDED.source_message_ids, "
                "source_digest = EXCLUDED.source_digest, "
                "updated_at = CURRENT_TIMESTAMP",
                (
                    tenant_id,
                    user_id,
                    session_id,
                    summary,
                    covered_message_count,
                    version,
                    json.dumps(source_message_ids or []),
                    source_digest,
                ),
            )

    def load_summary(
        self, tenant_id: str, user_id: str, session_id: str
    ) -> Optional[Dict[str, Any]]:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT summary, covered_message_count, version, "
                "source_message_ids, source_digest FROM memory_summaries "
                "WHERE tenant_id = %s AND user_id = %s AND session_id = %s",
                (tenant_id, user_id, session_id),
            ).fetchone()
        return dict(row) if row else None

    def list_summaries(
        self, tenant_id: str, user_id: str, limit: int = 100
    ) -> List[Dict[str, Any]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, summary, covered_message_count, version, "
                "source_message_ids, source_digest, updated_at "
                "FROM memory_summaries WHERE tenant_id = %s AND user_id = %s "
                "ORDER BY updated_at DESC LIMIT %s",
                (tenant_id, user_id, limit),
            ).fetchall()
        return [dict(row) for row in rows]

    def prune_retention(
        self,
        raw_message_days: int,
        episodic_days: int,
        superseded_fact_days: int,
        access_log_days: int,
        procedure_candidate_days: int,
    ) -> Dict[str, Any]:
        result: Dict[str, Any] = {}
        with self._connect() as conn:
            old_facts = conn.execute(
                "SELECT memory_id FROM memory_facts WHERE status = 'superseded' "
                "AND updated_at < CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')",
                (superseded_fact_days,),
            ).fetchall()
            deleted_ids = [row["memory_id"] for row in old_facts]
            result["superseded_facts"] = conn.execute(
                "DELETE FROM memory_facts WHERE status = 'superseded' "
                "AND updated_at < CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')",
                (superseded_fact_days,),
            ).rowcount
            result["events"] = conn.execute(
                "DELETE FROM memory_events WHERE created_at < "
                "CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')",
                (episodic_days,),
            ).rowcount
            result["access_logs"] = conn.execute(
                "DELETE FROM memory_access_log WHERE created_at < "
                "CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')",
                (access_log_days,),
            ).rowcount
            result["procedure_candidates"] = conn.execute(
                "DELETE FROM procedural_memories WHERE status = 'candidate' AND created_at < "
                "CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')",
                (procedure_candidate_days,),
            ).rowcount
            result["summaries"] = conn.execute(
                "DELETE FROM memory_summaries WHERE updated_at < "
                "CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')",
                (raw_message_days,),
            ).rowcount
            messages_table = conn.execute("SELECT to_regclass('session_messages') AS name").fetchone()
            result["raw_messages"] = 0
            if messages_table and messages_table["name"]:
                result["raw_messages"] = conn.execute(
                    "DELETE FROM session_messages WHERE created_at < "
                    "CURRENT_TIMESTAMP - (%s * INTERVAL '1 day')",
                    (raw_message_days,),
                ).rowcount
        result["deleted_memory_ids"] = deleted_ids
        return result

    def save_procedure(self, procedure: ProcedureMemory) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO procedural_memories(procedure_id, tenant_id, agent_version, status, "
                "title, content, evidence, created_at, approved_at) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s)",
                (
                    procedure.procedure_id, procedure.tenant_id, procedure.agent_version,
                    procedure.status, procedure.title, procedure.content,
                    json.dumps(procedure.evidence, ensure_ascii=False),
                    procedure.created_at, procedure.approved_at,
                ),
            )

    def approve_procedure(
        self, procedure_id: str, approved_at: datetime
    ) -> ProcedureMemory:
        with self._connect() as conn:
            row = conn.execute(
                "UPDATE procedural_memories SET status = 'approved', approved_at = %s "
                "WHERE procedure_id = %s AND status = 'candidate' RETURNING *",
                (approved_at, procedure_id),
            ).fetchone()
            if row is None:
                row = conn.execute(
                    "SELECT * FROM procedural_memories WHERE procedure_id = %s", (procedure_id,)
                ).fetchone()
        if row is None:
            raise KeyError(procedure_id)
        return self._procedure(row)

    def list_procedures(
        self, tenant_id: Optional[str], status: str = "approved"
    ) -> List[ProcedureMemory]:
        with self._connect() as conn:
            if tenant_id is None:
                rows = conn.execute(
                    "SELECT * FROM procedural_memories WHERE tenant_id IS NULL AND status = %s "
                    "ORDER BY created_at",
                    (status,),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM procedural_memories WHERE (tenant_id IS NULL OR tenant_id = %s) "
                    "AND status = %s ORDER BY created_at",
                    (tenant_id, status),
                ).fetchall()
        return [self._procedure(row) for row in rows]

    @staticmethod
    def _values(memory: MemoryRecord) -> tuple[Any, ...]:
        return (
            memory.memory_id,
            memory.tenant_id,
            memory.user_id,
            memory.key,
            memory.value,
            memory.category.value,
            memory.status,
            memory.version,
            memory.importance,
            memory.confidence,
            memory.reinforcement,
            memory.explicit,
            memory.created_at,
            memory.updated_at,
            memory.last_confirmed_at,
            memory.valid_from,
            memory.valid_to,
            memory.supersedes_id,
            memory.source_event_id,
            json.dumps(memory.metadata, ensure_ascii=False),
        )

    @staticmethod
    def _record(row: Dict[str, Any]) -> MemoryRecord:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        return MemoryRecord(
            memory_id=row["memory_id"], tenant_id=row["tenant_id"], user_id=row["user_id"],
            key=row["memory_key"], value=row["value"], category=MemoryCategory(row["category"]),
            status=row["status"], version=int(row["version"]), importance=float(row["importance"]),
            confidence=float(row["confidence"]), reinforcement=float(row["reinforcement"]),
            explicit=bool(row["explicit"]), created_at=row["created_at"],
            updated_at=row["updated_at"], last_confirmed_at=row["last_confirmed_at"],
            valid_from=row["valid_from"], valid_to=row["valid_to"],
            supersedes_id=row["supersedes_id"], source_event_id=row["source_event_id"],
            metadata=metadata or {},
        )

    @staticmethod
    def _event_record(row: Dict[str, Any], tenant_id: str, user_id: str) -> MemoryRecord:
        metadata = row["metadata"]
        if isinstance(metadata, str):
            metadata = json.loads(metadata)
        created_at = row["created_at"]
        return MemoryRecord(
            memory_id=row["event_id"], tenant_id=tenant_id, user_id=user_id,
            key="episode", value=row["content"], category=MemoryCategory.EPISODIC,
            status="active", version=1, importance=0.4, confidence=0.8,
            reinforcement=1.0, explicit=False, created_at=created_at,
            updated_at=created_at, last_confirmed_at=created_at, valid_from=created_at,
            metadata=metadata or {},
        )

    @staticmethod
    def _procedure(row: Dict[str, Any]) -> ProcedureMemory:
        evidence = row["evidence"]
        if isinstance(evidence, str):
            evidence = json.loads(evidence)
        return ProcedureMemory(
            procedure_id=row["procedure_id"], tenant_id=row["tenant_id"],
            agent_version=row["agent_version"], status=row["status"], title=row["title"],
            content=row["content"], evidence=evidence or {}, created_at=row["created_at"],
            approved_at=row["approved_at"],
        )
