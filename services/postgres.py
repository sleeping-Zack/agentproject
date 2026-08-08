from __future__ import annotations

import json
from typing import Any, Dict, List, Optional
from uuid import uuid4

from safety.approval import ApprovalRecord, utc_now_iso
from services.artifact_store import ArtifactRecord
from services.persistence import _split_tenant_session


class _PostgresBackend:
    def __init__(self, database_url: str) -> None:
        if not database_url:
            raise ValueError("database_url is required")
        self.database_url = database_url

    def _connect(self):
        try:
            import psycopg
        except ImportError as exc:  # pragma: no cover - depends on production extra
            raise RuntimeError(
                "Postgres backend requires the 'production' dependency extra"
            ) from exc
        return psycopg.connect(self.database_url)

    @staticmethod
    def _json(value: Any) -> Any:
        return json.loads(value) if isinstance(value, str) else value


class PostgresStore(_PostgresBackend):
    shared = True

    def __init__(self, database_url: str) -> None:
        super().__init__(database_url)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(hashtext('sweeper_agent_schema'))")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS session_messages ("
                "id BIGSERIAL PRIMARY KEY,"
                "session_id TEXT NOT NULL,"
                "tenant_id TEXT NOT NULL DEFAULT 'default',"
                "request_id TEXT,"
                "role TEXT NOT NULL,"
                "content TEXT NOT NULL,"
                "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.execute(
                "ALTER TABLE session_messages ADD COLUMN IF NOT EXISTS "
                "user_id TEXT NOT NULL DEFAULT ''"
            )
            memory_events = conn.execute(
                "SELECT to_regclass('memory_events') AS name"
            ).fetchone()
            if memory_events and memory_events[0]:
                conn.execute(
                    "UPDATE session_messages AS messages SET user_id = events.user_id FROM ("
                    "SELECT tenant_id, session_id, MIN(user_id) AS user_id FROM memory_events "
                    "GROUP BY tenant_id, session_id HAVING COUNT(DISTINCT user_id) = 1) AS events "
                    "WHERE messages.tenant_id = events.tenant_id "
                    "AND messages.session_id = events.session_id AND messages.user_id = ''"
                )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS traces ("
                "request_id TEXT PRIMARY KEY,"
                "session_id TEXT NOT NULL,"
                "tenant_id TEXT NOT NULL DEFAULT 'default',"
                "payload JSONB NOT NULL,"
                "created_at TIMESTAMPTZ NOT NULL DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_messages_tenant_session "
                "ON session_messages(tenant_id, session_id)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_messages_owner "
                "ON session_messages(tenant_id, user_id, created_at)"
            )
            conn.execute("DROP INDEX IF EXISTS idx_session_message_idempotency")
            conn.execute(
                "CREATE UNIQUE INDEX idx_session_message_idempotency "
                "ON session_messages(tenant_id, user_id, session_id, request_id, role)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_traces_tenant ON traces(tenant_id)"
            )

    def save_session_message(
        self,
        session_id: str,
        role: str,
        content: str,
        tenant_id: str = "default",
        request_id: Optional[str] = None,
        user_id: str = "",
    ) -> bool:
        with self._connect() as conn:
            cursor = conn.execute(
                "INSERT INTO session_messages("
                "session_id, tenant_id, user_id, request_id, role, content) "
                "VALUES (%s, %s, %s, %s, %s, %s) "
                "ON CONFLICT(tenant_id, user_id, session_id, request_id, role) DO NOTHING",
                (session_id, tenant_id, user_id, request_id, role, content),
            )
        return cursor.rowcount == 1

    def get_session_messages(
        self,
        session_id: str,
        tenant_id: str = "default",
        user_id: Optional[str] = None,
    ) -> List[Dict[str, str]]:
        owner_clause = " AND user_id = %s" if user_id is not None else ""
        params = (session_id, tenant_id, user_id) if user_id is not None else (session_id, tenant_id)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM session_messages "
                f"WHERE session_id = %s AND tenant_id = %s{owner_clause} ORDER BY id",
                params,
            ).fetchall()
        return [{"role": role, "content": content} for role, content in rows]

    def list_sessions(
        self, tenant_id: str, user_id: str, limit: int = 100
    ) -> List[Dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT messages.session_id, COUNT(*) AS message_count, "
                "MIN(messages.created_at) AS created_at, MAX(messages.created_at) AS updated_at, "
                "COALESCE((SELECT first_message.content FROM session_messages first_message "
                "WHERE first_message.tenant_id = messages.tenant_id "
                "AND first_message.user_id = messages.user_id "
                "AND first_message.session_id = messages.session_id "
                "AND first_message.role = 'user' ORDER BY first_message.id LIMIT 1), '') AS title "
                "FROM session_messages messages WHERE messages.tenant_id = %s "
                "AND messages.user_id = %s GROUP BY messages.session_id, messages.tenant_id, "
                "messages.user_id ORDER BY updated_at DESC LIMIT %s",
                (tenant_id, user_id, limit),
            ).fetchall()
        return [
            {
                "session_id": row[0], "message_count": row[1], "created_at": row[2],
                "updated_at": row[3], "title": row[4],
            }
            for row in rows
        ]

    def load_messages(self, session_id: str) -> List[Dict[str, str]]:
        sid, tid, uid = _split_tenant_session(session_id)
        return self.get_session_messages(sid, tenant_id=tid, user_id=uid or None)

    def list_message_refs(self, session_id: str) -> List[str]:
        sid, tid, uid = _split_tenant_session(session_id)
        owner_clause = " AND user_id = %s" if uid else ""
        params = (sid, tid, uid) if uid else (sid, tid)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, role FROM session_messages "
                f"WHERE session_id = %s AND tenant_id = %s{owner_clause} ORDER BY id",
                params,
            ).fetchall()
        return [f"{row[0]}:{row[1]}" for row in rows]

    def append_message(
        self,
        session_id: str,
        role: str,
        content: str,
        request_id: Optional[str] = None,
        user_id: Optional[str] = None,
    ) -> bool:
        sid, tid, encoded_uid = _split_tenant_session(session_id)
        return self.save_session_message(
            sid,
            role,
            content,
            tenant_id=tid,
            request_id=request_id,
            user_id=user_id or encoded_uid,
        )

    def save_trace(
        self,
        request_id: str,
        session_id: str,
        payload: Dict,
        tenant_id: str = "default",
    ) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO traces(request_id, session_id, tenant_id, payload) "
                "VALUES (%s, %s, %s, %s::jsonb) "
                "ON CONFLICT(request_id) DO UPDATE SET "
                "session_id = EXCLUDED.session_id, tenant_id = EXCLUDED.tenant_id, "
                "payload = EXCLUDED.payload, created_at = CURRENT_TIMESTAMP",
                (
                    request_id,
                    session_id,
                    tenant_id,
                    json.dumps(payload, ensure_ascii=False),
                ),
            )

    def get_trace(self, request_id: str, tenant_id: Optional[str] = None) -> Dict:
        tenant_clause = " AND tenant_id = %s" if tenant_id is not None else ""
        params = (request_id, tenant_id) if tenant_id is not None else (request_id,)
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT payload FROM traces WHERE request_id = %s{tenant_clause}",
                params,
            ).fetchone()
        if row is None:
            raise KeyError(request_id)
        return self._json(row[0])


class PostgresApprovalStore(_PostgresBackend):
    _COLUMNS = (
        "approval_id, request_id, tenant_id, principal_id, user_role, tool_name, args, "
        "reason, status, created_at, decided_at, decided_by"
    )

    def __init__(self, database_url: str) -> None:
        super().__init__(database_url)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(hashtext('sweeper_agent_schema'))")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS approvals ("
                "approval_id TEXT PRIMARY KEY,"
                "request_id TEXT NOT NULL,"
                "tenant_id TEXT NOT NULL,"
                "principal_id TEXT NOT NULL DEFAULT '',"
                "user_role TEXT NOT NULL,"
                "tool_name TEXT NOT NULL,"
                "args JSONB NOT NULL,"
                "reason TEXT NOT NULL,"
                "status TEXT NOT NULL,"
                "created_at TEXT NOT NULL,"
                "decided_at TEXT,"
                "decided_by TEXT)"
            )
            conn.execute(
                "ALTER TABLE approvals "
                "ADD COLUMN IF NOT EXISTS principal_id TEXT NOT NULL DEFAULT ''"
            )
            conn.execute(
                "UPDATE approvals "
                "SET status = 'denied', decided_at = %s, "
                "decided_by = 'system:legacy-scope-migration' "
                "WHERE status = 'pending' AND principal_id = ''",
                (utc_now_iso(),),
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_approvals_request_tool "
                "ON approvals(tenant_id, request_id, tool_name)"
            )

    def create_pending(
        self,
        request_id: str,
        tenant_id: str,
        user_role: str,
        tool_name: str,
        args: Dict[str, Any],
        reason: str,
        principal_id: str = "",
    ) -> ApprovalRecord:
        record = ApprovalRecord(
            approval_id=str(uuid4()),
            request_id=request_id,
            tenant_id=tenant_id,
            principal_id=principal_id,
            user_role=user_role,
            tool_name=tool_name,
            args=args,
            reason=reason,
            created_at=utc_now_iso(),
        )
        with self._connect() as conn:
            row = conn.execute(
                f"INSERT INTO approvals({self._COLUMNS}) "
                "VALUES (%s, %s, %s, %s, %s, %s, %s::jsonb, %s, %s, %s, %s, %s) "
                "ON CONFLICT(tenant_id, request_id, tool_name) DO NOTHING "
                f"RETURNING {self._COLUMNS}",
                (
                    record.approval_id,
                    record.request_id,
                    record.tenant_id,
                    record.principal_id,
                    record.user_role,
                    record.tool_name,
                    json.dumps(record.args, ensure_ascii=False),
                    record.reason,
                    record.status,
                    record.created_at,
                    record.decided_at,
                    record.decided_by,
                ),
            ).fetchone()
            if row is None:
                row = conn.execute(
                    f"SELECT {self._COLUMNS} FROM approvals "
                    "WHERE tenant_id = %s AND request_id = %s AND tool_name = %s",
                    (tenant_id, request_id, tool_name),
                ).fetchone()
        return self._row_to_record(row)

    def get(self, approval_id: str) -> ApprovalRecord:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {self._COLUMNS} FROM approvals WHERE approval_id = %s",
                (approval_id,),
            ).fetchone()
        if row is None:
            raise KeyError(approval_id)
        return self._row_to_record(row)

    def list_approvals(
        self,
        tenant_id: str,
        status: Optional[str] = None,
        limit: int = 100,
    ) -> List[ApprovalRecord]:
        query = f"SELECT {self._COLUMNS} FROM approvals WHERE tenant_id = %s"
        params: list[Any] = [tenant_id]
        if status:
            query += " AND status = %s"
            params.append(status)
        query += " ORDER BY created_at DESC LIMIT %s"
        params.append(limit)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def approve(self, approval_id: str, decided_by: str) -> ApprovalRecord:
        return self._decide(approval_id, "approved", decided_by)

    def deny(self, approval_id: str, decided_by: str) -> ApprovalRecord:
        return self._decide(approval_id, "denied", decided_by)

    def _decide(self, approval_id: str, status: str, decided_by: str) -> ApprovalRecord:
        with self._connect() as conn:
            row = conn.execute(
                "UPDATE approvals SET status = %s, decided_at = %s, decided_by = %s "
                "WHERE approval_id = %s AND status = 'pending' "
                f"RETURNING {self._COLUMNS}",
                (status, utc_now_iso(), decided_by, approval_id),
            ).fetchone()
            if row is None:
                row = conn.execute(
                    f"SELECT {self._COLUMNS} FROM approvals WHERE approval_id = %s",
                    (approval_id,),
                ).fetchone()
        if row is None:
            raise KeyError(approval_id)
        return self._row_to_record(row)

    def _row_to_record(self, row) -> ApprovalRecord:
        return ApprovalRecord(
            approval_id=row[0],
            request_id=row[1],
            tenant_id=row[2],
            principal_id=row[3],
            user_role=row[4],
            tool_name=row[5],
            args=self._json(row[6]),
            reason=row[7],
            status=row[8],
            created_at=row[9],
            decided_at=row[10],
            decided_by=row[11],
        )


class PostgresArtifactStore(_PostgresBackend):
    _COLUMNS = (
        "artifact_id, request_id, tenant_id, artifact_type, name, "
        "payload, metadata, created_at"
    )

    def __init__(self, database_url: str) -> None:
        super().__init__(database_url)
        self._init_db()

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute("SELECT pg_advisory_xact_lock(hashtext('sweeper_agent_schema'))")
            conn.execute(
                "CREATE TABLE IF NOT EXISTS artifacts ("
                "artifact_id TEXT PRIMARY KEY,"
                "request_id TEXT NOT NULL,"
                "tenant_id TEXT NOT NULL,"
                "artifact_type TEXT NOT NULL,"
                "name TEXT NOT NULL,"
                "payload JSONB NOT NULL,"
                "metadata JSONB NOT NULL,"
                "created_at TEXT NOT NULL)"
            )
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_artifacts_request "
                "ON artifacts(request_id, tenant_id)"
            )
            conn.execute(
                "CREATE UNIQUE INDEX IF NOT EXISTS idx_artifacts_idempotency "
                "ON artifacts(tenant_id, request_id, artifact_type, name)"
            )

    def save_artifact(
        self,
        request_id: str,
        tenant_id: str,
        artifact_type: str,
        name: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRecord:
        record = ArtifactRecord(
            artifact_id=str(uuid4()),
            request_id=request_id,
            tenant_id=tenant_id,
            artifact_type=artifact_type,
            name=name,
            payload=payload,
            metadata=metadata or {},
            created_at=utc_now_iso(),
        )
        with self._connect() as conn:
            row = conn.execute(
                f"INSERT INTO artifacts({self._COLUMNS}) "
                "VALUES (%s, %s, %s, %s, %s, %s::jsonb, %s::jsonb, %s) "
                "ON CONFLICT(tenant_id, request_id, artifact_type, name) DO NOTHING "
                f"RETURNING {self._COLUMNS}",
                (
                    record.artifact_id,
                    record.request_id,
                    record.tenant_id,
                    record.artifact_type,
                    record.name,
                    json.dumps(record.payload, ensure_ascii=False),
                    json.dumps(record.metadata, ensure_ascii=False),
                    record.created_at,
                ),
            ).fetchone()
            if row is None:
                row = conn.execute(
                    f"SELECT {self._COLUMNS} FROM artifacts "
                    "WHERE tenant_id = %s AND request_id = %s "
                    "AND artifact_type = %s AND name = %s",
                    (tenant_id, request_id, artifact_type, name),
                ).fetchone()
        return self._row_to_record(row)

    def get_artifact(self, artifact_id: str) -> ArtifactRecord:
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT {self._COLUMNS} FROM artifacts WHERE artifact_id = %s",
                (artifact_id,),
            ).fetchone()
        if row is None:
            raise KeyError(artifact_id)
        return self._row_to_record(row)

    def list_artifacts(
        self,
        request_id: str,
        tenant_id: Optional[str] = None,
    ) -> List[ArtifactRecord]:
        if tenant_id is None:
            query = (
                f"SELECT {self._COLUMNS} FROM artifacts "
                "WHERE request_id = %s ORDER BY created_at"
            )
            params = (request_id,)
        else:
            query = (
                f"SELECT {self._COLUMNS} FROM artifacts "
                "WHERE request_id = %s AND tenant_id = %s ORDER BY created_at"
            )
            params = (request_id, tenant_id)
        with self._connect() as conn:
            rows = conn.execute(query, params).fetchall()
        return [self._row_to_record(row) for row in rows]

    def _row_to_record(self, row) -> ArtifactRecord:
        return ArtifactRecord(
            artifact_id=row[0],
            request_id=row[1],
            tenant_id=row[2],
            artifact_type=row[3],
            name=row[4],
            payload=self._json(row[5]),
            metadata=self._json(row[6]),
            created_at=row[7],
        )
