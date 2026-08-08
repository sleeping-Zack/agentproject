import json
import os
import sqlite3
from typing import Dict, List, Optional


class SQLiteStore:
    def __init__(self, db_path: str) -> None:
        self.db_path = db_path
        directory = os.path.dirname(db_path)
        if directory:
            os.makedirs(directory, exist_ok=True)
        self._init_db()

    def _connect(self):
        return sqlite3.connect(self.db_path)

    def _init_db(self) -> None:
        with self._connect() as conn:
            conn.execute(
                "CREATE TABLE IF NOT EXISTS session_messages ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "session_id TEXT NOT NULL,"
                "tenant_id TEXT NOT NULL DEFAULT 'default',"
                "request_id TEXT,"
                "role TEXT NOT NULL,"
                "content TEXT NOT NULL,"
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS traces ("
                "request_id TEXT PRIMARY KEY,"
                "session_id TEXT NOT NULL,"
                "tenant_id TEXT NOT NULL DEFAULT 'default',"
                "payload TEXT NOT NULL,"
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            self._ensure_column(conn, "session_messages", "tenant_id",
                                "TEXT NOT NULL DEFAULT 'default'")
            self._ensure_column(conn, "session_messages", "request_id", "TEXT")
            self._ensure_column(conn, "session_messages", "user_id", "TEXT NOT NULL DEFAULT ''")
            self._ensure_column(conn, "traces", "tenant_id",
                                "TEXT NOT NULL DEFAULT 'default'")
            has_memory_events = conn.execute(
                "SELECT 1 FROM sqlite_master WHERE type = 'table' AND name = 'memory_events'"
            ).fetchone()
            if has_memory_events:
                conn.execute(
                    "UPDATE session_messages SET user_id = COALESCE(("
                    "SELECT user_id FROM memory_events WHERE memory_events.tenant_id = "
                    "session_messages.tenant_id AND memory_events.session_id = "
                    "session_messages.session_id GROUP BY tenant_id, session_id "
                    "HAVING COUNT(DISTINCT user_id) = 1), user_id) WHERE user_id = ''"
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
                "CREATE INDEX IF NOT EXISTS idx_traces_tenant "
                "ON traces(tenant_id)"
            )

    @staticmethod
    def _ensure_column(conn, table: str, column: str, ddl: str) -> None:
        cols = {row[1] for row in conn.execute(f"PRAGMA table_info({table})")}
        if column not in cols:
            conn.execute(f"ALTER TABLE {table} ADD COLUMN {column} {ddl}")

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
                "VALUES (?, ?, ?, ?, ?, ?) "
                "ON CONFLICT(tenant_id, user_id, session_id, request_id, role) DO NOTHING",
                (session_id, tenant_id, user_id, request_id, role, content),
            )
        return cursor.rowcount == 1

    def get_session_messages(
        self, session_id: str, tenant_id: str = "default", user_id: Optional[str] = None
    ) -> List[Dict[str, str]]:
        owner_clause = " AND user_id = ?" if user_id is not None else ""
        params = (session_id, tenant_id, user_id) if user_id is not None else (session_id, tenant_id)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM session_messages "
                f"WHERE session_id = ? AND tenant_id = ?{owner_clause} ORDER BY id",
                params,
            ).fetchall()
        return [{"role": role, "content": content} for role, content in rows]

    def list_sessions(
        self, tenant_id: str, user_id: str, limit: int = 100
    ) -> List[Dict[str, object]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT session_id, COUNT(*) AS message_count, MIN(created_at) AS created_at, "
                "MAX(created_at) AS updated_at, "
                "COALESCE((SELECT content FROM session_messages first_message "
                "WHERE first_message.tenant_id = session_messages.tenant_id "
                "AND first_message.user_id = session_messages.user_id "
                "AND first_message.session_id = session_messages.session_id "
                "AND first_message.role = 'user' ORDER BY first_message.id LIMIT 1), '') AS title "
                "FROM session_messages WHERE tenant_id = ? AND user_id = ? "
                "GROUP BY session_id ORDER BY updated_at DESC LIMIT ?",
                (tenant_id, user_id, limit),
            ).fetchall()
        return [
            {
                "session_id": row[0], "message_count": row[1], "created_at": row[2],
                "updated_at": row[3], "title": row[4],
            }
            for row in rows
        ]

    # ----- SessionStore protocol -----
    def load_messages(self, session_id: str) -> List[Dict[str, str]]:
        sid, tid, uid = _split_tenant_session(session_id)
        return self.get_session_messages(sid, tenant_id=tid, user_id=uid or None)

    def list_message_refs(self, session_id: str) -> List[str]:
        sid, tid, uid = _split_tenant_session(session_id)
        owner_clause = " AND user_id = ?" if uid else ""
        params = (sid, tid, uid) if uid else (sid, tid)
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT id, role FROM session_messages "
                f"WHERE session_id = ? AND tenant_id = ?{owner_clause} ORDER BY id",
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

    def save_trace(self, request_id: str, session_id: str, payload: Dict,
                   tenant_id: str = "default") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO traces(request_id, session_id, tenant_id, payload) "
                "VALUES (?, ?, ?, ?)",
                (request_id, session_id, tenant_id,
                 json.dumps(payload, ensure_ascii=False)),
            )

    def get_trace(self, request_id: str, tenant_id: Optional[str] = None) -> Dict:
        tenant_clause = " AND tenant_id = ?" if tenant_id is not None else ""
        params = (request_id, tenant_id) if tenant_id is not None else (request_id,)
        with self._connect() as conn:
            row = conn.execute(
                f"SELECT payload FROM traces WHERE request_id = ?{tenant_clause}",
                params,
            ).fetchone()
        if not row:
            raise KeyError(request_id)
        return json.loads(row[0])


def _split_tenant_session(session_id: str):
    """SessionStore.append_message 接口只有单参数；我们用 'tenant|session' 串编码。

    没有 '|' 前缀的旧 session_id 默认 tenant=default，向后兼容。
    """
    parts = session_id.split("|", 2)
    if len(parts) == 3:
        tenant, user_id, sid = parts
        return sid, tenant, user_id
    if len(parts) == 2:
        tenant, sid = parts
        return sid, tenant, ""
    return session_id, "default", ""
