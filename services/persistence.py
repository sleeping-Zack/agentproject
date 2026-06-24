import json
import os
import sqlite3
from typing import Dict, List


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
            self._ensure_column(conn, "traces", "tenant_id",
                                "TEXT NOT NULL DEFAULT 'default'")
            conn.execute(
                "CREATE INDEX IF NOT EXISTS idx_session_messages_tenant_session "
                "ON session_messages(tenant_id, session_id)"
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

    def save_session_message(self, session_id: str, role: str, content: str,
                             tenant_id: str = "default") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO session_messages(session_id, tenant_id, role, content) "
                "VALUES (?, ?, ?, ?)",
                (session_id, tenant_id, role, content),
            )

    def get_session_messages(self, session_id: str,
                             tenant_id: str = "default") -> List[Dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM session_messages "
                "WHERE session_id = ? AND tenant_id = ? ORDER BY id",
                (session_id, tenant_id),
            ).fetchall()
        return [{"role": role, "content": content} for role, content in rows]

    # ----- SessionStore protocol -----
    def load_messages(self, session_id: str) -> List[Dict[str, str]]:
        sid, tid = _split_tenant_session(session_id)
        return self.get_session_messages(sid, tenant_id=tid)

    def append_message(self, session_id: str, role: str, content: str) -> None:
        sid, tid = _split_tenant_session(session_id)
        self.save_session_message(sid, role, content, tenant_id=tid)

    def save_trace(self, request_id: str, session_id: str, payload: Dict,
                   tenant_id: str = "default") -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO traces(request_id, session_id, tenant_id, payload) "
                "VALUES (?, ?, ?, ?)",
                (request_id, session_id, tenant_id,
                 json.dumps(payload, ensure_ascii=False)),
            )

    def get_trace(self, request_id: str) -> Dict:
        with self._connect() as conn:
            row = conn.execute(
                "SELECT payload FROM traces WHERE request_id = ?",
                (request_id,),
            ).fetchone()
        if not row:
            raise KeyError(request_id)
        return json.loads(row[0])


def _split_tenant_session(session_id: str):
    """SessionStore.append_message 接口只有单参数；我们用 'tenant|session' 串编码。

    没有 '|' 前缀的旧 session_id 默认 tenant=default，向后兼容。
    """
    if "|" in session_id:
        tenant, sid = session_id.split("|", 1)
        return sid, tenant
    return session_id, "default"
