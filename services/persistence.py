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
                "role TEXT NOT NULL,"
                "content TEXT NOT NULL,"
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.execute(
                "CREATE TABLE IF NOT EXISTS traces ("
                "request_id TEXT PRIMARY KEY,"
                "session_id TEXT NOT NULL,"
                "payload TEXT NOT NULL,"
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )

    def save_session_message(self, session_id: str, role: str, content: str) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT INTO session_messages(session_id, role, content) VALUES (?, ?, ?)",
                (session_id, role, content),
            )

    def get_session_messages(self, session_id: str) -> List[Dict[str, str]]:
        with self._connect() as conn:
            rows = conn.execute(
                "SELECT role, content FROM session_messages WHERE session_id = ? ORDER BY id",
                (session_id,),
            ).fetchall()
        return [{"role": role, "content": content} for role, content in rows]

    def save_trace(self, request_id: str, session_id: str, payload: Dict) -> None:
        with self._connect() as conn:
            conn.execute(
                "INSERT OR REPLACE INTO traces(request_id, session_id, payload) VALUES (?, ?, ?)",
                (request_id, session_id, json.dumps(payload, ensure_ascii=False)),
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
