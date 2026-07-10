import os
import sqlite3
import tempfile
from contextlib import closing

from services.persistence import SQLiteStore


def test_sqlite_store_isolates_messages_by_tenant():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "agent.db")
        store = SQLiteStore(db_path)

        store.save_session_message("s1", "user", "a的消息", tenant_id="a")
        store.save_session_message("s1", "user", "b的消息", tenant_id="b")

        a = store.get_session_messages("s1", tenant_id="a")
        b = store.get_session_messages("s1", tenant_id="b")
        assert [m["content"] for m in a] == ["a的消息"]
        assert [m["content"] for m in b] == ["b的消息"]


def test_sqlite_store_protocol_decodes_tenant_session_key():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "agent.db")
        store = SQLiteStore(db_path)

        # SessionStore 协议把 tenant 编进 session_id 前缀
        store.append_message("acme|s1", "user", "hi acme")
        store.append_message("nous|s1", "user", "hi nous")

        assert store.load_messages("acme|s1") == [{"role": "user", "content": "hi acme"}]
        assert store.load_messages("nous|s1") == [{"role": "user", "content": "hi nous"}]


def test_sqlite_store_backward_compatible_without_tenant():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "agent.db")
        store = SQLiteStore(db_path)

        # 没有 '|' 前缀的旧调用路径
        store.append_message("s1", "user", "legacy")
        assert store.load_messages("s1") == [{"role": "user", "content": "legacy"}]


def test_sqlite_store_deduplicates_request_role_but_not_tenant():
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteStore(os.path.join(tmp, "agent.db"))

        assert store.save_session_message(
            "s1", "user", "first", tenant_id="a", request_id="req-1"
        )
        assert not store.save_session_message(
            "s1", "user", "retry", tenant_id="a", request_id="req-1"
        )
        assert store.save_session_message(
            "s1", "assistant", "answer", tenant_id="a", request_id="req-1"
        )
        assert store.save_session_message(
            "s1", "user", "tenant b", tenant_id="b", request_id="req-1"
        )

        assert store.get_session_messages("s1", tenant_id="a") == [
            {"role": "user", "content": "first"},
            {"role": "assistant", "content": "answer"},
        ]
        assert store.get_session_messages("s1", tenant_id="b") == [
            {"role": "user", "content": "tenant b"}
        ]


def test_sqlite_store_migrates_legacy_null_request_ids():
    with tempfile.TemporaryDirectory() as tmp:
        db_path = os.path.join(tmp, "legacy.db")
        with closing(sqlite3.connect(db_path)) as conn:
            conn.execute(
                "CREATE TABLE session_messages ("
                "id INTEGER PRIMARY KEY AUTOINCREMENT,"
                "session_id TEXT NOT NULL,"
                "tenant_id TEXT NOT NULL DEFAULT 'default',"
                "role TEXT NOT NULL,"
                "content TEXT NOT NULL,"
                "created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP)"
            )
            conn.execute(
                "INSERT INTO session_messages(session_id, tenant_id, role, content) "
                "VALUES ('legacy', 'default', 'user', 'old message')"
            )
            conn.commit()

        store = SQLiteStore(db_path)

        assert store.get_session_messages("legacy") == [
            {"role": "user", "content": "old message"}
        ]
        assert store.save_session_message(
            "legacy", "assistant", "new answer", request_id="req-new"
        )
        with closing(sqlite3.connect(db_path)) as conn:
            request_ids = conn.execute(
                "SELECT request_id FROM session_messages ORDER BY id"
            ).fetchall()
            index_rows = conn.execute("PRAGMA index_list(session_messages)").fetchall()

        assert request_ids == [(None,), ("req-new",)]
        assert any(
            row[1] == "idx_session_message_idempotency" and row[2] == 1
            for row in index_rows
        )
