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


def test_sqlite_store_lists_and_loads_sessions_by_stable_user():
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteStore(os.path.join(tmp, "agent.db"))
        store.save_session_message(
            "session-a", "user", "我的问题", tenant_id="tenant-a",
            user_id="user-1", request_id="request-a",
        )
        store.save_session_message(
            "session-a", "assistant", "回答", tenant_id="tenant-a",
            user_id="user-1", request_id="request-a",
        )
        store.save_session_message(
            "session-b", "user", "其他用户", tenant_id="tenant-a",
            user_id="user-2", request_id="request-b",
        )

        sessions = store.list_sessions("tenant-a", "user-1")

        assert [(item["session_id"], item["message_count"], item["title"]) for item in sessions] == [
            ("session-a", 2, "我的问题")
        ]
        assert store.get_session_messages(
            "session-a", tenant_id="tenant-a", user_id="user-1"
        ) == [
            {"role": "user", "content": "我的问题"},
            {"role": "assistant", "content": "回答"},
        ]
        assert store.get_session_messages(
            "session-a", tenant_id="tenant-a", user_id="user-2"
        ) == []


def test_sqlite_store_protocol_decodes_tenant_user_session_key():
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteStore(os.path.join(tmp, "agent.db"))

        store.append_message(
            "tenant-a|user-1|session-a", "user", "hello", request_id="request-a",
            user_id="user-1",
        )

        assert store.load_messages("tenant-a|user-1|session-a") == [
            {"role": "user", "content": "hello"}
        ]


def test_same_tenant_session_and_request_are_isolated_by_user():
    with tempfile.TemporaryDirectory() as tmp:
        store = SQLiteStore(os.path.join(tmp, "agent.db"))

        assert store.save_session_message(
            "same-session",
            "user",
            "user one",
            tenant_id="tenant-a",
            user_id="user-1",
            request_id="same-request",
        )
        assert store.save_session_message(
            "same-session",
            "user",
            "user two",
            tenant_id="tenant-a",
            user_id="user-2",
            request_id="same-request",
        )

        assert store.get_session_messages(
            "same-session", tenant_id="tenant-a", user_id="user-1"
        )[0]["content"] == "user one"
        assert store.get_session_messages(
            "same-session", tenant_id="tenant-a", user_id="user-2"
        )[0]["content"] == "user two"
