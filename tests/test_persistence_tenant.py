import os
import tempfile

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
