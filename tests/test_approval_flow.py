import json
import sqlite3

from services.approval_store import SQLiteApprovalStore


def test_approval_store_persists_pending_approve_and_deny(tmp_path):
    store = SQLiteApprovalStore(str(tmp_path / "approvals.db"))

    pending = store.create_pending(
        request_id="req-1",
        tenant_id="tenant-a",
        user_role="user",
        tool_name="fetch_external_data",
        args={"user_id": "u-1", "month": "2026-06"},
        reason="sensitive report data",
        principal_id="user-1",
    )

    loaded = store.get(pending.approval_id)
    assert loaded.status == "pending"
    assert loaded.tool_name == "fetch_external_data"
    assert loaded.principal_id == "user-1"
    assert store.list_approvals("tenant-a", status="pending") == [loaded]

    approved = store.approve(pending.approval_id, decided_by="reviewer")
    assert approved.status == "approved"
    assert approved.decided_by == "reviewer"

    second = store.create_pending(
        request_id="req-2",
        tenant_id="tenant-a",
        user_role="user",
        tool_name="fetch_external_data",
        args={},
        reason="sensitive report data",
    )
    denied = store.deny(second.approval_id, decided_by="reviewer")
    assert denied.status == "denied"


def test_approval_store_is_idempotent_and_decision_is_terminal(tmp_path):
    store = SQLiteApprovalStore(str(tmp_path / "approvals.db"))
    first = store.create_pending(
        request_id="req-idempotent",
        tenant_id="tenant-a",
        user_role="user",
        tool_name="fetch_external_data",
        args={"month": "2026-07"},
        reason="sensitive data",
    )
    duplicate = store.create_pending(
        request_id="req-idempotent",
        tenant_id="tenant-a",
        user_role="user",
        tool_name="fetch_external_data",
        args={"month": "2026-07"},
        reason="sensitive data",
    )

    assert duplicate.approval_id == first.approval_id
    approved = store.approve(first.approval_id, decided_by="operator-a")
    denied_after_approval = store.deny(first.approval_id, decided_by="operator-b")

    assert approved.status == "approved"
    assert denied_after_approval.status == "approved"
    assert denied_after_approval.decided_by == "operator-a"


def test_approval_store_closes_unbound_legacy_pending_records(tmp_path):
    db_path = tmp_path / "legacy-approvals.db"
    with sqlite3.connect(db_path) as conn:
        conn.execute(
            "CREATE TABLE approvals ("
            "approval_id TEXT PRIMARY KEY, request_id TEXT NOT NULL, "
            "tenant_id TEXT NOT NULL, user_role TEXT NOT NULL, "
            "tool_name TEXT NOT NULL, args TEXT NOT NULL, reason TEXT NOT NULL, "
            "status TEXT NOT NULL, created_at TEXT NOT NULL, "
            "decided_at TEXT, decided_by TEXT)"
        )
        conn.execute(
            "INSERT INTO approvals VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
            (
                "legacy-1",
                "request-1",
                "tenant-a",
                "user",
                "fetch_external_data",
                json.dumps({"user_id": "runtime_user", "month": "runtime_month"}),
                "legacy approval",
                "pending",
                "2026-07-01T00:00:00+00:00",
                None,
                None,
            ),
        )

    store = SQLiteApprovalStore(str(db_path))
    migrated = store.get("legacy-1")

    assert migrated.status == "denied"
    assert migrated.decided_by == "system:legacy-scope-migration"
