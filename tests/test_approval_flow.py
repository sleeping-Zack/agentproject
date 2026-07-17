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
    )

    loaded = store.get(pending.approval_id)
    assert loaded.status == "pending"
    assert loaded.tool_name == "fetch_external_data"

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
