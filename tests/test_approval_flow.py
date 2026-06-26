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
