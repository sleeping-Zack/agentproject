import json
from concurrent.futures import ThreadPoolExecutor

import pytest

import agent.tools.agent_tools as agent_tools_module
from observability.context import bind_request_context
from safety.security import sensitive_tool_approval
from services.approval_store import SQLiteApprovalStore
from services.artifact_store import SQLiteArtifactStore


TICKET_ARGS = {
    "model": "S20",
    "issue_type": "repair",
    "description": "设备持续显示 E12，重装水箱后仍未恢复。",
    "error_code": "E12",
}


def test_new_tools_are_exposed_to_the_react_agent():
    names = {tool.name for tool in agent_tools_module.REACT_TOOLS}

    assert {
        "lookup_error_code",
        "get_product_specs",
        "create_support_ticket",
    } <= names


def _approved_ticket(approval_store, *, request_id="approval-request-1"):
    pending = approval_store.create_pending(
        request_id=request_id,
        tenant_id="tenant-a",
        user_role="user",
        tool_name="create_support_ticket",
        args=dict(TICKET_ARGS),
        reason="test approval",
        principal_id="user-a",
    )
    return approval_store.approve(pending.approval_id, decided_by="reviewer")


def test_create_support_ticket_persists_one_artifact_per_approval(tmp_path, monkeypatch):
    store = SQLiteArtifactStore(str(tmp_path / "artifacts.db"))
    approvals = SQLiteApprovalStore(str(tmp_path / "approvals.db"))
    monkeypatch.setattr(agent_tools_module, "artifact_store", store)
    monkeypatch.setattr(agent_tools_module, "approval_store", approvals)
    approval = _approved_ticket(approvals)

    with bind_request_context(
        request_id="execution-1",
        tenant_id="tenant-a",
        user_id="user-a",
        approval_id=approval.approval_id,
    ), sensitive_tool_approval("create_support_ticket"):
        first = json.loads(agent_tools_module.create_support_ticket.invoke(TICKET_ARGS))

    with bind_request_context(
        request_id="execution-2",
        tenant_id="tenant-a",
        user_id="user-a",
        approval_id=approval.approval_id,
    ), sensitive_tool_approval("create_support_ticket"):
        duplicate = json.loads(agent_tools_module.create_support_ticket.invoke(TICKET_ARGS))

    assert duplicate["ticket_id"] == first["ticket_id"]
    assert duplicate["status"] == "open"
    artifacts = store.list_artifacts(approval.approval_id, tenant_id="tenant-a")
    assert len(artifacts) == 1
    assert artifacts[0].artifact_type == "support_ticket"
    assert artifacts[0].payload["principal_id"] == "user-a"


def test_create_support_ticket_rejects_direct_unapproved_invocation():
    with pytest.raises(PermissionError):
        agent_tools_module.create_support_ticket.invoke(TICKET_ARGS)


def test_create_support_ticket_requires_concrete_approval_id():
    with bind_request_context(
        request_id="execution-1",
        tenant_id="tenant-a",
        user_id="user-a",
    ), sensitive_tool_approval("create_support_ticket"):
        with pytest.raises(PermissionError, match="审批"):
            agent_tools_module.create_support_ticket.invoke(TICKET_ARGS)


def test_create_support_ticket_rejects_cross_tenant_approval(tmp_path, monkeypatch):
    store = SQLiteArtifactStore(str(tmp_path / "artifacts.db"))
    approvals = SQLiteApprovalStore(str(tmp_path / "approvals.db"))
    monkeypatch.setattr(agent_tools_module, "artifact_store", store)
    monkeypatch.setattr(agent_tools_module, "approval_store", approvals)
    approval = _approved_ticket(approvals)

    with bind_request_context(
        request_id="execution-cross-tenant",
        tenant_id="tenant-b",
        user_id="user-a",
        approval_id=approval.approval_id,
    ), sensitive_tool_approval("create_support_ticket"):
        with pytest.raises(PermissionError, match="不匹配"):
            agent_tools_module.create_support_ticket.invoke(TICKET_ARGS)

    assert store.list_artifacts(approval.approval_id, tenant_id="tenant-b") == []


def test_create_support_ticket_rejects_changed_args_after_approval(tmp_path, monkeypatch):
    store = SQLiteArtifactStore(str(tmp_path / "artifacts.db"))
    approvals = SQLiteApprovalStore(str(tmp_path / "approvals.db"))
    monkeypatch.setattr(agent_tools_module, "artifact_store", store)
    monkeypatch.setattr(agent_tools_module, "approval_store", approvals)
    approval = _approved_ticket(approvals)
    changed = {**TICKET_ARGS, "description": "另一台设备的问题"}

    with bind_request_context(
        request_id="execution-changed",
        tenant_id="tenant-a",
        user_id="user-a",
        approval_id=approval.approval_id,
    ), sensitive_tool_approval("create_support_ticket"):
        with pytest.raises(PermissionError, match="不匹配"):
            agent_tools_module.create_support_ticket.invoke(changed)

    assert store.list_artifacts(approval.approval_id, tenant_id="tenant-a") == []


def test_create_support_ticket_is_idempotent_under_concurrent_resume(tmp_path, monkeypatch):
    store = SQLiteArtifactStore(str(tmp_path / "artifacts.db"))
    approvals = SQLiteApprovalStore(str(tmp_path / "approvals.db"))
    monkeypatch.setattr(agent_tools_module, "artifact_store", store)
    monkeypatch.setattr(agent_tools_module, "approval_store", approvals)
    approval = _approved_ticket(approvals)

    def resume(execution_id):
        with bind_request_context(
            request_id=execution_id,
            tenant_id="tenant-a",
            user_id="user-a",
            approval_id=approval.approval_id,
        ), sensitive_tool_approval("create_support_ticket"):
            return json.loads(agent_tools_module.create_support_ticket.invoke(TICKET_ARGS))

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = list(executor.map(resume, ("execution-a", "execution-b")))

    assert results[0]["ticket_id"] == results[1]["ticket_id"]
    assert len(store.list_artifacts(approval.approval_id, tenant_id="tenant-a")) == 1


def test_structured_catalog_tools_return_json():
    specs = json.loads(agent_tools_module.get_product_specs.invoke({"model": "s10"}))
    error = json.loads(
        agent_tools_module.lookup_error_code.invoke(
            {"model": "S20", "error_code": "e12"}
        )
    )

    assert specs["model"] == "S10"
    assert specs["catalog_version"] == "demo-v1"
    assert error["model"] == "S20"
    assert error["error_code"] == "E12"


def test_structured_catalog_tools_return_empty_string_for_unknown_records():
    assert agent_tools_module.get_product_specs.invoke({"model": "missing"}) == ""
    assert agent_tools_module.lookup_error_code.invoke(
        {"model": "S10", "error_code": "E99"}
    ) == ""
