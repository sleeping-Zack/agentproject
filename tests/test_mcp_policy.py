from safety.auth import AuthContext
from agent.policies import ToolPolicy
from agent.tools.registry import build_default_tool_registry
from mcp_adapter.server import MCPToolServer
from mcp_server import stdio_auth_context
from services.approval_store import SQLiteApprovalStore
from observability.context import request_context
from safety.security import is_sensitive_tool_approved


def test_mcp_sensitive_tool_returns_pending_approval_without_invoking_handler(tmp_path):
    called = {"value": False}

    def raw_handler(args):
        called["value"] = True
        return "sensitive data"

    approval_store = SQLiteApprovalStore(str(tmp_path / "approvals.db"))
    server = MCPToolServer(
        tool_handlers={"fetch_external_data": raw_handler},
        policy=ToolPolicy(build_default_tool_registry(["fetch_external_data"])),
        approval_store=approval_store,
    )

    response = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 1,
            "method": "tools/call",
            "params": {
                "name": "fetch_external_data",
                "scene": "report",
                "arguments": {"user_id": "1001", "month": "2025-09"},
            },
        },
        context=AuthContext(tenant_id="tenant-a", user_role="user", principal_id="user:tenant-a"),
    )

    result = response["result"]
    assert result["status"] == "pending_approval"
    assert result["approval_id"]
    assert called["value"] is False
    assert approval_store.get(result["approval_id"]).tenant_id == "tenant-a"


def test_mcp_sensitive_tool_defaults_to_approval_scene(tmp_path):
    approval_store = SQLiteApprovalStore(str(tmp_path / "approvals.db"))
    server = MCPToolServer(
        tool_handlers={"fetch_external_data": lambda args: "sensitive data"},
        policy=ToolPolicy(build_default_tool_registry(["fetch_external_data"])),
        approval_store=approval_store,
    )

    response = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 2,
            "method": "tools/call",
            "params": {
                "name": "fetch_external_data",
                "arguments": {"user_id": "1001", "month": "2025-09"},
            },
        },
        context=AuthContext(tenant_id="tenant-a", user_role="user", principal_id="user:tenant-a"),
    )

    assert response["result"]["status"] == "pending_approval"


def test_mcp_manifest_requires_approval_even_without_policy(tmp_path):
    called = {"value": False}

    def raw_handler(_args):
        called["value"] = True
        return "sensitive data"

    server = MCPToolServer(
        tool_handlers={"fetch_external_data": raw_handler},
        approval_store=SQLiteApprovalStore(str(tmp_path / "approvals.db")),
    )
    response = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 22,
            "method": "tools/call",
            "params": {
                "name": "fetch_external_data",
                "arguments": {"user_id": "1001", "month": "2025-09"},
            },
        },
        context=AuthContext(
            tenant_id="tenant-a",
            user_role="user",
            principal_id="user:tenant-a",
        ),
    )

    assert response["result"]["status"] == "pending_approval"
    assert called["value"] is False


def test_stdio_mcp_default_auth_context_is_local_user():
    context = stdio_auth_context()

    assert context.tenant_id == "mcp-local"
    assert context.user_role == "user"
    assert context.principal_id == "mcp-stdio"


def test_mcp_support_ticket_runs_only_after_matching_approval(tmp_path):
    calls = []

    def create_ticket(args):
        calls.append(
            {
                "args": args,
                "approval_id": request_context().extra.get("approval_id"),
                "approved": is_sensitive_tool_approved("create_support_ticket"),
            }
        )
        return "ticket-created"

    approval_store = SQLiteApprovalStore(str(tmp_path / "approvals.db"))
    server = MCPToolServer(
        tool_handlers={"create_support_ticket": create_ticket},
        policy=ToolPolicy(build_default_tool_registry(["create_support_ticket"])),
        approval_store=approval_store,
    )
    context = AuthContext(
        tenant_id="tenant-a",
        user_role="user",
        principal_id="user:tenant-a",
    )
    arguments = {
        "model": "S20",
        "issue_type": "repair",
        "description": "设备持续显示 E12，重装水箱后仍未恢复。",
    }
    resumed_arguments = {**arguments, "error_code": ""}

    pending = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {
                "name": "create_support_ticket",
                "request_id": "mcp-ticket-1",
                "arguments": arguments,
            },
        },
        context=context,
    )["result"]

    assert pending["status"] == "pending_approval"
    assert calls == []
    approval_store.approve(pending["approval_id"], decided_by="operator-1")

    completed = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {
                "name": "create_support_ticket",
                "request_id": "mcp-ticket-1-resume",
                "approval_id": pending["approval_id"],
                "arguments": resumed_arguments,
            },
        },
        context=context,
    )["result"]

    assert completed["content"][0]["text"] == "ticket-created"
    assert calls == [
        {
            "args": resumed_arguments,
            "approval_id": pending["approval_id"],
            "approved": True,
        }
    ]
