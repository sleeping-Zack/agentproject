from safety.auth import AuthContext
from agent.policies import ToolPolicy
from agent.tools.registry import build_default_tool_registry
from mcp_adapter.server import MCPToolServer
from mcp_server import stdio_auth_context
from services.approval_store import SQLiteApprovalStore


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


def test_stdio_mcp_default_auth_context_is_local_user():
    context = stdio_auth_context()

    assert context.tenant_id == "mcp-local"
    assert context.user_role == "user"
    assert context.principal_id == "mcp-stdio"
