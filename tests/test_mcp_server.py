from mcp_adapter.server import MCPToolServer
from observability.context import request_context
from safety.security import is_sensitive_tool_approved
from services.approval_store import SQLiteApprovalStore


def test_mcp_initialize_and_list_tools():
    server = MCPToolServer(
        tool_handlers={
            "get_weather": lambda args: f"城市{args['city']}天气为晴天",
        }
    )

    init_response = server.handle_jsonrpc(
        {"jsonrpc": "2.0", "id": 1, "method": "initialize", "params": {}}
    )
    list_response = server.handle_jsonrpc(
        {"jsonrpc": "2.0", "id": 2, "method": "tools/list", "params": {}}
    )

    assert init_response["result"]["serverInfo"]["name"] == "sweeper-agent-mcp"
    assert any(tool["name"] == "get_weather" for tool in list_response["result"]["tools"])


def test_mcp_call_tool_executes_registered_handler():
    server = MCPToolServer(
        tool_handlers={
            "get_weather": lambda args: f"城市{args['city']}天气为晴天",
        }
    )

    response = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 3,
            "method": "tools/call",
            "params": {"name": "get_weather", "arguments": {"city": "深圳"}},
        }
    )

    assert response["result"]["content"][0]["text"] == "城市深圳天气为晴天"


def test_mcp_call_unknown_tool_returns_error():
    server = MCPToolServer(tool_handlers={})

    response = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 4,
            "method": "tools/call",
            "params": {"name": "missing", "arguments": {}},
        }
    )

    assert response["error"]["code"] == -32602


def test_requires_approval_tool_cannot_run_without_policy(tmp_path):
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
        approval_store=approval_store,
    )
    arguments = {
        "model": "S20",
        "issue_type": "repair",
        "description": "设备持续显示 E12。",
    }

    response = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 5,
            "method": "tools/call",
            "params": {
                "name": "create_support_ticket",
                "request_id": "ticket-without-policy",
                "arguments": arguments,
            },
        }
    )

    pending = response["result"]
    assert pending["status"] == "pending_approval"
    assert pending["approval_id"]
    assert calls == []

    approval_store.approve(pending["approval_id"], decided_by="operator-1")
    completed = server.handle_jsonrpc(
        {
            "jsonrpc": "2.0",
            "id": 6,
            "method": "tools/call",
            "params": {
                "name": "create_support_ticket",
                "request_id": "ticket-without-policy-resume",
                "approval_id": pending["approval_id"],
                "arguments": arguments,
            },
        }
    )["result"]

    assert completed["content"][0]["text"] == "ticket-created"
    assert calls == [
        {
            "args": arguments,
            "approval_id": pending["approval_id"],
            "approved": True,
        }
    ]
