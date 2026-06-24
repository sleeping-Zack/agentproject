from mcp_adapter.server import MCPToolServer


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
