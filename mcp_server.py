import json
import sys
from typing import Dict

from agent.tools.agent_tools import fetch_external_data, get_weather, rag_summarize
from agent.policies import ToolPolicy
from agent.tools.registry import build_default_tool_registry
from mcp_adapter.server import MCPToolServer
from safety.auth import AuthContext
from services.factories import create_approval_store
from utils.config_handler import agent_conf


def build_server() -> MCPToolServer:
    registry = build_default_tool_registry(agent_conf.get("allowed_tools", []))
    return MCPToolServer(
        tool_handlers={
            "rag_summarize": lambda args: rag_summarize.invoke({"query": args["query"]}),
            "get_weather": lambda args: get_weather.invoke({"city": args["city"]}),
            "fetch_external_data": lambda args: fetch_external_data.invoke(
                {"user_id": args["user_id"], "month": args["month"]}
            ),
        },
        policy=ToolPolicy(registry),
        approval_store=create_approval_store(),
    )


def stdio_auth_context() -> AuthContext:
    return AuthContext(
        tenant_id="mcp-local",
        user_role="user",
        principal_id="mcp-stdio",
    )


def main() -> None:
    server = build_server()
    for line in sys.stdin:
        if not line.strip():
            continue
        request: Dict = json.loads(line)
        response = server.handle_jsonrpc(request, context=stdio_auth_context())
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
