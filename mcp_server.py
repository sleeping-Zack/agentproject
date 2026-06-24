import json
import sys
from typing import Dict

from agent.tools.agent_tools import fetch_external_data, get_weather, rag_summarize
from mcp_adapter.server import MCPToolServer


def build_server() -> MCPToolServer:
    return MCPToolServer(
        tool_handlers={
            "rag_summarize": lambda args: rag_summarize.invoke({"query": args["query"]}),
            "get_weather": lambda args: get_weather.invoke({"city": args["city"]}),
            "fetch_external_data": lambda args: fetch_external_data.invoke(
                {"user_id": args["user_id"], "month": args["month"]}
            ),
        }
    )


def main() -> None:
    server = build_server()
    for line in sys.stdin:
        if not line.strip():
            continue
        request: Dict = json.loads(line)
        response = server.handle_jsonrpc(request)
        print(json.dumps(response, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
