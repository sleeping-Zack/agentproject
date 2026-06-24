from typing import Callable, Dict

from agent.tools.registry import build_default_tool_registry


class MCPToolServer:
    """Small JSON-RPC MCP adapter for stdio and HTTP use.

    It implements the MCP methods needed by common clients for discovery and
    tool invocation: `initialize`, `tools/list`, and `tools/call`.
    """

    def __init__(self, tool_handlers: Dict[str, Callable[[Dict], str]]) -> None:
        self.tool_handlers = tool_handlers
        self.registry = build_default_tool_registry(tool_handlers.keys())

    def handle_jsonrpc(self, request: Dict) -> Dict:
        method = request.get("method")
        request_id = request.get("id")
        try:
            if method == "initialize":
                result = {
                    "protocolVersion": "2024-11-05",
                    "serverInfo": {"name": "sweeper-agent-mcp", "version": "0.2.0"},
                    "capabilities": {"tools": {}},
                }
            elif method == "tools/list":
                result = {"tools": self.registry.as_mcp_manifest()["tools"]}
            elif method == "tools/call":
                result = self._call_tool(request.get("params", {}))
            else:
                return self._error(request_id, -32601, f"Unknown method: {method}")
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except (KeyError, ValueError, PermissionError) as exc:
            return self._error(request_id, -32602, str(exc))

    def _call_tool(self, params: Dict) -> Dict:
        name = params["name"]
        arguments = params.get("arguments", {})
        if name not in self.tool_handlers:
            raise ValueError(f"Unknown tool: {name}")
        result = self.tool_handlers[name](arguments)
        return {"content": [{"type": "text", "text": str(result)}]}

    @staticmethod
    def _error(request_id, code: int, message: str) -> Dict:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
