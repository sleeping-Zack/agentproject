from typing import Callable, Dict, Optional
from uuid import uuid4

from agent.policies import PolicyAction, ToolPolicy
from agent.tools.registry import build_default_tool_registry
from safety.auth import AuthContext
from safety.security import sensitive_tool_approval
from services.approval_store import SQLiteApprovalStore


class MCPToolServer:
    """Small JSON-RPC MCP adapter for stdio and HTTP use.

    It implements the MCP methods needed by common clients for discovery and
    tool invocation: `initialize`, `tools/list`, and `tools/call`.
    """

    def __init__(
        self,
        tool_handlers: Dict[str, Callable[[Dict], str]],
        policy: Optional[ToolPolicy] = None,
        approval_store: Optional[SQLiteApprovalStore] = None,
    ) -> None:
        self.tool_handlers = tool_handlers
        self.registry = build_default_tool_registry(tool_handlers.keys())
        self.policy = policy
        self.approval_store = approval_store

    def handle_jsonrpc(self, request: Dict, context: Optional[AuthContext] = None) -> Dict:
        method = request.get("method")
        request_id = request.get("id")
        context = context or AuthContext()
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
                result = self._call_tool(request.get("params", {}), context)
            else:
                return self._error(request_id, -32601, f"Unknown method: {method}")
            return {"jsonrpc": "2.0", "id": request_id, "result": result}
        except (KeyError, ValueError, PermissionError) as exc:
            return self._error(request_id, -32602, str(exc))

    def _call_tool(self, params: Dict, context: AuthContext) -> Dict:
        name = params["name"]
        arguments = params.get("arguments", {})
        if name not in self.tool_handlers:
            raise ValueError(f"Unknown tool: {name}")
        if self.policy is not None:
            scene = params.get("scene") or arguments.get("scene")
            if scene is None and name == "fetch_external_data":
                scene = "report"
            decision = self.policy.decide(
                tenant_id=context.tenant_id,
                user_role=context.user_role,
                scene=scene or "mcp",
                tool_name=name,
                args=arguments,
            )
            if decision.action == PolicyAction.DENY:
                return {
                    "status": "denied",
                    "reason": decision.reason,
                    "content": [{"type": "text", "text": f"工具调用被拒绝：{decision.reason}"}],
                }
            if decision.action == PolicyAction.NEED_REDACTION:
                return {
                    "status": "denied",
                    "reason": decision.reason,
                    "redactions": decision.redactions,
                    "content": [{"type": "text", "text": "工具参数包含敏感字段，已拒绝执行。"}],
                }
            if decision.action == PolicyAction.NEED_APPROVAL:
                approved = self._resolve_approval(params, context, name, arguments)
                if approved is not True:
                    return approved
        if name == "fetch_external_data":
            with sensitive_tool_approval(name):
                result = self.tool_handlers[name](arguments)
        else:
            result = self.tool_handlers[name](arguments)
        return {"content": [{"type": "text", "text": str(result)}]}

    def _resolve_approval(
        self,
        params: Dict,
        context: AuthContext,
        tool_name: str,
        arguments: Dict,
    ):
        if self.approval_store is None:
            return {
                "status": "denied",
                "reason": "approval store is not configured",
                "content": [{"type": "text", "text": "审批存储未配置，拒绝执行敏感工具。"}],
            }
        approval_id = params.get("approval_id")
        if approval_id:
            approval = self.approval_store.get(approval_id)
            if approval.tenant_id != context.tenant_id or approval.tool_name != tool_name:
                return {
                    "status": "denied",
                    "reason": "approval does not match tenant or tool",
                    "content": [{"type": "text", "text": "审批记录与当前租户或工具不匹配。"}],
                }
            if approval.is_approved:
                if approval.args != arguments:
                    return {
                        "status": "denied",
                        "approval_id": approval_id,
                        "reason": "approval arguments do not match",
                        "content": [{"type": "text", "text": "审批参数与当前工具参数不匹配。"}],
                    }
                return True
            if approval.is_denied:
                return {
                    "status": "denied",
                    "approval_id": approval_id,
                    "reason": "approval_denied",
                    "content": [{"type": "text", "text": "敏感工具调用审批已被拒绝。"}],
                }
            return {
                "status": "pending_approval",
                "approval_id": approval_id,
                "content": [{"type": "text", "text": "等待敏感工具调用审批。"}],
            }

        approval = self.approval_store.create_pending(
            request_id=params.get("request_id") or str(uuid4()),
            tenant_id=context.tenant_id,
            user_role=context.user_role,
            tool_name=tool_name,
            args=arguments,
            reason=f"tool requires approval: {tool_name}",
        )
        return {
            "status": "pending_approval",
            "approval_id": approval.approval_id,
            "content": [{"type": "text", "text": "等待敏感工具调用审批。"}],
        }

    @staticmethod
    def _error(request_id, code: int, message: str) -> Dict:
        return {"jsonrpc": "2.0", "id": request_id, "error": {"code": code, "message": message}}
