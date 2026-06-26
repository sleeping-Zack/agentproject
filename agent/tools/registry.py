from dataclasses import dataclass
from typing import Dict, Iterable, List


@dataclass(frozen=True)
class ToolSpec:
    name: str
    description: str
    scope: str
    input_schema: Dict
    risk_level: str = "low"
    side_effect: str = "read"
    requires_approval: bool = False
    timeout_seconds: int = 30

    def as_manifest_item(self) -> Dict:
        return {
            "name": self.name,
            "description": self.description,
            "scope": self.scope,
            "input_schema": self.input_schema,
            "risk_level": self.risk_level,
            "side_effect": self.side_effect,
            "requires_approval": self.requires_approval,
            "timeout_seconds": self.timeout_seconds,
        }


class ToolRegistry:
    def __init__(self, allowed_tools: Iterable[str]) -> None:
        self.allowed_tools = set(allowed_tools)
        self._tools: Dict[str, ToolSpec] = {}

    def register(self, spec: ToolSpec) -> None:
        if spec.name in self._tools:
            raise ValueError(f"Tool already registered: {spec.name}")
        self._tools[spec.name] = spec

    def require_allowed(self, tool_name: str) -> None:
        if tool_name not in self.allowed_tools:
            raise PermissionError(f"Tool is not allowed: {tool_name}")

    def allowed_specs(self) -> List[ToolSpec]:
        return [spec for name, spec in self._tools.items() if name in self.allowed_tools]

    def get_spec(self, tool_name: str) -> ToolSpec | None:
        return self._tools.get(tool_name)

    def as_mcp_manifest(self) -> Dict:
        return {
            "protocol": "mcp",
            "tools": [spec.as_manifest_item() for spec in self.allowed_specs()],
        }


def build_default_tool_registry(allowed_tools: Iterable[str]) -> ToolRegistry:
    registry = ToolRegistry(allowed_tools=allowed_tools)
    registry.register(
        ToolSpec(
            name="rag_summarize",
            description="从向量知识库检索并总结扫地机器人资料",
            scope="knowledge:read",
            input_schema={
                "type": "object",
                "properties": {"query": {"type": "string"}},
                "required": ["query"],
            },
        )
    )
    registry.register(
        ToolSpec(
            name="get_weather",
            description="获取指定城市的天气和环境信息",
            scope="environment:read",
            input_schema={
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
        )
    )
    registry.register(
        ToolSpec(
            name="get_user_location",
            description="获取当前用户所在城市",
            scope="user:read",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
    )
    registry.register(
        ToolSpec(
            name="get_user_id",
            description="获取当前用户 ID",
            scope="user:read",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
    )
    registry.register(
        ToolSpec(
            name="get_current_month",
            description="获取当前报告月份",
            scope="time:read",
            input_schema={"type": "object", "properties": {}, "required": []},
        )
    )
    registry.register(
        ToolSpec(
            name="fetch_external_data",
            description="获取指定用户在指定月份的设备使用记录",
            scope="usage_record:read",
            input_schema={
                "type": "object",
                "properties": {
                    "user_id": {"type": "string"},
                    "month": {"type": "string"},
                },
                "required": ["user_id", "month"],
            },
            risk_level="medium",
            side_effect="read_sensitive",
            requires_approval=True,
            timeout_seconds=10,
        )
    )
    registry.register(
        ToolSpec(
            name="fill_context_for_report",
            description="标记后续模型调用进入报告生成场景",
            scope="runtime:write",
            input_schema={"type": "object", "properties": {}, "required": []},
            side_effect="runtime_context",
        )
    )
    return registry
