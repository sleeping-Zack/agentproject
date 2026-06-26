import pytest

from agent.tools.registry import ToolSpec, build_default_tool_registry


def test_registry_blocks_tools_outside_allowlist():
    registry = build_default_tool_registry(allowed_tools=["rag_summarize", "get_weather"])

    registry.require_allowed("get_weather")
    with pytest.raises(PermissionError):
        registry.require_allowed("fetch_external_data")


def test_registry_exports_mcp_style_manifest():
    registry = build_default_tool_registry(allowed_tools=["get_weather"])
    manifest = registry.as_mcp_manifest()

    assert manifest["protocol"] == "mcp"
    assert manifest["tools"] == [
        {
            "name": "get_weather",
            "description": "获取指定城市的天气和环境信息",
            "scope": "environment:read",
            "input_schema": {
                "type": "object",
                "properties": {"city": {"type": "string"}},
                "required": ["city"],
            },
            "risk_level": "low",
            "side_effect": "read",
            "requires_approval": False,
            "timeout_seconds": 30,
        }
    ]


def test_registry_rejects_duplicate_tool_names():
    registry = build_default_tool_registry(allowed_tools=[])

    with pytest.raises(ValueError):
        registry.register(
            ToolSpec(
                name="get_weather",
                description="duplicate",
                scope="environment:read",
                input_schema={"type": "object"},
            )
        )
