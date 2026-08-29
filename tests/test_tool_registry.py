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


def test_rag_manifest_exposes_optional_information_gap():
    registry = build_default_tool_registry(allowed_tools=["rag_summarize"])

    schema = registry.as_mcp_manifest()["tools"][0]["input_schema"]

    assert schema["required"] == ["query", "information_gap"]
    assert schema["properties"]["information_gap"]["type"] == "string"


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


def test_new_domain_tools_export_risk_and_schema_metadata():
    registry = build_default_tool_registry(
        ["lookup_error_code", "get_product_specs", "create_support_ticket"]
    )
    manifest = {item["name"]: item for item in registry.as_mcp_manifest()["tools"]}

    assert manifest["lookup_error_code"]["input_schema"]["required"] == [
        "model",
        "error_code",
    ]
    assert manifest["get_product_specs"]["scope"] == "product_catalog:read"
    ticket = manifest["create_support_ticket"]
    assert ticket["risk_level"] == "high"
    assert ticket["side_effect"] == "write"
    assert ticket["requires_approval"] is True
    assert ticket["input_schema"]["properties"]["issue_type"]["enum"] == [
        "repair",
        "maintenance",
        "warranty",
        "other",
    ]
