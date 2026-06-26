from agent.policies import PolicyAction, ToolPolicy
from agent.tools.registry import build_default_tool_registry


def test_policy_allows_safe_read_tools():
    registry = build_default_tool_registry(["rag_summarize", "get_weather"])
    policy = ToolPolicy(tool_registry=registry)

    decision = policy.decide(
        tenant_id="tenant-a",
        user_role="user",
        scene="qa",
        tool_name="rag_summarize",
        args={"query": "怎么保养滤网"},
    )

    assert decision.action == PolicyAction.ALLOW


def test_policy_requires_approval_for_sensitive_report_data():
    registry = build_default_tool_registry(["fetch_external_data"])
    policy = ToolPolicy(tool_registry=registry)

    decision = policy.decide(
        tenant_id="tenant-a",
        user_role="user",
        scene="report",
        tool_name="fetch_external_data",
        args={"user_id": "u-1", "month": "2026-06"},
    )

    assert decision.action == PolicyAction.NEED_APPROVAL
    assert "requires approval" in decision.reason


def test_policy_denies_sensitive_data_outside_report_scene():
    registry = build_default_tool_registry(["fetch_external_data"])
    policy = ToolPolicy(tool_registry=registry)

    decision = policy.decide(
        tenant_id="tenant-a",
        user_role="user",
        scene="chat",
        tool_name="fetch_external_data",
        args={"user_id": "u-1", "month": "2026-06"},
    )

    assert decision.action == PolicyAction.DENY


def test_policy_allows_admin_to_read_report_data():
    registry = build_default_tool_registry(["fetch_external_data"])
    policy = ToolPolicy(tool_registry=registry)

    decision = policy.decide(
        tenant_id="tenant-a",
        user_role="admin",
        scene="report",
        tool_name="fetch_external_data",
        args={"user_id": "u-1", "month": "2026-06"},
    )

    assert decision.action == PolicyAction.ALLOW
