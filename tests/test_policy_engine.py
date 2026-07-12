from datetime import datetime, timezone

import pytest

from agent.policies import PolicyAction, PolicyContext, PolicyDecision, ToolPolicy
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


def test_default_policy_applies_real_tenant_entitlement_difference():
    registry = build_default_tool_registry(["get_weather"])
    policy = ToolPolicy(tool_registry=registry)

    tenant_a = policy.decide(
        tenant_id="tenant-a",
        user_role="user",
        scene="qa",
        tool_name="get_weather",
        args={"city": "合肥"},
    )
    tenant_b = policy.decide(
        tenant_id="tenant-b",
        user_role="user",
        scene="qa",
        tool_name="get_weather",
        args={"city": "合肥"},
    )

    assert tenant_a.action == PolicyAction.ALLOW
    assert tenant_a.matched_rule_id == "standard-tools"
    assert tenant_b.action == PolicyAction.DENY
    assert tenant_b.reason_code == "tenant_feature_not_enabled"
    assert tenant_b.matched_rule_id == "tenant-b-weather-disabled"


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


def _write_policy(tmp_path, content: str):
    path = tmp_path / "tool_policy.yml"
    path.write_text(content, encoding="utf-8")
    return path


def test_policy_decision_keeps_legacy_fields_and_exposes_audit_metadata():
    registry = build_default_tool_registry(["get_weather"])
    events = []
    policy = ToolPolicy(tool_registry=registry, audit_sink=events.append)

    decision = policy.decide(
        tenant_id="tenant-a",
        principal_id="user-1001",
        user_role="user",
        scene="qa",
        tool_name="get_weather",
        args={"city": "合肥"},
    )

    assert decision.action == PolicyAction.ALLOW
    assert decision.reason == "allowed"
    assert decision.redactions == []
    assert decision.reason_code == "standard_tool_allowed"
    assert decision.policy_version == "tool-policy-v1"
    assert decision.matched_rule_id == "standard-tools"
    assert decision.redacted_args == {"city": "合肥"}
    assert events[0]["principal_id"] == "user-1001"
    assert events[0]["action"] == "allow"

    legacy = PolicyDecision(PolicyAction.ALLOW, "legacy allowed")
    assert legacy.reason == "legacy allowed"
    assert legacy.redactions == []


def test_tenant_rules_use_priority_and_default_deny(tmp_path):
    config_path = _write_policy(
        tmp_path,
        """
version: tenant-policy-v7
default:
  action: deny
  reason_code: no_rule
rules:
  - id: tenant-b-deny
    priority: 200
    match:
      tenants: [tenant-b]
      roles: [user]
      scenes: [qa]
      tools: [get_weather]
    action: deny
    reason_code: tenant_weather_disabled
  - id: qa-weather
    priority: 100
    match:
      tenants: ["*"]
      roles: [user]
      scenes: [qa]
      tools: [get_weather]
    action: allow
    reason_code: weather_allowed
""",
    )
    policy = ToolPolicy(
        build_default_tool_registry(["get_weather"]),
        config_path=config_path,
    )

    allowed = policy.decide("tenant-a", "user", "qa", "get_weather", {"city": "合肥"})
    denied = policy.decide("tenant-b", "user", "qa", "get_weather", {"city": "合肥"})
    defaulted = policy.decide(
        "tenant-a", "user", "report", "get_weather", {"city": "合肥"}
    )

    assert allowed.action == PolicyAction.ALLOW
    assert allowed.matched_rule_id == "qa-weather"
    assert denied.action == PolicyAction.DENY
    assert denied.matched_rule_id == "tenant-b-deny"
    assert defaulted.action == PolicyAction.DENY
    assert defaulted.matched_rule_id == "default-deny"
    assert defaulted.reason_code == "no_rule"


def test_rule_matches_arguments_scope_risk_and_time_window(tmp_path):
    config_path = _write_policy(
        tmp_path,
        """
version: constrained-v1
default:
  action: deny
rules:
  - id: scoped-business-hours
    priority: 100
    match:
      tenants: [tenant-a]
      roles: [user]
      scenes: [qa]
      tools: [get_weather]
      data_scopes: [environment:read]
      risk_levels: [low]
      argument_constraints:
        required: [city, owner]
        forbidden: [raw_payload]
        fields:
          city:
            regex: "^(合肥|深圳)$"
          owner:
            equals_context: principal_id
      time_window:
        timezone: UTC
        days: [mon, tue, wed, thu, fri]
        start: "09:00"
        end: "17:00"
    action: allow
    reason_code: scoped_access
""",
    )
    monday_morning = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
    policy = ToolPolicy(
        build_default_tool_registry(["get_weather"]),
        config_path=config_path,
        clock=lambda: monday_morning,
    )

    context = PolicyContext(
        tenant_id="tenant-a",
        principal_id="user-1001",
        role="user",
        scene="qa",
        tool="get_weather",
        args={"city": "合肥", "owner": "user-1001"},
    )
    assert policy.decide_context(context).action == PolicyAction.ALLOW

    wrong_owner = PolicyContext(
        tenant_id=context.tenant_id,
        principal_id=context.principal_id,
        role=context.role,
        scene=context.scene,
        tool=context.tool,
        args={"city": "合肥", "owner": "another-user"},
    )
    assert policy.decide_context(wrong_owner).action == PolicyAction.DENY


def test_rule_enforces_approval_and_process_local_rate_limit(tmp_path):
    config_path = _write_policy(
        tmp_path,
        """
version: rate-policy-v1
default:
  action: deny
rules:
  - id: limited-report-data
    priority: 100
    match:
      tenants: [tenant-a]
      roles: [user]
      scenes: [report]
      tools: [fetch_external_data]
      data_scopes: [usage_record:read]
      risk_levels: [medium]
    rate_limit:
      max_calls: 1
      window_seconds: 60
      key_by: [tenant_id, principal_id, tool]
    action: allow
    requires_approval: true
    reason: "tool requires approval: fetch_external_data"
    reason_code: approval_required
""",
    )
    fixed_time = datetime(2026, 1, 5, 10, 0, tzinfo=timezone.utc)
    policy = ToolPolicy(
        build_default_tool_registry(["fetch_external_data"]),
        config_path=config_path,
        clock=lambda: fixed_time,
    )
    kwargs = {
        "tenant_id": "tenant-a",
        "principal_id": "user-1001",
        "user_role": "user",
        "scene": "report",
        "tool_name": "fetch_external_data",
        "args": {"user_id": "u-1", "month": "2026-01"},
    }

    first = policy.decide(**kwargs)
    second = policy.decide(**kwargs)

    assert first.action == PolicyAction.NEED_APPROVAL
    assert first.reason_code == "approval_required"
    assert second.action == PolicyAction.DENY
    assert second.reason_code == "rate_limit_exceeded"
    assert second.matched_rule_id == "limited-report-data"


def test_policy_audit_never_exposes_sensitive_argument_values():
    events = []
    policy = ToolPolicy(
        build_default_tool_registry(["get_weather"]),
        audit_sink=events.append,
    )

    decision = policy.decide(
        tenant_id="tenant-a",
        principal_id="user-1001",
        user_role="user",
        scene="qa",
        tool_name="get_weather",
        args={
            "city": "合肥",
            "api_key": "top-secret-key",
            "nested": {"access_token": "nested-secret-token"},
        },
    )

    assert decision.action == PolicyAction.NEED_REDACTION
    assert decision.redacted_args["api_key"] == "<redacted>"
    assert decision.redacted_args["nested"]["access_token"] == "<redacted>"
    assert set(decision.redactions) == {"api_key", "nested.access_token"}
    assert "top-secret-key" not in repr(events[0])
    assert "nested-secret-token" not in repr(events[0])


def test_policy_rejects_fail_open_default(tmp_path):
    config_path = _write_policy(
        tmp_path,
        """
version: unsafe-v1
default:
  action: allow
rules: []
""",
    )

    with pytest.raises(ValueError, match="default action must be deny"):
        ToolPolicy(
            build_default_tool_registry(["get_weather"]),
            config_path=config_path,
        )
