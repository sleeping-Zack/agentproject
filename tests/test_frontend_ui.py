import inspect

from app import (
    NAVIGATION,
    QUICK_PROMPTS,
    _audit_event,
    _event_detail,
    _event_label,
    _inject_theme,
    _merge_answer,
)


def test_merge_answer_replaces_provisional_content_for_final_answer():
    assert _merge_answer("旧的中间内容", {"delta": "最终回答", "replace": True}) == "最终回答"


def test_merge_answer_appends_regular_delta():
    assert _merge_answer("您好", {"delta": "，请问有什么可以帮助您？"}) == (
        "您好，请问有什么可以帮助您？"
    )


def test_primary_navigation_keeps_complex_tasks_inside_chat():
    assert list(NAVIGATION) == ["对话", "记忆", "审批", "诊断"]
    assert "planner" not in NAVIGATION.values()
    assert len(QUICK_PROMPTS) == 4
    assert all("租户" not in label and "ID" not in label for label, _ in QUICK_PROMPTS)


def test_event_label_translates_backend_events_for_customer_view():
    assert _event_label("tool_started", {"tool": "rag_summarize"}) == (
        "正在调用企业知识库"
    )
    assert _event_label("verification_started", {}) == "正在校验答案的完整性与可靠性"


def test_auto_routing_and_plan_events_are_visible_in_the_chat_audit():
    routing = _audit_event(
        {
            "id": "2",
            "event": "routing_completed",
            "data": {
                "execution_mode": "plan_execute",
                "complexity_score": 7,
                "reasons": ["cross_domain_request"],
            },
        }
    )
    plan_step = _audit_event(
        {
            "id": "4",
            "event": "plan_step_completed",
            "data": {
                "id": "t1",
                "kind": "report",
                "description": "读取使用报告",
                "status": "completed",
                "result": "报告读取完成",
            },
        }
    )

    assert routing["label"] == "已自动选择处理方式：先规划再分步执行"
    assert routing["detail"]["处理方式"] == "先规划再分步执行"
    assert routing["detail"]["判定依据"] == ["涉及多个信息域"]
    assert plan_step["label"] == "计划步骤完成：读取使用报告"
    assert plan_step["detail"]["步骤结果"] == "报告读取完成"


def test_runtime_routing_transition_is_visible_without_exposing_reasoning_text():
    event = _audit_event(
        {
            "id": "5",
            "event": "routing_transition",
            "data": {
                "from_mode": "react",
                "to_mode": "plan_execute",
                "decision_source": "runtime_feedback",
                "reasons": ["verification_retry_escalation"],
            },
        }
    )

    assert event["label"] == (
        "处理方式已动态调整：边分析边调用所需服务 → 先规划再分步执行"
    )
    assert event["detail"]["调整依据"] == [
        "上轮结果未通过校验，升级为规划执行"
    ]


def test_budget_degradation_is_visible_as_a_verified_partial_result():
    event = _audit_event(
        {
            "id": "6",
            "event": "execution_degraded",
            "data": {
                "status": "completed",
                "reason": "retry_token_budget_insufficient",
                "strategy": "verified_partial_result",
            },
        }
    )

    assert event["label"] == "预算不足，已返回通过校验的部分结果"
    assert event["detail"]["降级原因"] == "剩余 Token 不足以执行安全重试"
    assert event["detail"]["降级策略"] == "verified_partial_result"


def test_audit_event_keeps_structured_results_and_drops_stream_noise():
    event = _audit_event(
        {
            "id": "4",
            "event": "tool_completed",
            "data": {
                "tool": "rag_summarize",
                "status": "success",
                "duration_ms": 18.2,
                "result": "滤网应定期清理",
                "result_truncated": False,
            },
        }
    )

    assert event["id"] == "4"
    assert event["detail"]["工具"] == "企业知识库"
    assert event["detail"]["返回结果"] == "滤网应定期清理"
    assert _audit_event({"event": "token_delta", "data": {"delta": "x"}}) is None
    assert _audit_event({"event": "heartbeat", "data": {}}) is None


def test_verification_detail_exposes_quality_result():
    detail = _event_detail(
        "verification_completed",
        {
            "passed": False,
            "action": "retry",
            "score": 0.6,
            "reasons": ["citation_missing"],
            "citation_coverage": 0.5,
        },
    )

    assert detail["是否通过"] is False
    assert detail["未通过原因"] == ["citation_missing"]
    assert detail["引用覆盖率"] == 0.5


def test_sidebar_controls_remain_available_after_collapse():
    theme_source = inspect.getsource(_inject_theme)

    assert '[data-testid="stToolbar"] { display: flex !important; }' in theme_source
    assert '[data-testid="stSidebarCollapseButton"] {' in theme_source
    assert '[data-testid="stExpandSidebarButton"] {' in theme_source
    assert (
        '[data-testid="stToolbar"], #MainMenu, footer { display: none; }'
        not in theme_source
    )
