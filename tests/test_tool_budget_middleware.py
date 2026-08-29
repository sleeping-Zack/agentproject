from types import SimpleNamespace

import agent.tools.middleware as middleware_module
from agent.budget import BudgetManager
from agent.tools.retry import RetryPolicy
from langchain_core.messages import ToolMessage

from agent.tools.middleware import (
    _approval_arguments_match,
    _authorize_rag_call,
    _enforce_tool_budget,
    _enforce_tool_policy,
    _is_tool_cacheable,
    _record_rag_outcome,
    _run_with_timeout,
    _tool_result_event_payload,
    monitor_tool,
)
from observability.context import bind_request_context, request_context
from observability.tracing import trace_recorder
from services.approval_store import SQLiteApprovalStore


def test_tool_budget_blocks_before_handler_invocation():
    runtime_context = {"max_tool_calls": 1, "used_tool_calls": 1}

    result = _enforce_tool_budget(
        runtime_context=runtime_context,
        tool_name="rag_summarize",
        tool_call_id="call-1",
    )

    assert result is not None
    assert "工具调用预算已耗尽" in result.content
    assert runtime_context["used_tool_calls"] == 1


def test_tool_budget_increments_before_allowed_call():
    runtime_context = {"max_tool_calls": 2, "used_tool_calls": 1}

    result = _enforce_tool_budget(
        runtime_context=runtime_context,
        tool_name="rag_summarize",
        tool_call_id="call-1",
    )

    assert result is None
    assert runtime_context["used_tool_calls"] == 2


def test_monitor_tool_retries_consume_one_logical_tool_call(monkeypatch):
    monkeypatch.setattr(
        middleware_module,
        "default_retry_policy",
        RetryPolicy(max_attempts=3, base_delay=0, max_delay=0, jitter=0),
    )
    manager = BudgetManager(max_tool_calls=3)
    request = SimpleNamespace(
        tool_call={
            "id": "retry-budget-call",
            "name": "get_weather",
            "args": {"city": "retry-budget-test-city"},
        },
        runtime=SimpleNamespace(
            context={
                "budget_manager": manager,
                "tenant_id": "default",
                "user_role": "user",
                "scene": "default",
            }
        ),
    )
    attempts = 0

    def handler(_request):
        nonlocal attempts
        attempts += 1
        if attempts < 3:
            raise RuntimeError("temporary transport failure")
        return ToolMessage(
            content="success",
            tool_call_id="retry-budget-call",
            name="get_weather",
        )

    result = monitor_tool.wrap_tool_call(request, handler)

    assert result.content == "success"
    assert attempts == 3
    assert manager.snapshot()["used_tool_calls"] == 1


def test_rag_guard_allows_distinct_gap_driven_retrievals():
    runtime_context = {"max_rag_calls": 3, "final_response_token_reserve": 3000}
    manager = BudgetManager(max_tokens=8000)

    first = _authorize_rag_call(
        runtime_context,
        "rag_summarize",
        {
            "query": "复杂家庭如何选择扫地机器人",
            "information_gap": "需要知识库提供复杂家庭的选购证据",
        },
        "call-1",
        manager,
    )
    second = _authorize_rag_call(
        runtime_context,
        "rag_summarize",
        {
            "query": "宠物毛发清洁能力",
            "information_gap": "还缺少宠物家庭的毛发处理能力",
        },
        "call-2",
        manager,
    )

    assert first is None
    assert second is None
    assert len(runtime_context["rag_query_history"]) == 2


def test_rag_guard_blocks_paraphrased_duplicate_query():
    runtime_context = {"max_rag_calls": 3, "final_response_token_reserve": 3000}

    assert _authorize_rag_call(
        runtime_context,
        "rag_summarize",
        {
            "query": "扫地机器人的优缺点",
            "information_gap": "需要知识库提供优缺点证据",
        },
        "call-1",
    ) is None
    blocked = _authorize_rag_call(
        runtime_context,
        "rag_summarize",
        {
            "query": "智能扫地机器人的好处、不足和清洁效果",
            "information_gap": "补充优点和缺点",
        },
        "call-2",
    )

    assert blocked is not None
    assert "语义重复" in blocked.content
    assert runtime_context["rag_loop_closed"] is True


def test_rag_guard_requires_information_gap_after_first_call():
    runtime_context = {}

    assert _authorize_rag_call(
        runtime_context,
        "rag_summarize",
        {
            "query": "扫地机器人的选购方法",
            "information_gap": "需要知识库提供选购依据",
        },
        "call-1",
    ) is None
    blocked = _authorize_rag_call(
        runtime_context,
        "rag_summarize",
        {"query": "宠物家庭选购方法"},
        "call-2",
    )

    assert blocked is not None
    assert "information_gap" in blocked.content


def test_rag_guard_requires_information_gap_on_first_call():
    blocked = _authorize_rag_call(
        {},
        "rag_summarize",
        {"query": "滤网维护"},
        "call-1",
    )

    assert blocked is not None
    assert "information_gap" in blocked.content


def test_rag_guard_preserves_final_answer_budget():
    runtime_context = {"final_response_token_reserve": 3000}
    manager = BudgetManager(max_tokens=8000, used_tokens=5100)

    blocked = _authorize_rag_call(
        runtime_context,
        "rag_summarize",
        {
            "query": "滤网维护",
            "information_gap": "需要知识库提供维护周期",
        },
        "call-1",
        manager,
    )

    assert blocked is not None
    assert "最终回答预算" in blocked.content


def test_rag_guard_does_not_limit_other_tools():
    runtime_context = {}

    assert _authorize_rag_call(
        runtime_context, "get_weather", {"city": "上海"}, "call-1"
    ) is None
    assert _authorize_rag_call(
        runtime_context, "get_weather", {"city": "北京"}, "call-2"
    ) is None


def test_only_read_tools_use_the_generic_tool_cache():
    assert _is_tool_cacheable("get_weather") is True
    assert _is_tool_cacheable("get_product_specs") is True
    assert _is_tool_cacheable("create_support_ticket") is False
    assert _is_tool_cacheable("fill_context_for_report") is False


def test_ticket_approval_treats_missing_optional_error_code_as_empty():
    approved = {
        "model": "s20",
        "issue_type": "repair",
        "description": "  水箱故障  ",
    }
    resumed = {
        "model": "S20",
        "issue_type": "repair",
        "description": "水箱故障",
        "error_code": "",
    }

    assert _approval_arguments_match(
        "create_support_ticket", approved, resumed
    ) is True


def test_ticket_policy_records_pending_approval_in_tool_trace(tmp_path, monkeypatch):
    monkeypatch.setattr(
        middleware_module,
        "approval_store",
        SQLiteApprovalStore(str(tmp_path / "approvals.db")),
    )
    request_id = "ticket-policy-pending-trace"
    trace_recorder.start_trace(request_id, "session-ticket")

    result = _enforce_tool_policy(
        tool_name="create_support_ticket",
        tool_args={
            "model": "S20",
            "issue_type": "repair",
            "description": "水箱故障，需要创建售后工单",
        },
        tool_call_id="tool-call-ticket",
        request_id=request_id,
        tenant_id="tenant-a",
        principal_id="user-a",
        data_user_id=None,
        user_role="user",
        scene="default",
        approval_id=None,
    )

    assert result is not None
    assert "pending_approval" in result.content
    events = trace_recorder.export_trace(request_id)["events"]
    pending = next(event for event in events if event.get("category") == "tool")
    assert pending["name"] == "create_support_ticket"
    assert pending["metadata"]["status"] == "pending_approval"
    assert pending["metadata"]["approval_id"]


def test_rag_guard_stops_after_empty_business_result():
    runtime_context = {}
    _record_rag_outcome(
        runtime_context,
        {"business_status": "empty", "evidence": []},
    )

    blocked = _authorize_rag_call(
        runtime_context,
        "rag_summarize",
        {
            "query": "换一个关键词继续查",
            "information_gap": "尝试用另一种说法查找相同问题",
        },
        "call-2",
    )

    assert blocked is not None
    assert "没有检索到" in blocked.content
    assert runtime_context["rag_loop_closed_reason"] == "empty"


def test_rag_guard_stops_after_relevant_evidence_is_available():
    runtime_context = {}
    _record_rag_outcome(
        runtime_context,
        {
            "business_status": "verification_failed",
            "evidence": [{"id": "manual-1", "content": "先断电再检查主刷。"}],
        },
    )

    blocked = _authorize_rag_call(
        runtime_context,
        "rag_summarize",
        {
            "query": "主刷异响其他原因",
            "information_gap": "继续更换关键词寻找更多资料",
        },
        "call-2",
    )

    assert blocked is not None
    assert "已有足够" in blocked.content
    assert runtime_context["rag_loop_closed_reason"] == "evidence_available"


def test_tool_result_event_payload_is_redacted_and_bounded():
    result = ToolMessage(
        content="token=private " + ("结果" * 3000),
        tool_call_id="call-1",
        name="rag_summarize",
    )

    payload = _tool_result_event_payload(result)

    assert "private" not in payload["result"]
    assert "<redacted>" in payload["result"]
    assert payload["result_truncated"] is True
    assert len(payload["result"]) <= 4001


def test_tool_timeout_worker_preserves_request_context():
    with bind_request_context(request_id="req-rag-context", tenant_id="tenant-a"):
        observed = _run_with_timeout(
            lambda: (
                request_context().request_id,
                request_context().tenant_id,
            ),
            timeout_seconds=1,
        )

    assert observed == ("req-rag-context", "tenant-a")
