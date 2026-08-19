from agent.budget import BudgetManager
from langchain_core.messages import ToolMessage

from agent.tools.middleware import (
    _authorize_rag_call,
    _enforce_tool_budget,
    _record_rag_outcome,
    _run_with_timeout,
    _tool_result_event_payload,
)
from observability.context import bind_request_context, request_context


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
