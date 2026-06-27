from agent.tools.middleware import _enforce_tool_budget


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
