import sys
from types import ModuleType

import pytest

from agent.budget import BudgetManager, bind_budget_manager
from agent.budgeted_text_model import estimate_text_tokens, invoke_budgeted_text_model
from observability.context import bind_request_context
from observability.tracing import trace_recorder


class _Response:
    def __init__(self, content="完成", usage=None):
        self.content = content
        self.usage_metadata = usage or {}
        self.response_metadata = {}


class _Model:
    def __init__(self, response=None, error=None):
        self.response = response or _Response()
        self.error = error
        self.settings = None

    def bind(self, **settings):
        self.settings = settings
        return self

    def invoke(self, _prompt):
        if self.error:
            raise self.error
        return self.response


def test_budgeted_text_model_commits_usage_and_records_trace():
    model = _Model(_Response(usage={"input_tokens": 7, "output_tokens": 3}))
    budget = BudgetManager(max_tokens=100, max_cost=1)
    request_id = "budgeted-text-usage"
    trace_recorder.start_trace(request_id, "session-1")

    with bind_request_context(request_id=request_id), bind_budget_manager(budget):
        invoke_budgeted_text_model(
            model, "请生成摘要", max_output_tokens=20, operation="conversation-summary"
        )

    assert model.settings["max_tokens"] == 20
    assert budget.used_tokens == 10
    events = trace_recorder.export_trace(request_id)["events"]
    usage = next(event for event in events if event["metadata"].get("type") == "model_usage")
    assert usage["metadata"]["tokens_in"] == 7
    assert usage["metadata"]["tokens_out"] == 3


def test_budgeted_text_model_error_charges_only_prompt_tokens():
    prompt = "模型异常时仍应只计入已发送的提示词"
    budget = BudgetManager(max_tokens=100, max_cost=1)

    with bind_budget_manager(budget), pytest.raises(RuntimeError, match="provider"):
        invoke_budgeted_text_model(
            _Model(error=RuntimeError("provider")),
            prompt,
            max_output_tokens=20,
            operation="memory-extraction",
        )

    assert budget.used_tokens == estimate_text_tokens(prompt)


def test_summary_uses_small_default_output_cap(monkeypatch):
    from agent.summarizer import ConversationSummarizer

    summary_model = _Model()

    class _SummaryChatModel:
        def resolve(self):
            return summary_model

    factory = ModuleType("model.factory")
    factory.chat_model = _SummaryChatModel()
    monkeypatch.setitem(sys.modules, "model.factory", factory)
    ConversationSummarizer()._default_invoker("摘要提示词")
    assert summary_model.settings["max_tokens"] == 500
