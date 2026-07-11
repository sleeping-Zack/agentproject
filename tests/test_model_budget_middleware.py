import pytest
from langchain.agents.middleware import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage
from langgraph.runtime import Runtime

from agent.budget import BudgetExceeded, BudgetManager
from agent.tools.middleware import enforce_model_budget


def _request(manager, *, max_output=100):
    return ModelRequest(
        model=object(),
        messages=[HumanMessage(content="12345678")],
        runtime=Runtime(
            context={
                "budget_manager": manager,
                "max_model_output_tokens": max_output,
                "estimated_cost_per_1k_tokens": 0.0,
            }
        ),
    )


def test_model_middleware_caps_output_before_call_and_commits_actual_usage():
    manager = BudgetManager(max_tokens=10)
    observed = {}

    def handler(request):
        observed.update(request.model_settings)
        return AIMessage(
            content="done",
            usage_metadata={"input_tokens": 2, "output_tokens": 3, "total_tokens": 5},
        )

    enforce_model_budget.wrap_model_call(_request(manager), handler)

    assert observed["max_tokens"] == 8
    snapshot = manager.snapshot()
    assert snapshot["used_tokens"] == 5
    assert snapshot["reserved_tokens"] == 0


def test_model_middleware_rejects_before_handler_when_prompt_uses_budget():
    manager = BudgetManager(max_tokens=2)
    invoked = False

    def handler(request):
        nonlocal invoked
        invoked = True
        return AIMessage(content="should not run")

    with pytest.raises(BudgetExceeded, match="max_tokens_exceeded"):
        enforce_model_budget.wrap_model_call(_request(manager), handler)

    assert not invoked


def test_model_middleware_cache_hit_does_not_charge_tokens():
    manager = BudgetManager(max_tokens=20)

    def handler(request):
        return AIMessage(
            content="cached",
            response_metadata={"cache_hit": True},
            usage_metadata={"input_tokens": 10, "output_tokens": 5, "total_tokens": 15},
        )

    enforce_model_budget.wrap_model_call(_request(manager), handler)

    snapshot = manager.snapshot()
    assert snapshot["used_tokens"] == 0
    assert snapshot["model_cache_hits"] == 1
