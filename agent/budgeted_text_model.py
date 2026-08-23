"""Budget-aware helpers for small, non-agent text generation calls."""
from __future__ import annotations

import os
from typing import Any, Callable

from agent.budget import BudgetManager, current_budget_manager
from observability.context import request_context
from observability.tracing import trace_recorder


def estimate_text_tokens(text: str) -> int:
    """Use the same CJK-aware approximation as the agent middleware."""

    text = str(text or "")
    non_ascii = sum(1 for character in text if ord(character) > 127)
    return max(1, non_ascii + ((len(text) - non_ascii + 3) // 4))


def invoke_budgeted_text_model(
    model: Any,
    prompt: str,
    *,
    max_output_tokens: int,
    operation: str,
    temperature: float | None = None,
) -> Any:
    """Invoke a text model while charging the request's shared budget."""

    def invoke(output_cap: int) -> Any:
        settings = {"max_tokens": output_cap}
        if temperature is not None:
            settings["temperature"] = temperature
        configured = model.bind(**settings) if hasattr(model, "bind") else model
        return configured.invoke(prompt)

    return invoke_budgeted_call(
        invoke,
        prompt,
        max_output_tokens=max_output_tokens,
        operation=operation,
        model_name=type(model).__name__,
    )


def invoke_budgeted_call(
    invoke: Callable[[int], Any],
    prompt: str,
    *,
    max_output_tokens: int,
    operation: str,
    budget_manager: BudgetManager | None = None,
    model_name: str = "",
    retry: int = 0,
) -> Any:
    """Budget any direct model call whose invoker accepts an output cap."""

    max_output_tokens = max(1, int(max_output_tokens))
    estimated_input = estimate_text_tokens(prompt)
    manager = budget_manager or current_budget_manager()
    cost_per_1k = max(
        0.0,
        float(os.getenv("AGENT_ESTIMATED_COST_PER_1K_TOKENS", "0.001")),
    )
    reservation = None
    output_cap = max_output_tokens
    if manager is not None:
        output_cap = min(output_cap, manager.remaining_output_tokens(estimated_input))
        if cost_per_1k > 0:
            affordable_total = int((manager.remaining_cost * 1000) / cost_per_1k)
            output_cap = min(output_cap, max(0, affordable_total - estimated_input))
        estimated_cost = round(
            ((estimated_input + output_cap) / 1000) * cost_per_1k,
            6,
        )
        reservation = manager.reserve_model_call(
            estimated_input_tokens=estimated_input,
            max_output_tokens=output_cap,
            estimated_cost=estimated_cost,
        )
        output_cap = reservation.max_output_tokens

    try:
        response = invoke(output_cap)
    except BaseException:
        if reservation is not None:
            manager.commit_model_call(
                reservation,
                actual_tokens=estimated_input,
                actual_cost=round((estimated_input / 1000) * cost_per_1k, 6),
            )
        raise

    tokens_in, tokens_out, actual_cost, has_usage = _response_usage(response)
    if not tokens_in:
        tokens_in = estimated_input
    if not tokens_out:
        tokens_out = estimate_text_tokens(model_response_text(response))
    total_tokens = tokens_in + tokens_out
    cost_mode = "actual" if has_usage else "estimated"
    if actual_cost is None:
        actual_cost = round((total_tokens / 1000) * cost_per_1k, 6)
    if reservation is not None:
        manager.commit_model_call(
            reservation,
            actual_tokens=total_tokens,
            actual_cost=actual_cost,
        )
    _record_model_usage(
        operation=operation,
        tokens_in=tokens_in,
        tokens_out=tokens_out,
        cost=actual_cost,
        cost_mode=cost_mode,
        model_name=model_name,
        retry=retry,
    )
    return response


def model_response_text(response: Any) -> str:
    content = getattr(response, "content", response)
    if isinstance(content, list):
        return "".join(
            str(item.get("text") or item.get("content") or "")
            if isinstance(item, dict)
            else str(item)
            for item in content
        )
    return str(content or "")


def _response_usage(response: Any) -> tuple[int, int, float | None, bool]:
    usage = getattr(response, "usage_metadata", None) or {}
    metadata = getattr(response, "response_metadata", None) or {}
    token_usage = metadata.get("token_usage") or metadata.get("usage") or {}
    tokens_in = int(
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or token_usage.get("input_tokens")
        or token_usage.get("prompt_tokens")
        or 0
    )
    tokens_out = int(
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or token_usage.get("output_tokens")
        or token_usage.get("completion_tokens")
        or 0
    )
    cost = metadata.get("cost") or token_usage.get("cost")
    return (
        tokens_in,
        tokens_out,
        float(cost) if cost is not None else None,
        bool(tokens_in or tokens_out),
    )


def _record_model_usage(
    *,
    operation: str,
    tokens_in: int,
    tokens_out: int,
    cost: float,
    cost_mode: str,
    model_name: str,
    retry: int,
) -> None:
    ctx = request_context()
    if not ctx.request_id:
        return
    try:
        trace_recorder.record_diagnostic_event(
            request_id=ctx.request_id,
            step_id=operation,
            event_type="model_usage",
            status="ok",
            latency_ms=0.0,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            cost_mode=cost_mode,
            retry=retry,
            model_name=model_name,
        )
    except KeyError:
        return
