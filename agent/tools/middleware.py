from __future__ import annotations

import os
from concurrent.futures import ThreadPoolExecutor, TimeoutError
from typing import Callable
from utils.prompt_loader import load_system_prompts, load_report_prompts
from langchain.agents import AgentState
from langchain.agents.middleware import wrap_tool_call, before_model, dynamic_prompt, ModelRequest
from langchain.tools.tool_node import ToolCallRequest
from langchain_core.messages import ToolMessage
from langgraph.runtime import Runtime
from langgraph.types import Command
from utils.logger_handler import logger
from utils.config_handler import agent_conf
from agent.policies import PolicyAction, ToolPolicy
from agent.tools.registry import build_default_tool_registry
from agent.tools.retry import RetryPolicy, run_with_retry
from observability.event_bus import event_bus
from observability.tracing import trace_recorder
from observability.metrics import metrics_registry
from services.cache import tool_call_cache
from services.circuit_breaker import CircuitOpenError, breaker_registry
from services.approval_store import SQLiteApprovalStore
from safety.security import args_hash, redact_sensitive, sensitive_tool_approval


tool_registry = build_default_tool_registry(agent_conf.get("allowed_tools", []))
tool_policy = ToolPolicy(tool_registry=tool_registry)
approval_store = SQLiteApprovalStore(
    os.getenv("AGENT_APPROVAL_DB_PATH", "storage/approvals.db")
)
default_retry_policy = RetryPolicy(
    max_attempts=int(agent_conf.get("tool_retry_max_attempts", 3)),
    base_delay=float(agent_conf.get("tool_retry_base_delay", 0.2)),
)
TOOL_BREAKER_FAILURE_THRESHOLD = int(agent_conf.get("tool_breaker_failure_threshold", 5))
TOOL_BREAKER_RECOVERY_TIMEOUT = float(agent_conf.get("tool_breaker_recovery_timeout", 30.0))


@wrap_tool_call
def monitor_tool(
        request: ToolCallRequest,
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    tool_name = request.tool_call["name"]
    tool_args = request.tool_call.get("args", {})
    redacted_args = redact_sensitive(tool_args)
    logger.info(f"[tool monitor]执行工具：{tool_name}")
    logger.info(f"[tool monitor]传入参数：{redacted_args}")

    try:
        tool_registry.require_allowed(tool_name)
    except PermissionError as exc:
        metrics_registry.inc_tool_call(tool_name, status="denied")
        logger.error(f"工具{tool_name}被拒绝：{exc}")
        return ToolMessage(
            content=f"工具调用被拒绝：{exc}",
            tool_call_id=request.tool_call.get("id", ""),
            name=tool_name,
        )

    request_id = request.runtime.context.get("request_id")
    tenant_id = request.runtime.context.get("tenant_id", "default")
    user_role = request.runtime.context.get("user_role", "user")
    scene = request.runtime.context.get("scene", "default")
    approval_id = request.runtime.context.get("approval_id")

    budget_result = _enforce_tool_budget(
        runtime_context=request.runtime.context,
        tool_name=tool_name,
        tool_call_id=request.tool_call.get("id", ""),
    )
    if budget_result is not None:
        return budget_result

    policy_result = _enforce_tool_policy(
        tool_name=tool_name,
        tool_args=tool_args,
        tool_call_id=request.tool_call.get("id", ""),
        request_id=request_id,
        tenant_id=tenant_id,
        user_role=user_role,
        scene=scene,
        approval_id=approval_id,
    )
    if policy_result is not None:
        return policy_result

    breaker = breaker_registry.get(
        f"tool:{tool_name}",
        failure_threshold=TOOL_BREAKER_FAILURE_THRESHOLD,
        recovery_timeout=TOOL_BREAKER_RECOVERY_TIMEOUT,
    )
    if not breaker.allow():
        metrics_registry.inc_tool_call(tool_name, status="short_circuit")
        logger.warning(f"[tool monitor]工具{tool_name}熔断打开，跳过实际调用")
        return ToolMessage(
            content=f"工具 {tool_name} 当前不可用（熔断保护中），请稍后重试或换一种方式回答用户。",
            tool_call_id=request.tool_call.get("id", ""),
            name=tool_name,
        )

    def _invoke():
        if request_id:
            with trace_recorder.span(
                    request_id,
                    category="tool",
                    name=tool_name,
                    metadata={
                        "args_hash": args_hash(tool_args),
                        "redacted_args": redacted_args,
                    },
            ):
                return _invoke_handler_with_approval(tool_name, request, handler)
        return _invoke_handler_with_approval(tool_name, request, handler)

    def _on_retry(attempt: int, exc: BaseException, wait: float) -> None:
        metrics_registry.inc_tool_call(tool_name, status="retry")
        logger.warning(
            f"[tool monitor]工具{tool_name}第{attempt}次调用失败：{exc}，{wait:.2f}s 后重试"
        )

    start = metrics_registry.now()
    if request_id:
        event_bus.publish(request_id, "tool_start",
                          {"tool": tool_name, "args": redacted_args})

    # 幂等缓存：相同 tool+args 在 TTL 窗口内复用上次结果，跳过实际调用
    cache_args = request.tool_call.get("args", {})
    cached = tool_call_cache.get(tool_name, cache_args)
    if cached is not None:
        metrics_registry.inc_tool_call(tool_name, status="cache_hit")
        if request_id:
            event_bus.publish(request_id, "tool_end",
                              {"tool": tool_name, "status": "cache_hit",
                               "duration_ms": 0.0})
        return cached

    try:
        result = _run_with_timeout(
            lambda: run_with_retry(_invoke, policy=default_retry_policy, on_retry=_on_retry),
            timeout_seconds=_timeout_seconds(tool_name),
        )
        breaker.record_success()
        metrics_registry.inc_tool_call(tool_name, status="success")
        elapsed = metrics_registry.elapsed_ms(start)
        metrics_registry.observe_tool_latency(tool_name, elapsed)
        logger.info(f"[tool monitor]工具{tool_name}调用成功")
        if request_id:
            event_bus.publish(request_id, "tool_end",
                              {"tool": tool_name, "status": "success",
                               "duration_ms": round(elapsed, 2)})
        # 只缓存成功的 ToolMessage；Command 类型有副作用不缓存
        if isinstance(result, ToolMessage):
            tool_call_cache.set(tool_name, cache_args, result)

        if tool_name == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except CircuitOpenError as exc:
        metrics_registry.inc_tool_call(tool_name, status="short_circuit")
        logger.warning(f"[tool monitor]工具{tool_name}熔断短路：{exc}")
        if request_id:
            event_bus.publish(request_id, "tool_end",
                              {"tool": tool_name, "status": "short_circuit"})
        return ToolMessage(
            content=f"工具 {tool_name} 当前不可用（熔断保护中）。",
            tool_call_id=request.tool_call.get("id", ""),
            name=tool_name,
        )
    except TimeoutError:
        breaker.record_failure()
        metrics_registry.inc_tool_call(tool_name, status="timeout")
        elapsed = metrics_registry.elapsed_ms(start)
        metrics_registry.observe_tool_latency(tool_name, elapsed)
        if request_id:
            event_bus.publish(request_id, "tool_end",
                              {"tool": tool_name, "status": "timeout",
                               "duration_ms": round(elapsed, 2)})
            trace_recorder.record_diagnostic_event(
                request_id=request_id,
                step_id="tool-timeout",
                event_type="tool_call",
                status="failed",
                latency_ms=elapsed,
                tool=tool_name,
                args_hash=args_hash(tool_args),
                failure_reason="tool_timeout",
            )
        return ToolMessage(
            content=f"工具调用超时：{tool_name}。请向用户说明暂时无法获取该数据。",
            tool_call_id=request.tool_call.get("id", ""),
            name=tool_name,
        )
    except Exception as exc:
        breaker.record_failure()
        metrics_registry.inc_tool_call(tool_name, status="failure")
        metrics_registry.observe_tool_latency(tool_name, metrics_registry.elapsed_ms(start))
        logger.error(f"工具{tool_name}调用失败，原因：{exc}")
        if request_id:
            event_bus.publish(request_id, "tool_end",
                              {"tool": tool_name, "status": "failure",
                               "error": str(exc)})
        return ToolMessage(
            content=f"工具调用失败：{exc}。请向用户说明无法获取该数据或换一种方式。",
            tool_call_id=request.tool_call.get("id", ""),
        name=tool_name,
    )


def _enforce_tool_budget(
    runtime_context: dict,
    tool_name: str,
    tool_call_id: str,
) -> ToolMessage | None:
    max_tool_calls = runtime_context.get("max_tool_calls")
    if max_tool_calls is None:
        return None
    used_tool_calls = int(runtime_context.get("used_tool_calls", 0))
    if used_tool_calls >= int(max_tool_calls):
        metrics_registry.inc_tool_call(tool_name, status="budget_exceeded")
        return ToolMessage(
            content="工具调用预算已耗尽，已拒绝继续调用工具。",
            tool_call_id=tool_call_id,
            name=tool_name,
        )
    runtime_context["used_tool_calls"] = used_tool_calls + 1
    return None


def _enforce_tool_policy(
    tool_name: str,
    tool_args: dict,
    tool_call_id: str,
    request_id: str | None,
    tenant_id: str,
    user_role: str,
    scene: str,
    approval_id: str | None,
) -> ToolMessage | None:
    decision = tool_policy.decide(
        tenant_id=tenant_id,
        user_role=user_role,
        scene=scene,
        tool_name=tool_name,
        args=tool_args,
    )
    if decision.action == PolicyAction.ALLOW:
        return None
    if decision.action == PolicyAction.DENY:
        metrics_registry.inc_tool_call(tool_name, status="denied")
        return ToolMessage(
            content=f"工具调用被拒绝：{decision.reason}",
            tool_call_id=tool_call_id,
            name=tool_name,
        )
    if decision.action == PolicyAction.NEED_REDACTION:
        metrics_registry.inc_tool_call(tool_name, status="denied")
        return ToolMessage(
            content="工具参数包含敏感字段，已拒绝执行。",
            tool_call_id=tool_call_id,
            name=tool_name,
        )
    if approval_id:
        try:
            approval = approval_store.get(approval_id)
        except KeyError:
            return ToolMessage(
                content="敏感工具审批记录不存在，已拒绝执行。",
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        if approval.tenant_id != tenant_id or approval.tool_name != tool_name:
            return ToolMessage(
                content="敏感工具审批记录与当前租户或工具不匹配，已拒绝执行。",
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        if approval.is_approved:
            return None
        if approval.is_denied:
            return ToolMessage(
                content="敏感工具调用审批已被拒绝。",
                tool_call_id=tool_call_id,
                name=tool_name,
            )
        return ToolMessage(
            content=f"pending_approval approval_id={approval.approval_id}",
            tool_call_id=tool_call_id,
            name=tool_name,
        )

    if request_id:
        approval = approval_store.create_pending(
            request_id=request_id,
            tenant_id=tenant_id,
            user_role=user_role,
            tool_name=tool_name,
            args=tool_args,
            reason=decision.reason,
        )
        event_bus.publish(
            request_id,
            "approval_required",
            {"tool": tool_name, "approval_id": approval.approval_id},
        )
        return ToolMessage(
            content=f"pending_approval approval_id={approval.approval_id}",
            tool_call_id=tool_call_id,
            name=tool_name,
        )
    return ToolMessage(
        content=f"工具需要审批：{decision.reason}",
        tool_call_id=tool_call_id,
        name=tool_name,
    )


def _invoke_handler_with_approval(
    tool_name: str,
    request: ToolCallRequest,
    handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:
    if tool_name == "fetch_external_data":
        with sensitive_tool_approval(tool_name):
            return handler(request)
    return handler(request)


def _timeout_seconds(tool_name: str) -> int:
    spec = tool_registry.get_spec(tool_name)
    return spec.timeout_seconds if spec is not None else 30


def _run_with_timeout(call: Callable[[], ToolMessage | Command], timeout_seconds: int):
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(call)
    try:
        return future.result(timeout=timeout_seconds)
    except TimeoutError:
        future.cancel()
        raise
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


@before_model
def log_before_model(
        state: AgentState,          # 整个Agent智能体中的状态记录
        runtime: Runtime,           # 记录了整个执行过程中的上下文信息
):         # 在模型执行前输出日志
    logger.info(f"[log_before_model]即将调用模型，带有{len(state['messages'])}条消息。")
    request_id = runtime.context.get("request_id")
    if request_id:
        with trace_recorder.span(
                request_id,
                category="model",
                name="before_model",
                metadata={"message_count": len(state["messages"])},
        ):
            logger.debug(
                f"[log_before_model]{type(state['messages'][-1]).__name__} | "
                f"{state['messages'][-1].content.strip()}"
            )
    else:
        logger.debug(f"[log_before_model]{type(state['messages'][-1]).__name__} | {state['messages'][-1].content.strip()}")

    return None


@dynamic_prompt                 # 每一次在生成提示词之前，调用此函数
def report_prompt_switch(request: ModelRequest):     # 动态切换提示词
    is_report = request.runtime.context.get("report", False)
    if is_report:               # 是报告生成场景，返回报告生成提示词内容
        return load_report_prompts()

    return load_system_prompts()
