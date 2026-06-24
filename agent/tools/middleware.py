from __future__ import annotations

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
from agent.tools.registry import build_default_tool_registry
from agent.tools.retry import RetryPolicy, run_with_retry
from observability.event_bus import event_bus
from observability.tracing import trace_recorder
from observability.metrics import metrics_registry
from services.cache import tool_call_cache
from services.circuit_breaker import CircuitOpenError, breaker_registry


tool_registry = build_default_tool_registry(agent_conf.get("allowed_tools", []))
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
    logger.info(f"[tool monitor]执行工具：{tool_name}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

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
                    metadata={"args": request.tool_call["args"]},
            ):
                return handler(request)
        return handler(request)

    def _on_retry(attempt: int, exc: BaseException, wait: float) -> None:
        metrics_registry.inc_tool_call(tool_name, status="retry")
        logger.warning(
            f"[tool monitor]工具{tool_name}第{attempt}次调用失败：{exc}，{wait:.2f}s 后重试"
        )

    start = metrics_registry.now()
    if request_id:
        event_bus.publish(request_id, "tool_start",
                          {"tool": tool_name, "args": request.tool_call.get("args", {})})

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
        result = run_with_retry(_invoke, policy=default_retry_policy, on_retry=_on_retry)
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
