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
from observability.tracing import trace_recorder


tool_registry = build_default_tool_registry(agent_conf.get("allowed_tools", []))


@wrap_tool_call
def monitor_tool(
        # 请求的数据封装
        request: ToolCallRequest,
        # 执行的函数本身
        handler: Callable[[ToolCallRequest], ToolMessage | Command],
) -> ToolMessage | Command:             # 工具执行的监控
    logger.info(f"[tool monitor]执行工具：{request.tool_call['name']}")
    logger.info(f"[tool monitor]传入参数：{request.tool_call['args']}")

    try:
        tool_name = request.tool_call["name"]
        tool_registry.require_allowed(tool_name)
        request_id = request.runtime.context.get("request_id")
        if request_id:
            with trace_recorder.span(
                    request_id,
                    category="tool",
                    name=tool_name,
                    metadata={"args": request.tool_call["args"]},
            ):
                result = handler(request)
        else:
            result = handler(request)
        logger.info(f"[tool monitor]工具{request.tool_call['name']}调用成功")

        if request.tool_call['name'] == "fill_context_for_report":
            request.runtime.context["report"] = True

        return result
    except Exception as e:
        logger.error(f"工具{request.tool_call['name']}调用失败，原因：{str(e)}")
        raise e


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
