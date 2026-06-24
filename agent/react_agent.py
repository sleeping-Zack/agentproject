from __future__ import annotations

import os
from typing import Optional
from uuid import uuid4

from langchain.agents import create_agent

from agent.memory import ConversationMemory
from agent.planner import (
    PlanRunResult,
    PlannerAgent,
    PlanExecutor,
    SubTask,
    SubTaskResult,
    TaskPlanner,
)
from agent.summarizer import ConversationSummarizer
from agent.tools.agent_tools import (fetch_external_data, fill_context_for_report,
                                     get_current_month, get_user_id, get_user_location,
                                     get_weather, rag, rag_summarize, tool_data_service)
from agent.tools.middleware import log_before_model, monitor_tool, report_prompt_switch
from agent.workflows.report_workflow import ReportWorkflow
from model.factory import chat_model
from observability.context import bind_request_context
from observability.metrics import metrics_registry
from observability.tracing import trace_recorder
from safety.security import UnsafeInputError, assert_safe_user_input
from services.persistence import SQLiteStore
from utils.prompt_loader import load_system_prompts


def _default_session_store() -> Optional[SQLiteStore]:
    db_path = os.getenv("AGENT_DB_PATH", "storage/agent.db")
    try:
        return SQLiteStore(db_path)
    except Exception:
        return None


class ReactAgent:
    def __init__(
        self,
        session_store=None,
        enable_summary: bool = True,
        max_messages: int = 20,
    ) -> None:
        store = session_store if session_store is not None else _default_session_store()
        summarizer = ConversationSummarizer() if enable_summary else None
        self.memory = ConversationMemory(
            max_messages=max_messages,
            store=store,
            summarizer=summarizer,
        )
        self.agent = create_agent(
            model=chat_model,
            system_prompt=load_system_prompts(),
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[monitor_tool, log_before_model, report_prompt_switch],
        )
        self.planner_agent = self._build_planner_agent()

    def _build_planner_agent(self) -> PlannerAgent:
        executor = PlanExecutor(max_workers=4)

        def handle_weather(task: SubTask) -> SubTaskResult:
            city = tool_data_service.get_user_location()
            content = tool_data_service.get_weather(city)
            return SubTaskResult(id=task.id, kind=task.kind, success=True, content=content)

        def handle_rag(task: SubTask) -> SubTaskResult:
            content = rag.rag_summarize(task.args.get("query", ""))
            return SubTaskResult(id=task.id, kind=task.kind, success=True, content=content)

        def handle_report(task: SubTask) -> SubTaskResult:
            workflow = ReportWorkflow(tool_service=tool_data_service, rag_service=rag)
            state = workflow.run(task.args.get("query", ""))
            return SubTaskResult(
                id=task.id, kind=task.kind,
                success=not state.get("fallback", False),
                content=state.get("answer", ""),
            )

        def handle_generic(task: SubTask) -> SubTaskResult:
            query = task.args.get("query", "")
            try:
                chunks = list(self.execute_stream(query, session_id=f"planner-{task.id}"))
                content = next((c for c in reversed(chunks) if c), "")
                return SubTaskResult(id=task.id, kind=task.kind, success=bool(content),
                                     content=content)
            except Exception as exc:
                return SubTaskResult(id=task.id, kind=task.kind, success=False,
                                     content="", error=str(exc))

        executor.register_handler("weather", handle_weather)
        executor.register_handler("rag_qa", handle_rag)
        executor.register_handler("report", handle_report)
        executor.register_handler("generic", handle_generic)
        return PlannerAgent(planner=TaskPlanner(), executor=executor)

    def run_plan(self, query: str, request_id: Optional[str] = None,
                 tenant_id: str = "default") -> PlanRunResult:
        request_id = request_id or str(uuid4())
        trace_recorder.start_trace(request_id=request_id, session_id="planner")
        with bind_request_context(request_id=request_id, tenant_id=tenant_id,
                                  session_id="planner"):
            try:
                assert_safe_user_input(query)
            except UnsafeInputError as exc:
                return PlanRunResult(plan=[], results=[], answer=f"请求未执行：{exc}")
            return self.planner_agent.run(query, request_id=request_id)

    def execute_stream(self, query: str, session_id: str = "default",
                       request_id: Optional[str] = None,
                       tenant_id: str = "default"):
        request_id = request_id or str(uuid4())
        trace_recorder.start_trace(request_id=request_id, session_id=session_id)
        request_start = metrics_registry.now()
        with bind_request_context(request_id=request_id, session_id=session_id,
                                  tenant_id=tenant_id):
            try:
                assert_safe_user_input(query)
            except UnsafeInputError as exc:
                metrics_registry.inc_request(status="rejected")
                yield f"请求未执行：{str(exc)}\n"
                return

            history = self.memory.get_messages(session_id, tenant_id=tenant_id)
            input_dict = {"messages": history + [{"role": "user", "content": query}]}

            latest_response = ""
            try:
                with trace_recorder.span(
                        request_id,
                        category="agent",
                        name="execute_stream",
                        metadata={"query": query, "history_count": len(history)},
                ):
                    for chunk in self.agent.stream(
                            input_dict,
                            stream_mode="values",
                            context={"report": False, "request_id": request_id,
                                     "session_id": session_id, "tenant_id": tenant_id},
                    ):
                        latest_message = chunk["messages"][-1]
                        if latest_message.content:
                            latest_response = latest_message.content.strip() + "\n"
                            yield latest_response
            except Exception as exc:
                metrics_registry.inc_request(status="error")
                metrics_registry.observe_request_latency(metrics_registry.elapsed_ms(request_start))
                yield f"Agent 运行失败：{exc}\n"
                return

            if latest_response:
                self.memory.add_message(session_id, "user", query, tenant_id=tenant_id)
                self.memory.add_message(session_id, "assistant", latest_response.strip(),
                                        tenant_id=tenant_id)
                metrics_registry.inc_request(status="success")
            else:
                metrics_registry.inc_request(status="empty")
            metrics_registry.observe_request_latency(metrics_registry.elapsed_ms(request_start))


if __name__ == '__main__':
    agent = ReactAgent()
    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
