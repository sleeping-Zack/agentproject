from __future__ import annotations

import os
import re
from dataclasses import asdict, is_dataclass
from typing import Optional
from uuid import uuid4

from langchain.agents import create_agent
from langchain_core.messages import AIMessage

from agent.answer_schema import AnswerClaim, StructuredAnswer
from agent.budget import BudgetManager
from agent.budgeted_text_model import invoke_budgeted_text_model
from agent.memory import ConversationMemory
from agent.long_term_memory import (
    HybridMemoryExtractor,
    LongTermMemoryService,
    StructuredMemoryExtractor,
)
from agent.planner import (
    PlanRunResult,
    PlannerAgent,
    PlanExecutor,
    SubTask,
    SubTaskResult,
    TaskPlanner,
    TaskRoutingDecision,
)
from agent.policies import PlanValidator, Replanner
from agent.summarizer import ConversationSummarizer
from agent.tools.agent_tools import (fetch_external_data, fill_context_for_report,
                                     get_current_month, get_user_id, get_user_location,
                                     get_weather, rag, rag_summarize, tool_data_service)
from agent.tools.middleware import (
    enforce_model_budget,
    log_before_model,
    monitor_tool,
    report_prompt_switch,
)
from agent.workflows.report_workflow import ReportWorkflow
from agent.verifier import AnswerVerifier
from model.factory import chat_model
from observability.context import bind_request_context
from observability.event_bus import EventBackpressureError, event_bus
from observability.metrics import metrics_registry
from observability.tracing import trace_recorder
from safety.security import UnsafeInputError, assert_safe_user_input
from services.factories import create_memory_index, create_memory_store, create_session_store
from utils.prompt_loader import load_system_prompts


def _default_session_store():
    try:
        return create_session_store()
    except Exception:
        return None


def _memory_extraction_model(prompt: str) -> str:
    started_at = metrics_registry.now()
    model = chat_model.resolve()
    response = invoke_budgeted_text_model(
        model,
        prompt,
        max_output_tokens=int(os.getenv("AGENT_MEMORY_EXTRACTION_MAX_TOKENS", "900")),
        operation="memory-extraction",
        temperature=0,
    )
    usage = getattr(response, "usage_metadata", None) or {}
    response_metadata = getattr(response, "response_metadata", None) or {}
    token_usage = (
        response_metadata.get("token_usage")
        or response_metadata.get("usage")
        or {}
    )
    tokens_in = int(
        usage.get("input_tokens")
        or usage.get("prompt_tokens")
        or token_usage.get("prompt_tokens")
        or token_usage.get("input_tokens")
        or 0
    )
    tokens_out = int(
        usage.get("output_tokens")
        or usage.get("completion_tokens")
        or token_usage.get("completion_tokens")
        or token_usage.get("output_tokens")
        or 0
    )
    if tokens_in:
        metrics_registry.inc_tokens("memory_extraction_input", tokens_in)
    if tokens_out:
        metrics_registry.inc_tokens("memory_extraction_output", tokens_out)
    metrics_registry.observe_histogram(
        "agent_memory_extraction_latency_ms",
        metrics_registry.elapsed_ms(started_at),
    )
    return ReactAgent._message_text(response)


def _default_memory_extractor():
    if (
        os.getenv("AGENT_MEMORY_MODEL_EXTRACTION_ENABLED", "true")
        .strip()
        .lower()
        != "true"
    ):
        return HybridMemoryExtractor()
    return HybridMemoryExtractor(
        semantic_extractor=StructuredMemoryExtractor(_memory_extraction_model)
    )


class ReactAgent:
    manages_budget = True

    def __init__(
        self,
        session_store=None,
        enable_summary: bool = True,
        max_messages: Optional[int] = None,
        memory_store=None,
        long_term_memory: Optional[LongTermMemoryService] = None,
    ) -> None:
        store = session_store if session_store is not None else _default_session_store()
        durable_memory_store = memory_store or create_memory_store()
        self.long_term_memory = long_term_memory or LongTermMemoryService(
            durable_memory_store,
            extractor=_default_memory_extractor(),
            search_index=create_memory_index(),
        )
        summarizer = ConversationSummarizer() if enable_summary else None
        self.memory = ConversationMemory(
            max_messages=max_messages,
            store=store,
            summarizer=summarizer,
            max_context_tokens=int(os.getenv("AGENT_MEMORY_CONTEXT_TOKENS", "10000")),
            summary_store=durable_memory_store,
            long_term_memory=self.long_term_memory,
        )
        system_prompt = load_system_prompts()
        self.agent = create_agent(
            model=chat_model,
            system_prompt=system_prompt,
            tools=[rag_summarize, get_weather, get_user_location, get_user_id,
                   get_current_month, fetch_external_data, fill_context_for_report],
            middleware=[
                monitor_tool,
                enforce_model_budget,
                log_before_model,
                report_prompt_switch,
            ],
        )
        self.direct_agent = create_agent(
            model=chat_model,
            system_prompt=system_prompt,
            tools=[],
            middleware=[
                enforce_model_budget,
                log_before_model,
            ],
        )
        self.planner_agent = self._build_planner_agent()

    def _build_planner_agent(self) -> PlannerAgent:
        executor = PlanExecutor(max_workers=4)

        def handle_weather(task: SubTask) -> SubTaskResult:
            context = task.args.get("_execution_context") or {}
            city = str(task.args.get("city") or tool_data_service.get_user_location())
            content = self._run_planner_tool(
                context.get("request_id"),
                "get_weather",
                {"city": city},
                lambda: tool_data_service.get_weather(city),
            )
            return SubTaskResult(id=task.id, kind=task.kind, success=True, content=content)

        def handle_rag(task: SubTask) -> SubTaskResult:
            context = task.args.get("_execution_context") or {}
            rag_query = self._query_with_dependency_results(task)
            result = self._run_planner_tool(
                context.get("request_id"),
                "rag_summarize",
                {"query": rag_query},
                lambda: rag.rag_summarize_result(
                    rag_query,
                    tenant_id=context.get("tenant_id"),
                    budget_manager=task.budget_manager,
                ),
                result_text=lambda value: value.answer,
            )
            self._record_plan_evidence(context.get("request_id"), result.evidence)
            content = self._replace_numeric_citations(result.answer, result.evidence)
            verification = result.verification or {}
            is_knowledge_gap = result.business_status == "empty"
            success = (
                is_knowledge_gap
                or verification.get("passed", True) is not False
            )
            if not success and result.evidence:
                content = self._extractive_rag_fallback(result.evidence)
                success = True
            return SubTaskResult(
                id=task.id,
                kind=task.kind,
                success=success,
                content=content,
                error=(
                    ",".join(verification.get("reasons") or ())
                    if not success
                    else None
                ),
            )

        def handle_report(task: SubTask) -> SubTaskResult:
            context = task.args.get("_execution_context") or {}
            workflow = ReportWorkflow(tool_service=tool_data_service, rag_service=rag)
            state = workflow.run(
                task.args.get("query", ""),
                user_id=(
                    task.args.get("user_id")
                    or context.get("data_user_id")
                    or context.get("user_id")
                ),
                month=task.args.get("month"),
                tenant_id=context.get("tenant_id"),
                intent="report",
                budget_manager=task.budget_manager,
            )
            evidence = state.get("evidence") or []
            self._record_plan_evidence(context.get("request_id"), evidence)
            answer = self._replace_numeric_citations(state.get("answer", ""), evidence)
            return SubTaskResult(
                id=task.id, kind=task.kind,
                success=not state.get("fallback", False),
                content=answer,
            )

        def handle_generic(task: SubTask) -> SubTaskResult:
            query = self._query_with_dependency_results(task)
            context = task.args.get("_execution_context") or {}
            try:
                chunks = list(
                    self.execute_stream(
                        query,
                        session_id=context.get("session_id") or f"planner-{task.id}",
                        request_id=context.get("request_id"),
                        tenant_id=context.get("tenant_id", "default"),
                        user_id=context.get("user_id"),
                        data_user_id=context.get("data_user_id"),
                        user_role=context.get("user_role", "user"),
                        scene=context.get("scene", "default"),
                        approval_id=context.get("approval_id"),
                        execution_mode="direct",
                        budget_manager=task.budget_manager,
                        emit_events=bool(context.get("emit_events")),
                        publish_answer_tokens=False,
                    )
                )
                content = next((c for c in reversed(chunks) if c), "")
                verification_error = self._verify_dependency_synthesis(
                    query,
                    content,
                    task.args.get("_dependency_results") or {},
                )
                return SubTaskResult(
                    id=task.id,
                    kind=task.kind,
                    success=bool(content) and verification_error is None,
                    content=content if verification_error is None else "",
                    error=verification_error,
                )
            except Exception as exc:
                return SubTaskResult(id=task.id, kind=task.kind, success=False,
                                     content="", error=str(exc))

        executor.register_handler("weather", handle_weather)
        executor.register_handler("rag_qa", handle_rag)
        executor.register_handler("report", handle_report)
        executor.register_handler("generic", handle_generic)
        return PlannerAgent(
            planner=TaskPlanner(),
            executor=executor,
            validator=PlanValidator(),
            replanner=Replanner(),
            max_replans=1,
        )

    @staticmethod
    def _run_planner_tool(
        request_id: str | None,
        tool_name: str,
        args: dict,
        handler,
        *,
        result_text=str,
    ):
        if not request_id:
            return handler()
        with trace_recorder.span(
            request_id,
            category="tool",
            name=tool_name,
            metadata={"redacted_args": dict(args)},
        ) as event:
            result = handler()
            text = str(result_text(result))
            event.metadata["result"] = text[:4000]
            event.metadata["result_truncated"] = len(text) > 4000
            business_status = getattr(result, "business_status", None)
            if business_status:
                event.metadata["business_status"] = str(business_status)
            verification = getattr(result, "verification", None)
            if isinstance(verification, dict):
                event.metadata["verification"] = {
                    key: verification[key]
                    for key in (
                        "passed",
                        "action",
                        "reasons",
                        "dense_relevance",
                        "sparse_relevance",
                    )
                    if key in verification
                }
            return result

    @staticmethod
    def _query_with_dependency_results(task: SubTask) -> str:
        query = str(task.args.get("query") or "")
        dependency_results = task.args.get("_dependency_results") or {}
        if not dependency_results:
            return query
        context = "\n".join(
            f"- {dependency}: {str(content)[:1600]}"
            for dependency, content in dependency_results.items()
        )
        return (
            f"{query}\n\n以下是已经完成并通过步骤级处理的前置结果，"
            "请仅根据这些结果完成当前目标，不得补充其中没有出现的设备型号、"
            "功能设置、因果关系或数值。证据不足时必须明确说明无法确定；"
            "每条结论都要标注直接支持它的前置步骤编号，例如 [t1]。\n"
            f"{context}"
        )[:5000]

    @staticmethod
    def _verify_dependency_synthesis(
        query: str,
        content: str,
        dependency_results: dict,
    ) -> Optional[str]:
        if not content:
            return "empty_synthesis"
        if not dependency_results:
            return None
        evidence = [
            {
                "id": dependency,
                "source": "planner_dependency",
                "content": str(result),
            }
            for dependency, result in dependency_results.items()
        ]
        claims = [
            re.sub(r"\[[^\]]+\]", "", claim).strip()
            for claim in re.split(r"(?<=[。！？!?；;.])\s*|\n+", content)
            if claim.strip()
            and claim.strip(" #*-")
            and not re.fullmatch(r"你好[！!，,\s]*", claim.strip())
        ]
        structured = StructuredAnswer(
            summary=content,
            claims=[
                AnswerClaim(
                    text=claim,
                    evidence_ids=list(dependency_results),
                )
                for claim in claims
            ],
            citations=list(dependency_results),
        )
        verified = AnswerVerifier().verify(
            query=query,
            answer=content,
            evidence=evidence,
            scene="planner_synthesis",
            structured_answer=structured,
        )
        if verified.passed:
            return None
        return "subtask_verification_failed:" + ",".join(verified.reasons)

    @staticmethod
    def _extractive_rag_fallback(evidence) -> str:
        sections = [
            "生成式总结未通过证据一致性校验，已安全降级为知识库原文摘录："
        ]
        for item in list(evidence)[:4]:
            if is_dataclass(item):
                payload = asdict(item)
            elif isinstance(item, dict):
                payload = item
            else:
                payload = {
                    "id": getattr(item, "id", ""),
                    "source": getattr(item, "source", ""),
                    "content": getattr(item, "content", ""),
                }
            evidence_id = str(payload.get("id") or "evidence")
            source = str(payload.get("source") or "知识库")
            content = str(payload.get("content") or "").strip()[:800]
            if content:
                sections.append(f"\n- [{evidence_id}] {source}：{content}")
        return "".join(sections)

    def run_plan(
        self,
        query: str,
        request_id: Optional[str] = None,
        tenant_id: str = "default",
        budget_manager: Optional[BudgetManager] = None,
        session_id: str = "default",
        user_id: Optional[str] = None,
        data_user_id: Optional[str] = None,
        user_role: str = "user",
        scene: str = "default",
        approval_id: Optional[str] = None,
        emit_events: bool = False,
        routing_decision: Optional[TaskRoutingDecision] = None,
    ) -> PlanRunResult:
        request_id = request_id or str(uuid4())
        self._ensure_trace(request_id, session_id)
        with bind_request_context(request_id=request_id, tenant_id=tenant_id,
                                  session_id=session_id, user_id=user_id,
                                  data_user_id=data_user_id):
            try:
                assert_safe_user_input(query)
            except UnsafeInputError as exc:
                return PlanRunResult(plan=[], results=[], answer=f"请求未执行：{exc}")

            def publish_event(event_type: str, payload: dict) -> None:
                if not emit_events or not event_bus.exists(request_id):
                    return
                try:
                    event_bus.publish(request_id, event_type, payload)
                except (EventBackpressureError, RuntimeError):
                    return

            return self.planner_agent.run(
                query,
                request_id=request_id,
                budget_manager=budget_manager,
                routing_decision=routing_decision,
                task_context={
                    "request_id": request_id,
                    "session_id": session_id,
                    "tenant_id": tenant_id,
                    "user_id": user_id,
                    "data_user_id": data_user_id,
                    "user_role": user_role,
                    "scene": scene,
                    "approval_id": approval_id,
                    "emit_events": emit_events,
                },
                event_callback=publish_event,
            )

    def execute_stream(self, query: str, session_id: str = "default",
                       request_id: Optional[str] = None,
                       tenant_id: str = "default",
                       user_id: Optional[str] = None,
                       data_user_id: Optional[str] = None,
                       user_role: str = "user",
                       scene: str = "default",
                       approval_id: Optional[str] = None,
                       execution_mode: str = "react",
                       max_tool_calls: Optional[int] = None,
                       budget_manager: Optional[BudgetManager] = None,
                       max_model_output_tokens: Optional[int] = None,
                       estimated_cost_per_1k_tokens: Optional[float] = None,
                       emit_events: bool = False,
                       publish_answer_tokens: bool = True):
        request_id = request_id or str(uuid4())
        if budget_manager is None and max_tool_calls is not None:
            budget_manager = BudgetManager(max_tool_calls=max_tool_calls)
        self._ensure_trace(request_id, session_id)
        request_start = metrics_registry.now()
        with bind_request_context(
            request_id=request_id,
            session_id=session_id,
            tenant_id=tenant_id,
            user_id=user_id,
            data_user_id=data_user_id,
        ):
            try:
                assert_safe_user_input(query)
            except UnsafeInputError as exc:
                metrics_registry.inc_request(status="rejected")
                yield f"请求未执行：{str(exc)}\n"
                return

            build_context = getattr(self.memory, "build_context", None)
            if callable(build_context):
                history = build_context(
                    session_id,
                    query,
                    tenant_id=tenant_id,
                    user_id=user_id,
                )
            else:
                history = self.memory.get_messages(session_id, tenant_id=tenant_id)
            input_dict = {"messages": history + [{"role": "user", "content": query}]}

            latest_response = ""
            pending_response = ""
            try:
                with trace_recorder.span(
                        request_id,
                        category="agent",
                        name="execute_stream",
                        metadata={"query": query, "history_count": len(history)},
                ):
                    runtime_context = {
                        "report": False,
                        "request_id": request_id,
                        "session_id": session_id,
                        "tenant_id": tenant_id,
                        "user_id": user_id,
                        "data_user_id": data_user_id,
                        "user_role": user_role,
                        "scene": scene,
                        "approval_id": approval_id,
                        "execution_mode": execution_mode,
                        "max_tool_calls": max_tool_calls,
                        "budget_manager": budget_manager,
                        "max_model_output_tokens": max_model_output_tokens,
                        "estimated_cost_per_1k_tokens": estimated_cost_per_1k_tokens,
                        "emit_events": emit_events,
                        "max_rag_calls": int(os.getenv("AGENT_MAX_RAG_CALLS", "3")),
                        "final_response_token_reserve": int(
                            os.getenv("AGENT_FINAL_RESPONSE_TOKEN_RESERVE", "4500")
                        ),
                        "rag_duplicate_threshold": float(
                            os.getenv("AGENT_RAG_DUPLICATE_THRESHOLD", "0.55")
                        ),
                    }
                    execution_agent = (
                        self.direct_agent if execution_mode == "direct" else self.agent
                    )
                    for part in execution_agent.stream(
                            input_dict,
                            stream_mode=["messages", "updates"],
                            context=runtime_context,
                            config={
                                "recursion_limit": max(
                                    1,
                                    int(os.getenv("AGENT_MAX_REACT_RECURSION", "12")),
                                )
                            },
                            version="v2",
                    ):
                        if emit_events and event_bus.is_cancelled(request_id):
                            break
                        part_type = part.get("type") if isinstance(part, dict) else None
                        if part_type == "messages":
                            message, _metadata = part["data"]
                            delta = self._message_text(message)
                            if not delta:
                                continue
                            pending_response += delta
                            continue
                        if part_type == "updates":
                            for update in (part.get("data") or {}).values():
                                messages = update.get("messages", []) if isinstance(update, dict) else []
                                if not messages:
                                    continue
                                latest_message = messages[-1]
                                if not isinstance(latest_message, AIMessage):
                                    continue
                                self._record_model_usage(request_id, latest_message)
                                if getattr(latest_message, "tool_calls", None):
                                    pending_response = ""
                                    continue
                                full_text = self._message_text(latest_message)
                                if not full_text:
                                    full_text = pending_response
                                pending_response = ""
                                if full_text:
                                    latest_response = full_text
                                    if publish_answer_tokens:
                                        self._publish_token(request_id, full_text, emit_events)
                                    yield latest_response
                    if pending_response and not latest_response:
                        latest_response = pending_response
                        if publish_answer_tokens:
                            self._publish_token(request_id, pending_response, emit_events)
                        yield latest_response
            except Exception:
                metrics_registry.inc_request(status="error")
                metrics_registry.observe_request_latency(metrics_registry.elapsed_ms(request_start))
                raise

            if latest_response:
                metrics_registry.inc_request(status="success")
            else:
                metrics_registry.inc_request(status="empty")
            metrics_registry.observe_request_latency(metrics_registry.elapsed_ms(request_start))

    @staticmethod
    def _message_text(message) -> str:
        text = getattr(message, "text", None)
        if isinstance(text, str) and text:
            return text
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        if isinstance(content, list):
            return "".join(
                str(block.get("text", ""))
                for block in content
                if isinstance(block, dict) and block.get("type") == "text"
            )
        return ""

    @staticmethod
    def _publish_token(request_id: str, delta: str, emit_events: bool) -> None:
        if not emit_events or not event_bus.exists(request_id):
            return
        try:
            event_bus.publish(
                request_id,
                "token_delta",
                {"delta": delta, "provisional": False, "replace": True},
            )
        except (EventBackpressureError, RuntimeError):
            return

    @staticmethod
    def _ensure_trace(request_id: str, session_id: str) -> None:
        try:
            trace_recorder.export_trace(request_id)
        except KeyError:
            trace_recorder.start_trace(request_id=request_id, session_id=session_id)

    @staticmethod
    def _record_plan_evidence(request_id: Optional[str], evidence) -> None:
        if not request_id or not evidence:
            return
        serialised = [
            asdict(item)
            if is_dataclass(item)
            else dict(item)
            if isinstance(item, dict)
            else {
                "id": getattr(item, "id", ""),
                "source": getattr(item, "source", ""),
                "content": getattr(item, "content", ""),
            }
            for item in evidence
        ]
        try:
            with trace_recorder.span(
                request_id,
                category="rag",
                name="evidence",
                metadata={"evidence": serialised},
            ):
                pass
        except KeyError:
            return

    @staticmethod
    def _replace_numeric_citations(answer: str, evidence) -> str:
        evidence_ids = [
            str(
                item.get("id")
                if isinstance(item, dict)
                else getattr(item, "id", "")
            )
            for item in evidence
        ]

        def replace(match: re.Match[str]) -> str:
            index = int(match.group(1)) - 1
            if 0 <= index < len(evidence_ids) and evidence_ids[index]:
                return evidence_ids[index]
            return match.group(0)

        return re.sub(r"\[\s*(\d+)\s*\]", replace, str(answer or ""))

    def _record_model_usage(self, request_id: str, message) -> None:
        usage = getattr(message, "usage_metadata", None) or {}
        response_metadata = getattr(message, "response_metadata", None) or {}
        token_usage = response_metadata.get("token_usage") or response_metadata.get("usage") or {}
        tokens_in = int(
            usage.get("input_tokens")
            or usage.get("prompt_tokens")
            or token_usage.get("prompt_tokens")
            or token_usage.get("input_tokens")
            or 0
        )
        tokens_out = int(
            usage.get("output_tokens")
            or usage.get("completion_tokens")
            or token_usage.get("completion_tokens")
            or token_usage.get("output_tokens")
            or 0
        )
        if not tokens_in and not tokens_out:
            return
        cost_per_1k = float(os.getenv("AGENT_ESTIMATED_COST_PER_1K_TOKENS", "0.001"))
        cost = round(((tokens_in + tokens_out) / 1000.0) * cost_per_1k, 6)
        trace_recorder.record_diagnostic_event(
            request_id=request_id,
            step_id="model-usage",
            event_type="model_usage",
            status="ok",
            latency_ms=0.0,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            cost_mode="actual",
            model_name=type(chat_model).__name__,
        )


if __name__ == '__main__':
    agent = ReactAgent()
    for chunk in agent.execute_stream("给我生成我的使用报告"):
        print(chunk, end="", flush=True)
