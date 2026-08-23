from __future__ import annotations

import asyncio
import hashlib
import inspect
import os
import re
import time
from dataclasses import dataclass, field, replace
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from agent.answer_schema import AnswerClaim, StructuredAnswer
from agent.budget import (
    DEFAULT_MAX_COST,
    DEFAULT_MAX_MODEL_OUTPUT_TOKENS,
    DEFAULT_MAX_STEPS,
    DEFAULT_MAX_TOKENS,
    DEFAULT_MAX_TOOL_CALLS,
    DEFAULT_MAX_VERIFICATION_RETRIES,
    DEFAULT_MIN_REPAIR_TOKENS,
    BudgetExceeded,
    Reservation,
    bind_budget_manager,
)
from agent.memory import ConversationMemory
from agent.planner import RoutingContext, TaskRouter, TaskRoutingDecision
from agent.policies import PolicyAction, ToolPolicy
from agent.state import AgentState, ArtifactRef, Budget, Observation, ToolCallRecord
from agent.tools.registry import ToolRegistry, build_default_tool_registry
from agent.verifier import AnswerVerifier, VerifyResult, build_default_answer_verifier
from observability.event_bus import EventBackpressureError, event_bus
from observability.context import bind_request_context
from observability.metrics import metrics_registry
from observability.tracing import trace_recorder
from rag.rag_utils import knowledge_gap_answer
from safety.security import UnsafeInputError, assert_safe_user_input
from services.approval_store import ApprovalStore
from services.artifact_store import ArtifactStore
from services.factories import create_approval_store, create_artifact_store
from utils.config_handler import agent_conf
from utils.streaming import get_final_response


@dataclass
class AgentTask:
    query: str
    session_id: str = "default"
    tenant_id: str = "default"
    user_id: Optional[str] = None
    data_user_id: Optional[str] = None
    user_role: str = "user"
    scene: str = "default"
    request_id: str = field(default_factory=lambda: str(uuid4()))
    approval_id: Optional[str] = None
    emit_events: bool = False
    execution_mode: str = "react"
    routing_decision: Optional[TaskRoutingDecision] = None
    required_tools: tuple[str, ...] = ()


@dataclass
class AgentBackendResult:
    answer: str
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    model_name: str = ""
    tokens_in: int = 0
    tokens_out: int = 0
    cost: float = 0.0
    cost_mode: str = "estimated"
    budget_accounted: bool = False
    structured_answer: Optional[StructuredAnswer] = None
    safe_fallback_answer: str = ""
    safe_fallback_structured_answer: Optional[StructuredAnswer] = None
    safe_fallback_strategy: str = "verified_evidence_excerpt"


def _public_citation_evidence(
    observations: List[Observation],
    answer: str,
) -> List[Dict[str, Any]]:
    """Return the cited evidence fields that are safe for client-side previews."""

    cited: List[tuple[int, Dict[str, Any]]] = []
    seen_ids: set[str] = set()
    for observation in observations:
        item = observation.metadata if isinstance(observation.metadata, dict) else {}
        metadata = item.get("metadata")
        metadata = metadata if isinstance(metadata, dict) else {}
        raw_evidence_id = (
            item.get("id") or item.get("evidence_id") or observation.source or ""
        )
        if not isinstance(raw_evidence_id, (str, int, float)):
            continue
        evidence_id = str(raw_evidence_id).strip()
        if not evidence_id or evidence_id in seen_ids:
            continue
        citation = re.search(rf"\[\s*{re.escape(evidence_id)}\s*\]", answer or "")
        if citation is None:
            continue

        raw_source = (
            item.get("source")
            or metadata.get("source_name")
            or metadata.get("source")
            or "知识库"
        )
        raw_source = raw_source if isinstance(raw_source, str) else "知识库"
        # A retrieved document may carry an absolute source path. The client only
        # needs its friendly file name, never deployment or workstation paths.
        source = raw_source.strip().replace("\\", "/").rsplit("/", 1)[-1][:200]
        raw_content = item.get("content") or observation.content
        raw_content = raw_content if isinstance(raw_content, str) else ""
        excerpt = " ".join(raw_content.split())
        if len(excerpt) > 360:
            excerpt = f"{excerpt[:359]}…"

        public_item: Dict[str, Any] = {
            "id": evidence_id,
            "source": source or "知识库",
            "excerpt": excerpt,
        }
        for public_key, metadata_key in (
            ("title", "document_title"),
            ("section", "section_title"),
            ("chunk_index", "chunk_index"),
        ):
            value = metadata.get(metadata_key)
            if value in (None, ""):
                continue
            if isinstance(value, bool) or not isinstance(value, (str, int, float)):
                continue
            if public_key == "title" and isinstance(value, str):
                value = value.strip().replace("\\", "/").rsplit("/", 1)[-1]
            public_item[public_key] = value if not isinstance(value, str) else value[:200]

        page = metadata.get("page_label")
        if page in (None, ""):
            page = metadata.get("page")
        if not isinstance(page, bool) and isinstance(page, (str, int, float)):
            public_item["page"] = page if not isinstance(page, str) else page[:40]

        seen_ids.add(evidence_id)
        cited.append((citation.start(), public_item))

    cited.sort(key=lambda pair: pair[0])
    return [item for _, item in cited]


def _targeted_repair_task(
    task: AgentTask,
    backend_result: AgentBackendResult,
    verifier_result: VerifyResult,
) -> AgentTask:
    evidence_sections = []
    for item in backend_result.evidence[:5]:
        evidence_id = str(item.get("id") or item.get("source") or "evidence")
        content = str(item.get("content") or "").strip()[:300]
        if content:
            evidence_sections.append(f"[{evidence_id}] {content}")
    evidence_text = "\n".join(evidence_sections) or "（没有可用证据）"
    reasons = "、".join(str(reason) for reason in verifier_result.reasons) or "证据校验失败"
    prompt = (
        "你正在修复一份未通过证据校验的回答。只允许使用下方已有证据，"
        "不要调用工具、不要补充证据外的事实。删除无法支持的结论，并在每项结论后"
        "用 [证据ID] 标注来源。直接输出修复后的最终回答。\n\n"
        f"原始问题：\n{task.query[:600]}\n\n"
        f"校验失败原因：\n{reasons}\n\n"
        f"待修复回答：\n{backend_result.answer[:1000]}\n\n"
        f"可用证据：\n{evidence_text}"
    )
    return AgentTask(
        query=prompt,
        session_id=task.session_id,
        tenant_id=task.tenant_id,
        user_id=task.user_id,
        data_user_id=task.data_user_id,
        user_role=task.user_role,
        scene=task.scene,
        request_id=task.request_id,
        approval_id=task.approval_id,
        emit_events=task.emit_events,
        execution_mode="direct",
        routing_decision=TaskRoutingDecision(
            execution_mode="direct",
            complexity_score=0,
            reasons=("verification_targeted_repair",),
            confidence=1.0,
            decision_source="verification_targeted_repair",
            proposed_mode="direct",
        ),
        required_tools=(),
    )


def _merge_repair_result(
    previous: AgentBackendResult,
    repaired: AgentBackendResult,
) -> AgentBackendResult:
    evidence: List[Dict[str, Any]] = []
    seen_evidence: set[tuple[str, str]] = set()
    for item in (*previous.evidence, *repaired.evidence):
        key = (
            str(item.get("id") or item.get("source") or ""),
            str(item.get("content") or ""),
        )
        if key in seen_evidence:
            continue
        seen_evidence.add(key)
        evidence.append(item)
    return AgentBackendResult(
        answer=repaired.answer,
        evidence=evidence,
        tool_results=[*previous.tool_results, *repaired.tool_results],
        model_name=repaired.model_name,
        tokens_in=repaired.tokens_in,
        tokens_out=repaired.tokens_out,
        cost=repaired.cost,
        cost_mode=repaired.cost_mode,
        budget_accounted=repaired.budget_accounted,
        structured_answer=repaired.structured_answer,
        safe_fallback_answer=previous.safe_fallback_answer,
        safe_fallback_structured_answer=previous.safe_fallback_structured_answer,
        safe_fallback_strategy=previous.safe_fallback_strategy,
    )


@dataclass
class AgentRunResult:
    state: AgentState
    answer: str
    request_id: str
    approval_id: Optional[str] = None
    artifacts: List[ArtifactRef] = field(default_factory=list)
    verifier: Optional[VerifyResult] = None
    evidence: List[Dict[str, Any]] = field(default_factory=list)


class ReactAgentBackend:
    defers_answer_tokens = True

    def __init__(self, agent=None) -> None:
        self.agent = agent

    @property
    def manages_budget(self) -> bool:
        # The lazily-created production ReactAgent installs budget middleware.
        # Test/legacy agents opt in explicitly via ``manages_budget``.
        return self.agent is None or bool(getattr(self.agent, "manages_budget", False))

    def __call__(self, task: AgentTask, state: AgentState) -> AgentBackendResult:
        if self.agent is None:
            from agent.react_agent import ReactAgent

            self.agent = ReactAgent()
        try:
            trace_event_offset = len(
                trace_recorder.export_trace(task.request_id).get("events", [])
            )
        except KeyError:
            trace_event_offset = 0
        budget_manager = state.budget.manager if state is not None else None
        chunks = list(
            self.agent.execute_stream(
                task.query,
                session_id=task.session_id,
                request_id=task.request_id,
                tenant_id=task.tenant_id,
                user_id=task.user_id,
                data_user_id=task.data_user_id,
                user_role=task.user_role,
                scene=task.scene,
                approval_id=task.approval_id,
                execution_mode=task.execution_mode,
                max_tool_calls=state.budget.max_tool_calls if state is not None else None,
                budget_manager=budget_manager,
                max_model_output_tokens=(
                    min(
                        DEFAULT_MAX_MODEL_OUTPUT_TOKENS,
                        state.budget.manager.remaining_output_tokens(),
                    )
                    if state is not None
                    else DEFAULT_MAX_MODEL_OUTPUT_TOKENS
                ),
                emit_events=task.emit_events,
                publish_answer_tokens=False,
            )
        )
        full_trace_payload = trace_recorder.export_trace(task.request_id)
        trace_payload = {
            **full_trace_payload,
            "events": full_trace_payload.get("events", [])[trace_event_offset:],
        }
        evidence = self._extract_evidence(trace_payload)
        tokens_in, tokens_out, cost, cost_mode = self._extract_usage(trace_payload)
        tool_results = self._extract_tool_results(trace_payload)
        fallback_answer, fallback_structured, fallback_strategy = self._evidence_fallback(
            evidence,
            tool_results,
        )
        return AgentBackendResult(
            answer=get_final_response(chunks),
            evidence=evidence,
            tool_results=tool_results,
            model_name=type(getattr(self.agent, "agent", self.agent)).__name__,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            cost_mode=cost_mode,
            budget_accounted=budget_manager is not None and self.manages_budget,
            safe_fallback_answer=fallback_answer,
            safe_fallback_structured_answer=fallback_structured,
            safe_fallback_strategy=fallback_strategy,
        )

    def repair(
        self,
        task: AgentTask,
        state: AgentState,
        previous: AgentBackendResult,
        verifier_result: VerifyResult,
    ) -> AgentBackendResult:
        repair_task = _targeted_repair_task(task, previous, verifier_result)
        return _merge_repair_result(previous, self(repair_task, state))

    @staticmethod
    def _evidence_fallback(
        evidence: List[Dict[str, Any]],
        tool_results: Optional[List[Dict[str, Any]]] = None,
    ) -> tuple[str, Optional[StructuredAnswer], str]:
        evidence_ids = {
            str(item.get("id") or item.get("evidence_id") or "").strip()
            for item in evidence
        }
        rag_results = [
            item
            for item in (tool_results or [])
            if str(item.get("tool") or "") == "rag_summarize"
        ]
        for result in reversed(rag_results):
            metadata = result.get("metadata") or {}
            verification = metadata.get("verification") or {}
            content = str(result.get("content") or "").strip()
            has_current_citation = any(
                evidence_id and evidence_id in content for evidence_id in evidence_ids
            ) or (
                len(rag_results) == 1
                and bool(re.search(r"\[\s*\d+\s*\]", content))
            )
            if (
                str(result.get("status") or "") == "success"
                and verification.get("passed") is True
                and not result.get("result_truncated")
                and content
                and has_current_citation
            ):
                return content, None, "verified_rag_answer"

        claims: List[AnswerClaim] = []
        citations: List[str] = []
        lines = ["生成式回答未通过校验，以下为知识库中可直接核验的原文："]
        for item in evidence[:5]:
            evidence_id = str(
                item.get("id") or item.get("evidence_id") or ""
            ).strip()
            content = str(item.get("content") or "").strip()[:800]
            source = str(item.get("source") or "知识库")
            if not evidence_id or not content:
                continue
            lines.append(f"- [{evidence_id}] {source}：{content}")
            citations.append(evidence_id)
            claims.extend(
                AnswerClaim(text=claim, evidence_ids=[evidence_id])
                for claim in re.split(r"(?<=[。！？!?；;.])\s*|\n+", content)
                if claim.strip()
            )
        if not claims:
            return "", None, "verified_evidence_excerpt"
        answer = "\n".join(lines)
        return (
            answer,
            StructuredAnswer(
                summary=answer,
                claims=claims,
                citations=citations,
            ),
            "verified_evidence_excerpt",
        )

    @staticmethod
    def _extract_tool_results(trace_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        tool_results = []
        pending_rag_results: List[Dict[str, Any]] = []
        for event in trace_payload.get("events", []):
            if event.get("category") == "rag" and event.get("name") == "evidence":
                pending_rag_results.append(dict(event.get("metadata") or {}))
                continue
            if event.get("category") != "tool":
                continue
            metadata = dict(event.get("metadata") or {})
            status = "error" if event.get("error") else "success"
            if event.get("name") == "rag_summarize":
                rag_metadata = pending_rag_results.pop(0) if pending_rag_results else {}
                metadata = {**rag_metadata, **metadata}
                business_status = str(metadata.get("business_status") or "")
                if business_status in {
                    "success",
                    "degraded",
                    "empty",
                    "verification_failed",
                    "error",
                }:
                    status = business_status
            tool_results.append({
                "tool": event["name"],
                "status": status,
                "args": dict(metadata.get("redacted_args") or {}),
                "content": str(metadata.get("result") or ""),
                "result_truncated": bool(metadata.get("result_truncated", False)),
                "metadata": metadata,
            })
        return tool_results

    @staticmethod
    def _extract_evidence(trace_payload: Dict[str, Any]) -> List[Dict[str, Any]]:
        evidence: List[Dict[str, Any]] = []
        for event in trace_payload.get("events", []):
            metadata = event.get("metadata", {})
            if event.get("category") == "rag" and event.get("name") == "evidence":
                for item in metadata.get("evidence", []):
                    if isinstance(item, dict):
                        evidence.append(item)
            if event.get("category") == "diagnostic" and metadata.get("type") == "rag_evidence":
                for item in metadata.get("evidence", []):
                    if isinstance(item, dict):
                        evidence.append(item)
        return evidence

    @staticmethod
    def _extract_usage(trace_payload: Dict[str, Any]) -> tuple[int, int, float, str]:
        tokens_in = 0
        tokens_out = 0
        cost = 0.0
        cost_mode = "estimated"
        for event in trace_payload.get("events", []):
            metadata = event.get("metadata", {})
            if event.get("category") != "diagnostic" or metadata.get("type") != "model_usage":
                continue
            tokens_in += int(metadata.get("tokens_in") or 0)
            tokens_out += int(metadata.get("tokens_out") or 0)
            cost += float(metadata.get("cost") or 0.0)
            cost_mode = metadata.get("cost_mode") or "actual"
        return tokens_in, tokens_out, round(cost, 6), cost_mode


class PlannerAgentBackend:
    """Adapt PlannerAgent to the same governed backend contract as ReAct."""

    manages_budget = True
    defers_answer_tokens = True

    def __init__(self, agent) -> None:
        self.agent = agent

    def __call__(self, task: AgentTask, state: AgentState) -> AgentBackendResult:
        try:
            trace_event_offset = len(
                trace_recorder.export_trace(task.request_id).get("events", [])
            )
        except KeyError:
            trace_event_offset = 0
        plan_kwargs = {
            "request_id": task.request_id,
            "tenant_id": task.tenant_id,
            "budget_manager": state.budget.manager,
            "session_id": task.session_id,
            "user_id": task.user_id,
            "data_user_id": task.data_user_id,
            "user_role": task.user_role,
            "scene": task.scene,
            "approval_id": task.approval_id,
            "emit_events": task.emit_events,
        }
        if task.routing_decision is not None:
            plan_kwargs["routing_decision"] = task.routing_decision
        plan_result = self.agent.run_plan(task.query, **plan_kwargs)
        state.plan = [
            {
                "id": item.id,
                "kind": item.kind,
                "description": item.description,
                "depends_on": list(item.depends_on),
                "arguments": {
                    key: value
                    for key, value in item.args.items()
                    if not key.startswith("_")
                },
            }
            for item in plan_result.plan
        ]
        state.current_step = len(plan_result.results)
        full_trace_payload = trace_recorder.export_trace(task.request_id)
        trace_payload = {
            **full_trace_payload,
            "events": full_trace_payload.get("events", [])[trace_event_offset:],
        }
        tokens_in, tokens_out, cost, cost_mode = ReactAgentBackend._extract_usage(
            trace_payload
        )
        planner_results = [
            {
                "tool": f"plan:{result.kind}",
                "status": "success" if result.success else "failed",
                "args": {"step_id": result.id},
                "content": result.content,
                "error": result.error,
                "record_type": "planner_step",
            }
            for result in plan_result.results
        ]
        tool_results = [
            *planner_results,
            *ReactAgentBackend._extract_tool_results(trace_payload),
        ]
        successful_results = [
            result
            for result in plan_result.results
            if result.success and result.content.strip()
        ]
        task_by_id = {item.id: item for item in plan_result.plan}
        result_by_id = {item.id: item for item in successful_results}

        def source_evidence_ids(
            result_id: str,
            visiting: Optional[set[str]] = None,
        ) -> List[str]:
            visiting = set(visiting or ())
            if result_id in visiting:
                return []
            visiting.add(result_id)
            source_result = result_by_id.get(result_id)
            source_task = task_by_id.get(result_id)
            if source_result is None or source_task is None:
                return []
            if source_result.kind != "generic":
                return [f"plan-step-{result_id}"]
            return list(
                dict.fromkeys(
                    evidence_id
                    for dependency in source_task.depends_on
                    for evidence_id in source_evidence_ids(dependency, visiting)
                )
            )

        source_results = [
            result for result in successful_results if result.kind != "generic"
        ]
        derived_evidence = [
            {
                "id": f"plan-step-{result.id}",
                "source": "planner",
                "content": result.content,
                "metadata": {
                    "step_id": result.id,
                    "kind": result.kind,
                    "lineage": "verified_source_subtask_output",
                },
            }
            for result in source_results
        ]
        raw_evidence = ReactAgentBackend._extract_evidence(trace_payload)
        claims = [
            AnswerClaim(
                text=claim,
                evidence_ids=source_evidence_ids(result.id),
            )
            for result in successful_results
            for claim in re.split(
                r"(?<=[。！？!?；;.])\s*|\n+",
                result.content,
            )
            if claim.strip()
        ]
        citations = list(
            dict.fromkeys(
                evidence_id
                for result in successful_results
                for evidence_id in source_evidence_ids(result.id)
            )
        )
        structured_answer = StructuredAnswer(
            summary=plan_result.answer,
            claims=claims,
            citations=citations,
        )
        safe_fallback_answer = ""
        safe_fallback_structured_answer = None
        if source_results:
            kind_label = {
                "weather": "环境与天气",
                "rag_qa": "知识库资料",
                "report": "使用报告",
            }
            safe_sections = [
                "## 已完成部分",
                "本次执行预算不足以继续安全重试，以下仅展示已经通过步骤级校验的结果；"
                "未完成的综合判断和处理顺序没有擅自补写。",
            ]
            safe_claims: List[AnswerClaim] = []
            safe_citations: List[str] = []
            for result in source_results:
                evidence_id = f"plan-step-{result.id}"
                content = result.content.strip()[:1400]
                if not content:
                    continue
                safe_sections.append(
                    f"\n### {kind_label.get(result.kind, result.kind)}\n{content}"
                )
                for claim in re.split(r"(?<=[。！？；;.])\s*|\n+", content):
                    claim = claim.strip()
                    meaningful = re.sub(
                        r"[\s#*`_\-\d.、:：\[\]()（）]+",
                        "",
                        claim,
                    )
                    if not meaningful:
                        continue
                    safe_claims.append(
                        AnswerClaim(
                            text=claim,
                            evidence_ids=[evidence_id],
                        )
                    )
                safe_citations.append(evidence_id)
            if safe_claims:
                safe_fallback_answer = "\n".join(safe_sections)
                safe_fallback_structured_answer = StructuredAnswer(
                    summary=safe_fallback_answer,
                    claims=safe_claims,
                    citations=safe_citations,
                )
        return AgentBackendResult(
            answer=plan_result.answer,
            evidence=[*raw_evidence, *derived_evidence],
            tool_results=tool_results,
            model_name="PlannerAgent",
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            cost_mode=cost_mode,
            budget_accounted=True,
            structured_answer=structured_answer,
            safe_fallback_answer=safe_fallback_answer,
            safe_fallback_structured_answer=safe_fallback_structured_answer,
        )


class AutoRoutingBackend:
    """Route every chat request without exposing a separate task-center API."""

    defers_answer_tokens = True

    def __init__(
        self,
        *,
        router: Optional[TaskRouter] = None,
        react_backend: Optional[Callable[[AgentTask, AgentState], AgentBackendResult]] = None,
        planner_backend: Optional[Callable[[AgentTask, AgentState], AgentBackendResult]] = None,
        tool_registry: Optional[ToolRegistry] = None,
        tool_policy: Optional[ToolPolicy] = None,
    ) -> None:
        self.router = router or TaskRouter()
        self.react_backend = react_backend or ReactAgentBackend()
        self.planner_backend = planner_backend or self.react_backend
        if tool_policy is not None:
            self.tool_policy = tool_policy
            self.tool_registry = tool_policy.tool_registry
        else:
            self.tool_registry = tool_registry or build_default_tool_registry(
                agent_conf.get("allowed_tools", [])
            )
            self.tool_policy = ToolPolicy(tool_registry=self.tool_registry)

    @property
    def manages_budget(self) -> bool:
        return bool(getattr(self.react_backend, "manages_budget", False)) and bool(
            getattr(self.planner_backend, "manages_budget", False)
        )

    def prepare(
        self,
        task: AgentTask,
        state: AgentState,
    ) -> TaskRoutingDecision:
        context = self._routing_context(task, state)
        try:
            with trace_recorder.span(
                task.request_id,
                category="router",
                name="select_execution_mode",
            ):
                decision = self._route(task.query, context)
        except Exception as exc:
            decision = TaskRoutingDecision(
                execution_mode="react",
                complexity_score=0,
                reasons=("router_error_fallback",),
                confidence=0.0,
                decision_source="router_error_fallback",
                proposed_mode="react",
            )
            trace_recorder.record_diagnostic_event(
                request_id=task.request_id,
                step_id="execution-router",
                event_type="routing_fallback",
                status="failed",
                latency_ms=0.0,
                failure_reason=str(exc),
            )

        decision = self._guarantee_required_knowledge_lookup(decision)
        metrics_registry.inc_counter(
            "agent_execution_route_total",
            {"mode": decision.execution_mode},
        )
        task.execution_mode = decision.execution_mode
        task.routing_decision = decision
        routing_history = getattr(state, "routing_history", None)
        if routing_history is not None:
            routing_history.append(decision.as_dict())
        self._publish_routing_event(task, decision)
        if decision.transition:
            self._publish_transition_event(task, decision)
        return decision

    @staticmethod
    def _guarantee_required_knowledge_lookup(
        decision: TaskRoutingDecision,
    ) -> TaskRoutingDecision:
        """Use the deterministic RAG handler when routing declares it mandatory."""

        if (
            decision.execution_mode == "plan_execute"
            or decision.required_tools != ("rag_summarize",)
            or len(decision.goals) != 1
            or decision.unavailable_tools
        ):
            return decision
        previous_mode = decision.execution_mode
        return replace(
            decision,
            execution_mode="plan_execute",
            reasons=tuple(
                dict.fromkeys(
                    (*decision.reasons, "required_rag_execution_guarantee")
                )
            ),
            transition=f"{previous_mode}_to_plan_execute",
        )

    def __call__(self, task: AgentTask, state: AgentState) -> AgentBackendResult:
        decision = task.routing_decision
        if decision is None or getattr(state, "routing_feedback", None):
            decision = self.prepare(task, state)
        if decision.execution_mode == "plan_execute":
            return self.planner_backend(task, state)
        return self.react_backend(task, state)

    def repair(
        self,
        task: AgentTask,
        state: AgentState,
        previous: AgentBackendResult,
        verifier_result: VerifyResult,
    ) -> AgentBackendResult:
        """Repair the answer with existing evidence, bypassing routing and tools."""

        repair_task = _targeted_repair_task(task, previous, verifier_result)
        repaired = self.react_backend(repair_task, state)
        return _merge_repair_result(previous, repaired)

    def _route(
        self,
        query: str,
        context: RoutingContext,
    ) -> TaskRoutingDecision:
        route_method = self.router.route
        parameters = inspect.signature(route_method).parameters
        if "context" in parameters:
            return route_method(query, context=context)
        return route_method(query)

    def _routing_context(
        self,
        task: AgentTask,
        state: AgentState,
    ) -> RoutingContext:
        capability_manifest = self.tool_policy.capability_manifest(
            tenant_id=task.tenant_id,
            user_role=task.user_role,
            scene=task.scene,
            alternate_scenes=("report",) if task.scene == "default" else (),
        )
        routing_history = getattr(state, "routing_history", [])
        prior = (
            TaskRoutingDecision.from_dict(routing_history[-1])
            if routing_history
            else None
        )
        memory_snapshot = getattr(state, "memory_snapshot", {})
        recent_messages = tuple(
            str(item)
            for item in memory_snapshot.get("routing_messages", ())
        )
        budget = getattr(state, "budget", None)
        manager = getattr(budget, "manager", None)
        if manager is None:
            return RoutingContext(
                request_id=task.request_id,
                tenant_id=task.tenant_id,
                user_role=task.user_role,
                scene=task.scene,
                available_tools=tuple(
                    str(item["name"]) for item in capability_manifest
                ),
                tool_manifest=tuple(capability_manifest),
                recent_messages=recent_messages,
                prior_decision=prior,
                verification_feedback=getattr(state, "routing_feedback", None),
            )
        return RoutingContext(
            request_id=task.request_id,
            tenant_id=task.tenant_id,
            user_role=task.user_role,
            scene=task.scene,
            available_tools=tuple(str(item["name"]) for item in capability_manifest),
            tool_manifest=tuple(capability_manifest),
            recent_messages=recent_messages,
            remaining_steps=max(0, manager.max_steps - manager.used_steps),
            remaining_tool_calls=manager.remaining_tool_calls,
            remaining_tokens=manager.remaining_tokens,
            prior_decision=prior,
            verification_feedback=getattr(state, "routing_feedback", None) or None,
            budget_manager=manager,
        )

    @staticmethod
    def _publish_routing_event(
        task: AgentTask,
        decision: TaskRoutingDecision,
    ) -> None:
        if not task.emit_events or not event_bus.exists(task.request_id):
            return
        try:
            event_bus.publish(
                task.request_id,
                "routing_completed",
                {
                    "execution_mode": decision.execution_mode,
                    "complexity_score": decision.complexity_score,
                    "reasons": list(decision.reasons),
                    "goals": [goal.as_dict() for goal in decision.goals],
                    "required_tools": list(decision.required_tools),
                    "unavailable_tools": list(decision.unavailable_tools),
                    "risk": decision.risk,
                    "confidence": decision.confidence,
                    "decision_source": decision.decision_source,
                    "proposed_mode": decision.proposed_mode,
                    "transition": decision.transition,
                    "router_version": "semantic-governed-v2",
                },
            )
        except (EventBackpressureError, RuntimeError):
            return

    @staticmethod
    def _publish_transition_event(
        task: AgentTask,
        decision: TaskRoutingDecision,
    ) -> None:
        if not task.emit_events or not event_bus.exists(task.request_id):
            return
        try:
            event_bus.publish(
                task.request_id,
                "routing_transition",
                {
                    "transition": decision.transition,
                    "from_mode": decision.proposed_mode,
                    "to_mode": decision.execution_mode,
                    "reasons": list(decision.reasons),
                    "decision_source": decision.decision_source,
                },
            )
        except (EventBackpressureError, RuntimeError):
            return


class AgentRunner:
    REPORT_KEYWORDS = ("报告", "使用记录", "月报", "总结")

    def __init__(
        self,
        backend: Optional[Callable[[AgentTask, AgentState], AgentBackendResult]] = None,
        policy: Optional[ToolPolicy] = None,
        approval_store: Optional[ApprovalStore] = None,
        artifact_store: Optional[ArtifactStore] = None,
        conversation_memory: Optional[ConversationMemory] = None,
        verifier: Optional[AnswerVerifier] = None,
        max_steps: int = DEFAULT_MAX_STEPS,
        max_tool_calls: int = DEFAULT_MAX_TOOL_CALLS,
        max_tokens: int = DEFAULT_MAX_TOKENS,
        max_cost: float = DEFAULT_MAX_COST,
        max_model_output_tokens: Optional[int] = None,
        max_duration_seconds: Optional[float] = None,
        max_verification_retries: int = DEFAULT_MAX_VERIFICATION_RETRIES,
        estimated_cost_per_1k_tokens: float = float(
            os.getenv("AGENT_ESTIMATED_COST_PER_1K_TOKENS", "0.001")
        ),
    ) -> None:
        self.backend = backend or ReactAgentBackend()
        self.policy = policy or ToolPolicy()
        self.approval_store = approval_store or create_approval_store()
        self.artifact_store = artifact_store or create_artifact_store()
        self.conversation_memory = conversation_memory
        self.verifier = verifier or build_default_answer_verifier()
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.max_tokens = max_tokens
        self.max_cost = max_cost
        self.max_model_output_tokens = (
            DEFAULT_MAX_MODEL_OUTPUT_TOKENS
            if max_model_output_tokens is None
            else max_model_output_tokens
        )
        self.max_duration_seconds = max_duration_seconds
        self.max_verification_retries = max_verification_retries
        self.estimated_cost_per_1k_tokens = estimated_cost_per_1k_tokens

    def run(self, task: AgentTask) -> AgentRunResult:
        self._ensure_trace(task.request_id, task.session_id)
        scene = self._resolve_scene(task)
        task.scene = scene
        state = AgentState(
            request_id=task.request_id,
            session_id=task.session_id,
            tenant_id=task.tenant_id,
            user_id=task.user_id,
            user_goal=task.query,
            user_role=task.user_role,
            scene=scene,
            budget=Budget(
                max_steps=self.max_steps,
                max_tool_calls=self.max_tool_calls,
                max_tokens=self.max_tokens,
                max_cost=self.max_cost,
                deadline_seconds=self.max_duration_seconds,
            ),
        )
        self._publish_event(
            task.request_id,
            "run_started",
            {
                "session_id": task.session_id,
                "scene": scene,
                "max_steps": state.budget.max_steps,
                "max_tool_calls": state.budget.max_tool_calls,
                "max_tokens": state.budget.max_tokens,
            },
        )
        try:
            assert_safe_user_input(task.query)
        except UnsafeInputError as exc:
            refusal = f"请求未执行：{exc}"
            state.mark_rejected("unsafe_input", refusal)
            self._record_diagnostic(state, "security", "rejected", failure_reason=str(exc))
            return self._result(state, refusal)

        if self.conversation_memory is not None and task.user_id:
            with bind_budget_manager(state.budget.manager), bind_request_context(
                request_id=task.request_id,
                session_id=task.session_id,
                tenant_id=task.tenant_id,
            ):
                memory_command = self.conversation_memory.handle_explicit_command(
                    tenant_id=task.tenant_id,
                    user_id=task.user_id,
                    session_id=task.session_id,
                    request_id=task.request_id,
                    user_message=task.query,
                )
            if memory_command.handled:
                metrics_registry.inc_counter(
                    "agent_memory_operation_total",
                    {
                        "action": memory_command.action or "unknown",
                        "status": memory_command.status,
                    },
                )
                answer = self.conversation_memory.apply_response_policies(
                    memory_command.message,
                    tenant_id=task.tenant_id,
                    user_id=task.user_id,
                )
                self._publish_event(
                    task.request_id,
                    "memory_operation_completed",
                    {
                        "status": memory_command.status,
                        "action": memory_command.action,
                        "memory_ids": [
                            memory.memory_id for memory in memory_command.records
                        ],
                        "saved_keys": [
                            memory.key for memory in memory_command.records
                        ],
                        "deleted": memory_command.deleted,
                        "rejected_reason": memory_command.rejected_reason,
                    },
                )
                artifact = self._save_artifact(
                    state,
                    artifact_type="memory_operation",
                    name="memory-operation",
                    payload={
                        "status": memory_command.status,
                        "action": memory_command.action,
                        "memory_ids": [
                            memory.memory_id for memory in memory_command.records
                        ],
                        "saved_keys": [
                            memory.key for memory in memory_command.records
                        ],
                        "deleted": memory_command.deleted,
                        "rejected_reason": memory_command.rejected_reason,
                    },
                )
                state.add_artifact(artifact)
                state.mark_completed(answer)
                self._record_diagnostic(
                    state,
                    "memory",
                    memory_command.status,
                    failure_reason=memory_command.rejected_reason,
                )
                return self._result(state, answer)

        if self.conversation_memory is not None:
            try:
                recent_messages = self.conversation_memory.get_messages(
                    task.session_id,
                    tenant_id=task.tenant_id,
                    token_budget=900,
                    user_id=task.user_id,
                )
                state.memory_snapshot["routing_messages"] = [
                    f"{message.get('role', 'unknown')}: "
                    f"{str(message.get('content') or '')[:500]}"
                    for message in recent_messages[-6:]
                ]
            except Exception:
                state.memory_snapshot["routing_messages"] = []

        initial_budget_error = self._backend_preflight_reason(state, task.query)
        if initial_budget_error is not None:
            state.mark_blocked(initial_budget_error)
            self._record_diagnostic(state, "budget", "blocked")
            return self._result(state, state.error or "budget exhausted")

        prepare_backend = getattr(self.backend, "prepare", None)
        if callable(prepare_backend):
            decision = prepare_backend(task, state)
            if (
                task.scene == "default"
                and "fetch_external_data" in decision.required_tools
            ):
                task.scene = "report"
                state.scene = "report"
                self._publish_event(
                    task.request_id,
                    "execution_context_updated",
                    {
                        "scene": "report",
                        "reason": "semantic_report_capability_required",
                    },
                )

        approval_result = self._handle_sensitive_report_data(task, state)
        if approval_result is not None:
            return approval_result

        answer = ""
        tool_results: List[Dict[str, Any]] = []
        verifier_result: Optional[VerifyResult] = None
        previous_backend_result: Optional[AgentBackendResult] = None
        for attempt in range(self.max_verification_retries + 1):
            repair_backend = getattr(self.backend, "repair", None)
            if attempt > 0 and not callable(repair_backend):
                refusal = (
                    "请求未执行：回答未通过证据校验，且当前后端不支持基于已有证据的"
                    "定向修复。已停止重新执行完整任务，避免重复调用。"
                )
                state.mark_rejected("verification_repair_unavailable", refusal)
                self._record_diagnostic(
                    state,
                    "verification_repair",
                    "rejected",
                    retry=attempt,
                    failure_reason="targeted_repair_unavailable",
                    verifier=verifier_result.__dict__ if verifier_result else {},
                )
                return self._result(state, refusal, verifier_result)
            budget_error = (
                self._retry_preflight_reason(state, task)
                if attempt > 0
                else None
            ) or self._backend_preflight_reason(state, task.query)
            if budget_error is not None:
                if attempt > 0 and previous_backend_result is not None:
                    return self._finish_budget_limited_retry(
                        task=task,
                        state=state,
                        backend_result=previous_backend_result,
                        previous_verifier=verifier_result,
                        reason=budget_error,
                    )
                state.mark_blocked(budget_error)
                self._record_diagnostic(
                    state,
                    "budget",
                    "blocked",
                    retry=attempt,
                    failure_reason=budget_error,
                )
                return self._result(state, state.error or "budget exhausted", verifier_result)
            step = state.record_step(
                step_type="backend",
                name=(
                    "repair_answer_from_verified_evidence"
                    if attempt > 0
                    else "execute_agent_backend"
                ),
                status="running",
                metadata={
                    "attempt": attempt,
                    "strategy": "targeted_evidence_repair" if attempt > 0 else "initial",
                },
            )
            model_reservation: Reservation | None = None
            attempt_previous = previous_backend_result
            try:
                if not self._backend_manages_budget():
                    model_reservation = self._reserve_backend_model_call(state, task.query)
                self._publish_event(
                    task.request_id,
                    "model_started",
                    {
                        "attempt": attempt,
                        "max_output_tokens": self.max_model_output_tokens,
                        "strategy": (
                            "targeted_evidence_repair" if attempt > 0 else "initial"
                        ),
                    },
                )
                if attempt > 0:
                    assert attempt_previous is not None
                    assert verifier_result is not None
                    backend_result = repair_backend(
                        task,
                        state,
                        attempt_previous,
                        verifier_result,
                    )
                else:
                    backend_result = self.backend(task, state)
            except BudgetExceeded as exc:
                if model_reservation is not None:
                    state.budget.manager.release_model_call(model_reservation)
                if attempt > 0 and previous_backend_result is not None:
                    return self._finish_budget_limited_retry(
                        task=task,
                        state=state,
                        backend_result=previous_backend_result,
                        previous_verifier=verifier_result,
                        reason=exc.reason,
                    )
                state.mark_blocked(exc.reason)
                self._record_diagnostic(
                    state,
                    "budget",
                    "blocked",
                    step_id=step.step_id,
                    retry=attempt,
                    failure_reason=exc.reason,
                )
                return self._result(state, state.error or "budget exhausted", verifier_result)
            except Exception as exc:
                if model_reservation is not None:
                    state.budget.manager.release_model_call(model_reservation)
                state.mark_failed(str(exc))
                self._record_diagnostic(
                    state, "backend", "failed", step_id=step.step_id,
                    failure_reason=str(exc),
                )
                return self._result(state, f"请求未执行：{exc}")

            previous_backend_result = backend_result
            prior_evidence_keys = {
                (
                    str(item.get("id") or item.get("source") or ""),
                    str(item.get("content") or ""),
                )
                for item in (attempt_previous.evidence if attempt_previous else [])
            }
            for item in backend_result.evidence:
                evidence_key = (
                    str(item.get("id") or item.get("source") or ""),
                    str(item.get("content") or ""),
                )
                if evidence_key in prior_evidence_keys:
                    continue
                state.add_observation(
                    Observation(
                        source=str(item.get("id") or item.get("source") or "evidence"),
                        content=str(item.get("content", "")),
                        metadata=item,
                    )
                )
            answer = backend_result.answer
            tool_results = backend_result.tool_results
            tokens_in, tokens_out, cost, cost_mode = self._usage_for_result(
                task.query,
                answer,
                backend_result,
            )
            if model_reservation is not None:
                state.budget.manager.commit_model_call(
                    model_reservation,
                    actual_tokens=tokens_in + tokens_out,
                    actual_cost=cost,
                )
            elif not backend_result.budget_accounted:
                # Compatibility for an opt-in backend that did not account a
                # result itself. Production ReactAgent responses take the
                # middleware path and never reach this fallback.
                state.budget.record_tokens(tokens_in + tokens_out)
                state.budget.record_cost(cost)
            self._publish_event(
                task.request_id,
                "model_completed",
                {
                    "attempt": attempt,
                    "model_name": backend_result.model_name,
                    "tokens_in": tokens_in,
                    "tokens_out": tokens_out,
                    "cost": cost,
                    "cost_mode": cost_mode,
                    "tool_result_count": len(backend_result.tool_results),
                    "evidence_count": len(backend_result.evidence),
                    "remaining_tokens": state.budget.manager.remaining_tokens,
                },
            )
            prior_tool_keys = {
                (
                    str(item.get("tool") or ""),
                    str(item.get("content") or ""),
                    str(item.get("status") or ""),
                )
                for item in (attempt_previous.tool_results if attempt_previous else [])
            }
            for tool_result in backend_result.tool_results:
                tool_key = (
                    str(tool_result.get("tool") or ""),
                    str(tool_result.get("content") or ""),
                    str(tool_result.get("status") or ""),
                )
                if tool_key in prior_tool_keys:
                    continue
                if tool_result.get("record_type") == "planner_step":
                    continue
                tool_name = str(tool_result.get("tool", "unknown"))
                if not backend_result.budget_accounted:
                    try:
                        tool_reservation = state.budget.manager.reserve_tool_call(tool_name)
                    except BudgetExceeded as exc:
                        state.mark_blocked(exc.reason)
                        self._record_diagnostic(
                            state,
                            "budget",
                            "blocked",
                            step_id=step.step_id,
                            tool=tool_name,
                            failure_reason=exc.reason,
                        )
                        return self._result(
                            state,
                            state.error or "budget exhausted",
                            verifier_result,
                        )
                    state.budget.manager.commit_tool_call(tool_reservation)
                state.add_tool_call(
                    ToolCallRecord(
                        tool_name=tool_name,
                        args=dict(
                            tool_result.get("args")
                            or (tool_result.get("metadata") or {}).get("redacted_args")
                            or {}
                        ),
                        status=str(tool_result.get("status", "completed")),
                        result=str(tool_result.get("content", ""))[:500],
                    ),
                    count_budget=False,
                )

            usage_error = self._usage_overrun_reason(state)
            if usage_error is not None:
                state.mark_blocked(usage_error)
                self._record_diagnostic(
                    state,
                    "budget",
                    "blocked",
                    step_id=step.step_id,
                    model_name=backend_result.model_name,
                    tokens_in=tokens_in,
                    tokens_out=tokens_out,
                    cost=cost,
                    cost_mode=cost_mode,
                    failure_reason=usage_error,
                )
                return self._result(state, state.error or "budget exhausted", verifier_result)

            self._publish_event(
                task.request_id,
                "verification_started",
                {"attempt": attempt, "evidence_count": len(backend_result.evidence)},
            )
            knowledge_gap = self._knowledge_gap_for_backend(task, backend_result)
            if knowledge_gap is not None:
                answer, gap_reason = knowledge_gap
                verifier_result = VerifyResult(
                    passed=True,
                    action="pass",
                    score=None,
                    reasons=[gap_reason],
                    judge={
                        "status": "not_evaluated",
                        "reason": "transparent_knowledge_gap",
                    },
                )
                self._publish_event(
                    task.request_id,
                    "verification_completed",
                    {
                        "attempt": attempt,
                        "passed": True,
                        "action": "pass",
                        "score": None,
                        "reasons": [gap_reason],
                        "citation_validity": 1.0,
                        "citation_coverage": 1.0,
                        "unsupported_claim_rate": 0.0,
                        "harmful_instruction": False,
                        "strategy": "knowledge_gap",
                    },
                )
                self._record_diagnostic(
                    state,
                    "knowledge_gap",
                    "completed",
                    step_id=step.step_id,
                    tool="rag_summarize",
                    verifier=verifier_result.__dict__,
                    retry=attempt,
                )
                self._publish_event(
                    task.request_id,
                    "execution_degraded",
                    {
                        "status": "completed",
                        "reason": gap_reason,
                        "strategy": "knowledge_gap",
                    },
                )
                state.routing_feedback = {}
                break
            verifier_result = self.verifier.verify(
                query=task.query,
                answer=answer,
                evidence=backend_result.evidence,
                scene=scene,
                tool_results=backend_result.tool_results,
                artifacts=[artifact.__dict__ for artifact in state.artifacts],
                structured_answer=backend_result.structured_answer,
                required_tools=self._verification_required_tools(task),
            )
            self._publish_event(
                task.request_id,
                "verification_completed",
                {
                    "attempt": attempt,
                    "passed": verifier_result.passed,
                    "action": verifier_result.action,
                    "score": verifier_result.score,
                    "reasons": verifier_result.reasons,
                    "citation_validity": verifier_result.citation_validity,
                    "citation_coverage": verifier_result.citation_coverage,
                    "unsupported_claim_rate": verifier_result.unsupported_claim_rate,
                    "harmful_instruction": verifier_result.harmful_instruction,
                },
            )
            self._record_diagnostic(
                state,
                "verifier",
                "ok" if verifier_result.passed else "failed",
                step_id=step.step_id,
                evidence_ids=[obs.source for obs in state.observations],
                verifier=verifier_result.__dict__,
                model_name=backend_result.model_name,
                retry=attempt,
                tokens_in=tokens_in,
                tokens_out=tokens_out,
                cost=cost,
                cost_mode=cost_mode,
            )
            if verifier_result.passed:
                state.routing_feedback = {}
                break
            missing_knowledge_lookup = self._required_knowledge_lookup_missing(
                backend_result,
                verifier_result,
            )
            fallback_answer = backend_result.safe_fallback_answer.strip()
            if fallback_answer and fallback_answer != answer:
                fallback_verifier = self.verifier.verify(
                    query=task.query,
                    answer=fallback_answer,
                    evidence=backend_result.evidence,
                    scene=scene,
                    tool_results=backend_result.tool_results,
                    artifacts=[artifact.__dict__ for artifact in state.artifacts],
                    structured_answer=backend_result.safe_fallback_structured_answer,
                    required_tools=self._verification_required_tools(task),
                )
                if fallback_verifier.passed:
                    answer = fallback_answer
                    verifier_result = fallback_verifier
                    state.routing_feedback = {}
                    self._publish_event(
                        task.request_id,
                        "verification_completed",
                        {
                            "attempt": attempt,
                            "passed": True,
                            "action": fallback_verifier.action,
                            "score": fallback_verifier.score,
                            "reasons": fallback_verifier.reasons,
                            "citation_validity": fallback_verifier.citation_validity,
                            "citation_coverage": fallback_verifier.citation_coverage,
                            "unsupported_claim_rate": (
                                fallback_verifier.unsupported_claim_rate
                            ),
                            "strategy": backend_result.safe_fallback_strategy,
                        },
                    )
                    self._record_diagnostic(
                        state,
                        "verifier",
                        "ok",
                        step_id=step.step_id,
                        evidence_ids=[obs.source for obs in state.observations],
                        verifier=fallback_verifier.__dict__,
                        model_name=backend_result.model_name,
                        retry=attempt,
                    )
                    self._publish_event(
                        task.request_id,
                        "execution_degraded",
                        {
                            "status": "completed",
                            "reason": "generated_answer_verification_failed",
                            "strategy": backend_result.safe_fallback_strategy,
                        },
                    )
                    break
            if (
                missing_knowledge_lookup
                or verifier_result.action != "retry"
                or attempt >= self.max_verification_retries
            ):
                refusal = self._verification_refusal_answer(
                    task,
                    missing_knowledge_lookup=missing_knowledge_lookup,
                )
                state.mark_rejected("verification_failed", refusal)
                artifact = self._save_artifact(
                    state,
                    artifact_type="verification_failure",
                    name="verifier-result",
                    payload={"answer": answer, "verifier": verifier_result.__dict__},
                )
                state.add_artifact(artifact)
                self._publish_event(task.request_id, "artifact_created", artifact.__dict__)
                return self._result(state, refusal, verifier_result)
            state.routing_feedback = {
                "action": verifier_result.action,
                "reasons": list(verifier_result.reasons),
                "citation_validity": verifier_result.citation_validity,
                "citation_coverage": verifier_result.citation_coverage,
                "unsupported_claim_rate": verifier_result.unsupported_claim_rate,
            }

        if self.conversation_memory is not None:
            answer = self.conversation_memory.apply_response_policies(
                answer,
                tenant_id=task.tenant_id,
                user_id=task.user_id,
            )

        if answer and bool(getattr(self.backend, "defers_answer_tokens", False)):
            self._publish_event(
                task.request_id,
                "token_delta",
                {"delta": answer, "provisional": False, "replace": True},
            )

        artifact = self._save_artifact(
            state,
            artifact_type="answer",
            name="final-answer",
            payload={
                "answer": answer,
                "evidence": [obs.metadata for obs in state.observations],
                "tool_results": tool_results,
            },
            metadata={"scene": scene},
        )
        state.add_artifact(artifact)
        self._publish_event(task.request_id, "artifact_created", artifact.__dict__)
        state.mark_completed(answer)
        self._record_diagnostic(state, "runner", "completed")
        return self._result(state, answer, verifier_result)

    async def run_stream(
        self,
        task: AgentTask,
        *,
        last_event_id: int = 0,
        heartbeat_seconds: float = 10.0,
        timeout_seconds: Optional[float] = None,
    ):
        """Yield live, sequenced events; reconnects replay without re-running."""
        if last_event_id < 0:
            raise ValueError("last_event_id must be non-negative")
        request_id = task.request_id
        task.emit_events = True
        owns_producer = event_bus.open(request_id, identity=self.stream_identity(task))
        producer_task = None
        if owns_producer:
            async def produce() -> None:
                try:
                    await asyncio.to_thread(self.run, task)
                except Exception as exc:  # pragma: no cover - defensive boundary
                    self._publish_event(
                        request_id,
                        "run_failed",
                        {"status": "failed", "error": str(exc)},
                    )
                finally:
                    event_bus.close(request_id)

            producer_task = asyncio.create_task(produce())

        cursor = last_event_id
        for item in event_bus.replay(request_id, after_sequence=cursor):
            cursor = item.sequence
            yield item
        if event_bus.is_closed(request_id):
            return

        effective_timeout = (
            timeout_seconds
            if timeout_seconds is not None
            else self.max_duration_seconds or 120.0
        )
        started = time.monotonic()
        last_heartbeat = started
        timed_out = False
        try:
            while True:
                now = time.monotonic()
                if (
                    not timed_out
                    and effective_timeout > 0
                    and now - started >= effective_timeout
                ):
                    timed_out = True
                    event_bus.cancel(request_id)
                    self._publish_event(
                        request_id,
                        "run_failed",
                        {"status": "failed", "error": "request_timeout"},
                    )
                    event_bus.close(request_id)
                poll_timeout = 0.5
                if heartbeat_seconds > 0:
                    poll_timeout = min(
                        poll_timeout,
                        max(0.01, heartbeat_seconds - (now - last_heartbeat)),
                    )
                if effective_timeout > 0 and not timed_out:
                    poll_timeout = min(
                        poll_timeout,
                        max(0.01, effective_timeout - (now - started)),
                    )
                item = await asyncio.to_thread(
                    event_bus.consume,
                    request_id,
                    poll_timeout,
                    cursor,
                )
                if item == "closed":
                    break
                if item is not None:
                    if item.sequence > cursor:
                        cursor = item.sequence
                        yield item
                    continue
                now = time.monotonic()
                if heartbeat_seconds > 0 and now - last_heartbeat >= heartbeat_seconds:
                    self._publish_event(
                        request_id,
                        "heartbeat",
                        {"elapsed_ms": round((now - started) * 1000, 1)},
                    )
                    last_heartbeat = now
        except asyncio.CancelledError:
            event_bus.cancel(request_id)
            raise
        finally:
            if producer_task is not None and producer_task.done():
                await producer_task

    @staticmethod
    def stream_identity(task: AgentTask) -> Dict[str, str]:
        return {
            "tenant_id": task.tenant_id,
            "user_id": task.user_id or "",
            "session_id": task.session_id,
            "query_sha256": hashlib.sha256(task.query.encode("utf-8")).hexdigest(),
        }

    def _handle_sensitive_report_data(
        self,
        task: AgentTask,
        state: AgentState,
    ) -> Optional[AgentRunResult]:
        if not self._needs_report_data(task, state.scene):
            return None
        args = self._report_access_args(task)
        decision = self.policy.decide(
            tenant_id=task.tenant_id,
            principal_id=task.user_id,
            data_user_id=task.data_user_id,
            user_role=task.user_role,
            scene=state.scene,
            tool_name="fetch_external_data",
            args=args,
        )
        if decision.action == PolicyAction.ALLOW:
            state.add_tool_call(
                ToolCallRecord(
                    tool_name="fetch_external_data",
                    args=args,
                    status="approved",
                    risk_level="medium",
                ),
                count_budget=False,
            )
            return None
        if decision.action == PolicyAction.DENY:
            state.add_tool_call(
                ToolCallRecord(
                    tool_name="fetch_external_data",
                    args=args,
                    status="denied",
                    error=decision.reason,
                    risk_level="medium",
                ),
                count_budget=False,
            )
            state.mark_rejected(decision.reason, "请求未执行：当前场景无权读取使用记录。")
            self._record_diagnostic(state, "policy", "denied", failure_reason=decision.reason)
            return self._result(state, state.final_answer or "", None)
        if decision.action == PolicyAction.NEED_APPROVAL:
            if task.approval_id:
                approval = self.approval_store.get(task.approval_id)
                if approval.tenant_id != task.tenant_id or approval.tool_name != "fetch_external_data":
                    state.mark_rejected(
                        "approval_mismatch",
                        "请求未执行：审批记录与当前租户或工具不匹配。",
                    )
                    return self._result(state, state.final_answer or "", None)
                if approval.principal_id and approval.principal_id != (task.user_id or ""):
                    state.mark_rejected(
                        "approval_principal_mismatch",
                        "请求未执行：审批记录不属于当前用户。",
                    )
                    return self._result(state, state.final_answer or "", None)
                if approval.args != args:
                    state.mark_rejected(
                        "approval_arguments_mismatch",
                        "请求未执行：审批记录与当前数据范围不匹配。",
                    )
                    return self._result(state, state.final_answer or "", None)
                if approval.is_approved:
                    state.add_tool_call(
                        ToolCallRecord(
                            tool_name="fetch_external_data",
                            args=approval.args,
                            status="approved",
                            approval_id=approval.approval_id,
                            risk_level="medium",
                        ),
                        count_budget=False,
                    )
                    return None
                if approval.is_denied:
                    state.mark_rejected(
                        "approval_denied",
                        "请求未执行：敏感工具调用审批被拒绝。",
                    )
                    return self._result(state, state.final_answer or "", None)
                state.mark_pending_approval(approval.approval_id)
                self._publish_event(
                    task.request_id,
                    "approval_required",
                    {"tool": "fetch_external_data", "approval_id": approval.approval_id},
                )
                return self._result(
                    state,
                    "请求已暂停：等待敏感工具调用审批。",
                    approval_id=approval.approval_id,
                )

            approval = self.approval_store.create_pending(
                request_id=task.request_id,
                tenant_id=task.tenant_id,
                user_role=task.user_role,
                tool_name="fetch_external_data",
                args=args,
                reason=decision.reason,
                principal_id=task.user_id or "",
            )
            state.add_tool_call(
                ToolCallRecord(
                    tool_name="fetch_external_data",
                    args=args,
                    status="pending_approval",
                    approval_id=approval.approval_id,
                    risk_level="medium",
                ),
                count_budget=False,
            )
            state.mark_pending_approval(approval.approval_id)
            self._publish_event(
                task.request_id,
                "approval_required",
                {"tool": "fetch_external_data", "approval_id": approval.approval_id},
            )
            self._record_diagnostic(
                state,
                "approval",
                "pending",
                tool="fetch_external_data",
                failure_reason=decision.reason,
            )
            return self._result(
                state,
                "请求已暂停：等待敏感工具调用审批。",
                approval_id=approval.approval_id,
            )
        state.mark_rejected(decision.reason, "请求未执行：工具参数需要先脱敏。")
        return self._result(state, state.final_answer or "", None)

    def _save_artifact(
        self,
        state: AgentState,
        artifact_type: str,
        name: str,
        payload: Dict[str, Any],
        metadata: Optional[Dict[str, Any]] = None,
    ) -> ArtifactRef:
        artifact = self.artifact_store.save_artifact(
            request_id=state.request_id,
            tenant_id=state.tenant_id,
            artifact_type=artifact_type,
            name=name,
            payload=payload,
            metadata=metadata,
        )
        return ArtifactRef(
            artifact_id=artifact.artifact_id,
            type=artifact.artifact_type,
            name=artifact.name,
        )

    def _result(
        self,
        state: AgentState,
        answer: str,
        verifier: Optional[VerifyResult] = None,
        approval_id: Optional[str] = None,
    ) -> AgentRunResult:
        if self.conversation_memory is not None:
            with bind_budget_manager(state.budget.manager), bind_request_context(
                request_id=state.request_id,
                session_id=state.session_id,
                tenant_id=state.tenant_id,
            ):
                self.conversation_memory.commit_turn(
                    session_id=state.session_id,
                    request_id=state.request_id,
                    user_message=state.user_goal,
                    assistant_message=answer,
                    status=state.status,
                    tenant_id=state.tenant_id,
                    user_id=state.user_id,
                )
        public_evidence = _public_citation_evidence(state.observations, answer)
        result = AgentRunResult(
            state=state,
            answer=answer,
            request_id=state.request_id,
            approval_id=approval_id or state.approval_id,
            artifacts=list(state.artifacts),
            verifier=verifier,
            evidence=public_evidence,
        )
        event_type = (
            "run_completed"
            if state.status in {"completed", "pending_approval"}
            else "run_failed"
        )
        self._publish_event(
            state.request_id,
            event_type,
            {
                "status": state.status,
                "answer": answer,
                "approval_id": result.approval_id,
                "artifacts": [artifact.__dict__ for artifact in result.artifacts],
                "verifier": verifier.__dict__ if verifier else None,
                "evidence": public_evidence,
                "error": state.error,
                "budget": state.budget.manager.snapshot(),
            },
        )
        return result

    @staticmethod
    def _publish_event(request_id: str, event_type: str, payload: Dict[str, Any]) -> None:
        if not event_bus.exists(request_id):
            return
        try:
            event_bus.publish(request_id, event_type, payload)
        except (EventBackpressureError, RuntimeError):
            # Client disconnects and closed channels must not turn a finished run into 500.
            return

    def _resolve_scene(self, task: AgentTask) -> str:
        if task.scene != "default":
            return task.scene
        return "report" if self._needs_report_data(task, task.scene) else task.scene

    def _needs_report_data(self, task: AgentTask, scene: str) -> bool:
        return scene == "report" or any(keyword in task.query for keyword in self.REPORT_KEYWORDS)

    @staticmethod
    def _report_access_args(task: AgentTask) -> Dict[str, str]:
        user_match = re.search(
            r"(?:用户(?:\s*ID)?|user(?:\s*id)?)\s*[：:=#-]?\s*"
            r"([A-Za-z0-9][A-Za-z0-9_-]{0,63})",
            task.query,
            flags=re.IGNORECASE,
        )
        month_match = re.search(
            r"(?<!\d)(20[0-9]{2}-(?:0[1-9]|1[0-2]))(?!\d)",
            task.query,
        )
        if month_match is None:
            chinese_month = re.search(
                r"(?<!\d)(20[0-9]{2})年(1[0-2]|0?[1-9])月",
                task.query,
            )
            month = (
                f"{chinese_month.group(1)}-{int(chinese_month.group(2)):02d}"
                if chinese_month
                else str(agent_conf.get("current_month") or "")
            )
        else:
            month = month_match.group(1)
        if not month:
            month = time.strftime("%Y-%m")
        return {
            "user_id": (
                user_match.group(1)
                if user_match
                else str(task.data_user_id or task.user_id or "")
            ),
            "month": month,
        }

    def _ensure_trace(self, request_id: str, session_id: str) -> None:
        try:
            trace_recorder.export_trace(request_id)
        except KeyError:
            trace_recorder.start_trace(request_id=request_id, session_id=session_id)

    def _record_diagnostic(
        self,
        state: AgentState,
        event_type: str,
        status: str,
        step_id: Optional[str] = None,
        tool: Optional[str] = None,
        failure_reason: Optional[str] = None,
        evidence_ids: Optional[List[str]] = None,
        verifier: Optional[Dict[str, Any]] = None,
        retry: int = 0,
        model_name: str = "",
        tokens_in: int = 0,
        tokens_out: int = 0,
        cost: float = 0.0,
        cost_mode: str = "estimated",
    ) -> None:
        trace_recorder.record_diagnostic_event(
            request_id=state.request_id,
            step_id=step_id or f"step-{state.current_step}",
            event_type=event_type,
            status=status,
            latency_ms=0.0,
            tool=tool,
            evidence_ids=evidence_ids or [],
            verifier=verifier or {},
            retry=retry,
            prompt_version="harness:v1",
            model_name=model_name,
            failure_reason=failure_reason,
            tokens_in=tokens_in,
            tokens_out=tokens_out,
            cost=cost,
            cost_mode=cost_mode,
        )

    def _usage_for_result(
        self,
        query: str,
        answer: str,
        backend_result: AgentBackendResult,
    ) -> tuple[int, int, float, str]:
        if backend_result.tokens_in or backend_result.tokens_out or backend_result.cost:
            return (
                backend_result.tokens_in,
                backend_result.tokens_out,
                backend_result.cost,
                backend_result.cost_mode or "actual",
            )
        tokens_in, tokens_out, cost = self._estimate_usage(query, answer)
        return tokens_in, tokens_out, cost, "estimated"

    def _backend_manages_budget(self) -> bool:
        return bool(getattr(self.backend, "manages_budget", False))

    def _retry_preflight_reason(
        self,
        state: AgentState,
        task: AgentTask,
    ) -> Optional[str]:
        manager = state.budget.manager
        minimum_tokens = DEFAULT_MIN_REPAIR_TOKENS
        if manager.remaining_tokens < minimum_tokens:
            return "retry_token_budget_insufficient"
        return None

    def _finish_budget_limited_retry(
        self,
        *,
        task: AgentTask,
        state: AgentState,
        backend_result: AgentBackendResult,
        previous_verifier: Optional[VerifyResult],
        reason: str,
    ) -> AgentRunResult:
        fallback_answer = backend_result.safe_fallback_answer.strip()
        fallback_verifier: Optional[VerifyResult] = None
        if fallback_answer:
            fallback_verifier = self.verifier.verify(
                query=task.query,
                answer=fallback_answer,
                evidence=backend_result.evidence,
                scene=state.scene,
                tool_results=backend_result.tool_results,
                artifacts=[artifact.__dict__ for artifact in state.artifacts],
                structured_answer=backend_result.safe_fallback_structured_answer,
                required_tools=self._verification_required_tools(task),
            )
        if not fallback_answer or fallback_verifier is None or not fallback_verifier.passed:
            refusal = (
                "请求未执行：现有综合回答未通过证据校验，且剩余预算不足以安全重试。"
                "已停止继续生成，避免输出未经验证的结论。"
            )
            state.mark_rejected("verification_retry_budget_insufficient", refusal)
            self._publish_event(
                task.request_id,
                "execution_degraded",
                {
                    "status": "rejected",
                    "reason": reason,
                    "strategy": "stop_unverified_retry",
                },
            )
            self._record_diagnostic(
                state,
                "budget_degraded",
                "rejected",
                retry=1,
                failure_reason=reason,
                verifier=(
                    fallback_verifier.__dict__
                    if fallback_verifier is not None
                    else (previous_verifier.__dict__ if previous_verifier else {})
                ),
            )
            return self._result(
                state,
                refusal,
                fallback_verifier or previous_verifier,
            )

        answer = fallback_answer
        if self.conversation_memory is not None:
            answer = self.conversation_memory.apply_response_policies(
                answer,
                tenant_id=task.tenant_id,
                user_id=task.user_id,
            )
        self._publish_event(
            task.request_id,
            "execution_degraded",
            {
                "status": "completed",
                "reason": reason,
                "strategy": "verified_partial_result",
            },
        )
        if answer and bool(getattr(self.backend, "defers_answer_tokens", False)):
            self._publish_event(
                task.request_id,
                "token_delta",
                {"delta": answer, "provisional": False, "replace": True},
            )
        artifact = self._save_artifact(
            state,
            artifact_type="answer",
            name="final-answer",
            payload={
                "answer": answer,
                "evidence": [obs.metadata for obs in state.observations],
                "tool_results": backend_result.tool_results,
            },
            metadata={
                "scene": state.scene,
                "degraded": True,
                "degraded_reason": reason,
            },
        )
        state.add_artifact(artifact)
        self._publish_event(task.request_id, "artifact_created", artifact.__dict__)
        state.routing_feedback = {}
        state.mark_completed(answer)
        self._record_diagnostic(
            state,
            "budget_degraded",
            "completed",
            retry=1,
            failure_reason=reason,
            evidence_ids=[obs.source for obs in state.observations],
            verifier=fallback_verifier.__dict__,
        )
        return self._result(state, answer, fallback_verifier)

    @staticmethod
    def _verification_required_tools(task: AgentTask) -> tuple[str, ...]:
        routed = (
            task.routing_decision.required_tools
            if task.routing_decision is not None
            else ()
        )
        return tuple(dict.fromkeys((*task.required_tools, *routed)))

    @staticmethod
    def _knowledge_gap_for_backend(
        task: AgentTask,
        backend_result: AgentBackendResult,
    ) -> Optional[tuple[str, str]]:
        """Translate a completed RAG lookup with no usable answer into a clear outcome."""

        hard_reasons = {
            "answer_empty",
            "citation_invalid",
            "citation_placeholder",
            "claim_evidence_id_invalid",
            "evidence_contradiction",
            "harmful_instruction",
        }
        for item in reversed(backend_result.tool_results):
            if str(item.get("tool") or item.get("name") or "") != "rag_summarize":
                continue
            metadata = item.get("metadata")
            metadata = metadata if isinstance(metadata, dict) else {}
            verification = metadata.get("verification")
            verification = verification if isinstance(verification, dict) else {}
            reasons = {
                str(reason)
                for reason in verification.get("reasons") or ()
            }
            status = str(
                metadata.get("business_status") or item.get("status") or ""
            ).lower()
            if status == "empty":
                gap_reason = (
                    "knowledge_irrelevant"
                    if "retrieval_relevance_below_threshold" in reasons
                    else "knowledge_no_results"
                )
                return knowledge_gap_answer(task.query, gap_reason), gap_reason
            content = str(item.get("content") or "").strip()
            if (
                status == "verification_failed"
                and not hard_reasons.intersection(reasons)
                and (
                    not content
                    or content.startswith("请求未执行")
                    or content.startswith("当前知识库")
                )
            ):
                gap_reason = "evidence_insufficient_for_conclusion"
                return knowledge_gap_answer(task.query, gap_reason), gap_reason
            return None
        return None

    @staticmethod
    def _required_knowledge_lookup_missing(
        backend_result: AgentBackendResult,
        verifier_result: VerifyResult,
    ) -> bool:
        if "rag_summarize" not in verifier_result.missing_required_tools:
            return False
        return not any(
            str(item.get("tool") or item.get("name") or "") == "rag_summarize"
            for item in backend_result.tool_results
        )

    @staticmethod
    def _verification_refusal_answer(
        task: AgentTask,
        *,
        missing_knowledge_lookup: bool,
    ) -> str:
        if missing_knowledge_lookup:
            if re.search(r"最贵|价格|售价|价位|多少钱|报价", task.query):
                return (
                    "本次未能完成必要的知识库检索，因此暂时无法确认哪款机器人最贵。"
                    "为避免猜测，我不会给出具体型号；请稍后重试，或提供候选型号及价格。"
                )
            return (
                "本次未能完成必要的知识库检索，因此暂时无法给出可靠结论。"
                "为避免猜测，我不会补充未经资料支持的内容，请稍后重试。"
            )
        return "请求未执行：回答未通过证据校验，已拒绝输出可能不可靠的结论。"

    def _backend_preflight_reason(self, state: AgentState, query: str) -> Optional[str]:
        manager = state.budget.manager
        try:
            manager.check_deadline()
        except BudgetExceeded as exc:
            return exc.reason
        if state.budget.used_steps >= state.budget.max_steps:
            return "max_steps_exceeded"
        input_tokens = max(1, (len(query) + 3) // 4)
        if manager.remaining_output_tokens(input_tokens) <= 0:
            return "max_tokens_exceeded"
        if manager.remaining_cost <= 0:
            return "max_cost_exceeded"
        return None

    def _reserve_backend_model_call(self, state: AgentState, query: str) -> Reservation:
        manager = state.budget.manager
        input_tokens = max(1, (len(query) + 3) // 4)
        output_cap = min(
            self.max_model_output_tokens,
            manager.remaining_output_tokens(input_tokens),
        )
        if self.estimated_cost_per_1k_tokens > 0:
            affordable_total = int(
                (manager.remaining_cost * 1000.0) / self.estimated_cost_per_1k_tokens
            )
            output_cap = min(output_cap, max(0, affordable_total - input_tokens))
        estimated_cost = round(
            ((input_tokens + output_cap) / 1000.0)
            * self.estimated_cost_per_1k_tokens,
            6,
        )
        return manager.reserve_model_call(
            estimated_input_tokens=input_tokens,
            max_output_tokens=output_cap,
            estimated_cost=estimated_cost,
        )

    @staticmethod
    def _usage_overrun_reason(state: AgentState) -> Optional[str]:
        snapshot = state.budget.manager.snapshot()
        if int(snapshot["used_tokens"]) > int(snapshot["max_tokens"]):
            return "max_tokens_exceeded"
        if float(snapshot["used_cost"]) > float(snapshot["max_cost"]):
            return "max_cost_exceeded"
        return None

    def _estimate_usage(self, query: str, answer: str) -> tuple[int, int, float]:
        tokens_in = max(1, (len(query) + 3) // 4)
        tokens_out = max(1, (len(answer) + 3) // 4)
        cost = round(
            ((tokens_in + tokens_out) / 1000.0) * self.estimated_cost_per_1k_tokens,
            6,
        )
        return tokens_in, tokens_out, cost
