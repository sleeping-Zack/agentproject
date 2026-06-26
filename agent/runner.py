from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional
from uuid import uuid4

from agent.policies import PolicyAction, ToolPolicy
from agent.state import AgentState, ArtifactRef, Budget, Observation, ToolCallRecord
from agent.verifier import AnswerVerifier, VerifyResult
from observability.tracing import trace_recorder
from services.approval_store import SQLiteApprovalStore
from services.artifact_store import SQLiteArtifactStore
from utils.streaming import get_final_response


@dataclass
class AgentTask:
    query: str
    session_id: str = "default"
    tenant_id: str = "default"
    user_role: str = "user"
    scene: str = "default"
    request_id: str = field(default_factory=lambda: str(uuid4()))
    approval_id: Optional[str] = None


@dataclass
class AgentBackendResult:
    answer: str
    evidence: List[Dict[str, Any]] = field(default_factory=list)
    tool_results: List[Dict[str, Any]] = field(default_factory=list)
    model_name: str = ""


@dataclass
class AgentRunResult:
    state: AgentState
    answer: str
    request_id: str
    approval_id: Optional[str] = None
    artifacts: List[ArtifactRef] = field(default_factory=list)
    verifier: Optional[VerifyResult] = None


class ReactAgentBackend:
    def __init__(self, agent=None) -> None:
        self.agent = agent

    def __call__(self, task: AgentTask, state: AgentState) -> AgentBackendResult:
        if self.agent is None:
            from agent.react_agent import ReactAgent

            self.agent = ReactAgent()
        chunks = list(
            self.agent.execute_stream(
                task.query,
                session_id=task.session_id,
                request_id=task.request_id,
                tenant_id=task.tenant_id,
            )
        )
        return AgentBackendResult(answer=get_final_response(chunks))


class AgentRunner:
    REPORT_KEYWORDS = ("报告", "使用记录", "月报", "总结")

    def __init__(
        self,
        backend: Optional[Callable[[AgentTask, AgentState], AgentBackendResult]] = None,
        policy: Optional[ToolPolicy] = None,
        approval_store: Optional[SQLiteApprovalStore] = None,
        artifact_store: Optional[SQLiteArtifactStore] = None,
        verifier: Optional[AnswerVerifier] = None,
        max_steps: int = 8,
        max_tool_calls: int = 5,
        max_tokens: int = 8000,
        max_cost: float = 1.0,
        max_verification_retries: int = 1,
    ) -> None:
        self.backend = backend or ReactAgentBackend()
        self.policy = policy or ToolPolicy()
        self.approval_store = approval_store or SQLiteApprovalStore()
        self.artifact_store = artifact_store or SQLiteArtifactStore()
        self.verifier = verifier or AnswerVerifier()
        self.max_steps = max_steps
        self.max_tool_calls = max_tool_calls
        self.max_tokens = max_tokens
        self.max_cost = max_cost
        self.max_verification_retries = max_verification_retries

    def run(self, task: AgentTask) -> AgentRunResult:
        self._ensure_trace(task.request_id, task.session_id)
        scene = self._resolve_scene(task)
        state = AgentState(
            request_id=task.request_id,
            session_id=task.session_id,
            tenant_id=task.tenant_id,
            user_goal=task.query,
            user_role=task.user_role,
            scene=scene,
            budget=Budget(
                max_steps=self.max_steps,
                max_tool_calls=self.max_tool_calls,
                max_tokens=self.max_tokens,
                max_cost=self.max_cost,
            ),
        )
        if not state.budget.can_continue():
            state.mark_blocked(state.budget.stop_reason() or "budget_exhausted")
            self._record_diagnostic(state, "budget", "blocked")
            return self._result(state, state.error or "budget exhausted")

        approval_result = self._handle_sensitive_report_data(task, state)
        if approval_result is not None:
            return approval_result

        answer = ""
        tool_results: List[Dict[str, Any]] = []
        verifier_result: Optional[VerifyResult] = None
        for attempt in range(self.max_verification_retries + 1):
            step = state.record_step(
                step_type="backend",
                name="execute_agent_backend",
                status="running",
                metadata={"attempt": attempt},
            )
            try:
                backend_result = self.backend(task, state)
            except Exception as exc:
                state.mark_failed(str(exc))
                self._record_diagnostic(
                    state, "backend", "failed", step_id=step.step_id,
                    failure_reason=str(exc),
                )
                return self._result(state, f"请求未执行：{exc}")

            for item in backend_result.evidence:
                state.add_observation(
                    Observation(
                        source=str(item.get("id") or item.get("source") or "evidence"),
                        content=str(item.get("content", "")),
                        metadata=item,
                    )
                )
            answer = backend_result.answer
            tool_results = backend_result.tool_results
            verifier_result = self.verifier.verify(
                query=task.query,
                answer=answer,
                evidence=backend_result.evidence,
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
            )
            if verifier_result.passed:
                break
            if verifier_result.action != "retry" or attempt >= self.max_verification_retries:
                refusal = "请求未执行：回答未通过证据校验，已拒绝输出可能不可靠的结论。"
                state.mark_rejected("verification_failed", refusal)
                artifact = self._save_artifact(
                    state,
                    artifact_type="verification_failure",
                    name="verifier-result",
                    payload={"answer": answer, "verifier": verifier_result.__dict__},
                )
                state.add_artifact(artifact)
                return self._result(state, refusal, verifier_result)

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
        state.mark_completed(answer)
        self._record_diagnostic(state, "runner", "completed")
        return self._result(state, answer, verifier_result)

    def _handle_sensitive_report_data(
        self,
        task: AgentTask,
        state: AgentState,
    ) -> Optional[AgentRunResult]:
        if not self._needs_report_data(task, state.scene):
            return None
        args = {"user_id": "runtime_user", "month": "runtime_month"}
        decision = self.policy.decide(
            tenant_id=task.tenant_id,
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
                )
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
                )
            )
            state.mark_rejected(decision.reason, "请求未执行：当前场景无权读取使用记录。")
            self._record_diagnostic(state, "policy", "denied", failure_reason=decision.reason)
            return self._result(state, state.final_answer or "", None)
        if decision.action == PolicyAction.NEED_APPROVAL:
            if task.approval_id:
                approval = self.approval_store.get(task.approval_id)
                if approval.is_approved:
                    state.add_tool_call(
                        ToolCallRecord(
                            tool_name="fetch_external_data",
                            args=approval.args,
                            status="approved",
                            approval_id=approval.approval_id,
                            risk_level="medium",
                        )
                    )
                    return None
                if approval.is_denied:
                    state.mark_rejected(
                        "approval_denied",
                        "请求未执行：敏感工具调用审批被拒绝。",
                    )
                    return self._result(state, state.final_answer or "", None)
                state.mark_pending_approval(approval.approval_id)
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
            )
            state.add_tool_call(
                ToolCallRecord(
                    tool_name="fetch_external_data",
                    args=args,
                    status="pending_approval",
                    approval_id=approval.approval_id,
                    risk_level="medium",
                )
            )
            state.mark_pending_approval(approval.approval_id)
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
        return AgentRunResult(
            state=state,
            answer=answer,
            request_id=state.request_id,
            approval_id=approval_id or state.approval_id,
            artifacts=list(state.artifacts),
            verifier=verifier,
        )

    def _resolve_scene(self, task: AgentTask) -> str:
        if task.scene != "default":
            return task.scene
        return "report" if self._needs_report_data(task, task.scene) else task.scene

    def _needs_report_data(self, task: AgentTask, scene: str) -> bool:
        return scene == "report" or any(keyword in task.query for keyword in self.REPORT_KEYWORDS)

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
        )
