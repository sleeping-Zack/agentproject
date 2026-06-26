from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional


RunStatus = Literal[
    "running",
    "pending_approval",
    "blocked",
    "failed",
    "rejected",
    "completed",
]


@dataclass
class Budget:
    max_steps: int = 8
    max_tool_calls: int = 5
    max_tokens: int = 8000
    max_cost: float = 1.0
    used_steps: int = 0
    used_tool_calls: int = 0
    used_tokens: int = 0
    used_cost: float = 0.0

    def record_step(self, count: int = 1) -> None:
        self.used_steps += count

    def record_tool_call(self, count: int = 1) -> None:
        self.used_tool_calls += count

    def record_tokens(self, count: int) -> None:
        self.used_tokens += count

    def record_cost(self, amount: float) -> None:
        self.used_cost = round(self.used_cost + amount, 6)

    def can_continue(self) -> bool:
        return self.stop_reason() is None

    def stop_reason(self) -> Optional[str]:
        if self.used_steps >= self.max_steps:
            return "max_steps_exceeded"
        if self.used_tool_calls >= self.max_tool_calls:
            return "max_tool_calls_exceeded"
        if self.used_tokens >= self.max_tokens:
            return "max_tokens_exceeded"
        if self.used_cost >= self.max_cost:
            return "max_cost_exceeded"
        return None


@dataclass
class StepRecord:
    step_id: str
    type: str
    name: str
    status: str
    content: str = ""
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Observation:
    source: str
    content: str
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ToolCallRecord:
    tool_name: str
    args: Dict[str, Any]
    status: str
    result: str = ""
    error: Optional[str] = None
    approval_id: Optional[str] = None
    risk_level: str = "low"


@dataclass
class ArtifactRef:
    artifact_id: str
    type: str
    name: str


@dataclass
class AgentState:
    request_id: str
    session_id: str
    tenant_id: str
    user_goal: str
    user_role: str = "user"
    scene: str = "default"
    plan: List[Dict[str, Any]] = field(default_factory=list)
    current_step: int = 0
    steps: List[StepRecord] = field(default_factory=list)
    observations: List[Observation] = field(default_factory=list)
    tool_calls: List[ToolCallRecord] = field(default_factory=list)
    artifacts: List[ArtifactRef] = field(default_factory=list)
    memory_snapshot: Dict[str, Any] = field(default_factory=dict)
    budget: Budget = field(default_factory=Budget)
    status: RunStatus = "running"
    final_answer: Optional[str] = None
    error: Optional[str] = None
    approval_id: Optional[str] = None

    def record_step(
        self,
        step_type: str,
        name: str,
        status: str,
        content: str = "",
        error: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> StepRecord:
        step = StepRecord(
            step_id=f"step-{len(self.steps) + 1}",
            type=step_type,
            name=name,
            status=status,
            content=content,
            error=error,
            metadata=metadata or {},
        )
        self.steps.append(step)
        self.current_step = len(self.steps)
        self.budget.record_step()
        return step

    def add_observation(self, observation: Observation) -> None:
        self.observations.append(observation)

    def add_tool_call(self, tool_call: ToolCallRecord) -> None:
        self.tool_calls.append(tool_call)
        self.budget.record_tool_call()

    def add_artifact(self, artifact: ArtifactRef) -> None:
        self.artifacts.append(artifact)

    def mark_pending_approval(self, approval_id: str) -> None:
        self.status = "pending_approval"
        self.approval_id = approval_id

    def mark_blocked(self, reason: str) -> None:
        self.status = "blocked"
        self.error = reason

    def mark_failed(self, error: str) -> None:
        self.status = "failed"
        self.error = error

    def mark_rejected(self, reason: str, answer: Optional[str] = None) -> None:
        self.status = "rejected"
        self.error = reason
        self.final_answer = answer

    def mark_completed(self, answer: str) -> None:
        self.status = "completed"
        self.final_answer = answer

    def should_stop(self) -> bool:
        return self.status != "running" or not self.budget.can_continue()
