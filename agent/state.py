from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Literal, Optional

from agent.budget import BudgetManager


RunStatus = Literal[
    "running",
    "pending_approval",
    "blocked",
    "failed",
    "rejected",
    "completed",
]


class Budget:
    """Backward-compatible facade over the run's shared ``BudgetManager``."""

    def __init__(
        self,
        max_steps: int = 8,
        max_tool_calls: int = 5,
        max_tokens: int = 8000,
        max_cost: float = 1.0,
        used_steps: int = 0,
        used_tool_calls: int = 0,
        used_tokens: int = 0,
        used_cost: float = 0.0,
        *,
        manager: Optional[BudgetManager] = None,
        deadline: Optional[float] = None,
        deadline_seconds: Optional[float] = None,
    ) -> None:
        self._manager = manager or BudgetManager(
            max_steps=max_steps,
            max_tool_calls=max_tool_calls,
            max_tokens=max_tokens,
            max_cost=max_cost,
            used_steps=used_steps,
            used_tool_calls=used_tool_calls,
            used_tokens=used_tokens,
            used_cost=used_cost,
            deadline=deadline,
            deadline_seconds=deadline_seconds,
        )

    @property
    def manager(self) -> BudgetManager:
        return self._manager

    @property
    def max_steps(self) -> int:
        return self._manager.max_steps

    @property
    def max_tool_calls(self) -> int:
        return self._manager.max_tool_calls

    @property
    def max_tokens(self) -> int:
        return self._manager.max_tokens

    @property
    def max_cost(self) -> float:
        return self._manager.max_cost

    @property
    def used_steps(self) -> int:
        return self._manager.used_steps

    @property
    def used_tool_calls(self) -> int:
        return self._manager.used_tool_calls

    @property
    def used_tokens(self) -> int:
        return self._manager.used_tokens

    @property
    def used_cost(self) -> float:
        return self._manager.used_cost

    def record_step(self, count: int = 1) -> None:
        self._manager.record_step(count)

    def record_tool_call(self, count: int = 1) -> None:
        self._manager.record_tool_call(count)

    def record_tokens(self, count: int) -> None:
        self._manager.record_tokens(count)

    def record_cost(self, amount: float) -> None:
        self._manager.record_cost(amount)

    def can_continue(self) -> bool:
        return self.stop_reason() is None

    def stop_reason(self) -> Optional[str]:
        return self._manager.stop_reason()


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
    user_id: Optional[str] = None
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

    def add_tool_call(self, tool_call: ToolCallRecord, *, count_budget: bool = True) -> None:
        self.tool_calls.append(tool_call)
        if count_budget:
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
