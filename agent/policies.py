from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Any, Dict, Iterable, List, Optional

from agent.planner import SubTask
from agent.tools.registry import ToolRegistry, build_default_tool_registry


class PolicyAction(str, Enum):
    ALLOW = "allow"
    DENY = "deny"
    NEED_APPROVAL = "need_approval"
    NEED_REDACTION = "need_redaction"


@dataclass
class PolicyDecision:
    action: PolicyAction
    reason: str
    redactions: List[str] = field(default_factory=list)


class ToolPolicy:
    ADMIN_ROLES = {"admin", "operator"}
    REPORT_SCENES = {"report", "monthly_report"}
    SENSITIVE_TOOLS = {"fetch_external_data"}

    def __init__(self, tool_registry: Optional[ToolRegistry] = None) -> None:
        self.tool_registry = tool_registry or build_default_tool_registry(
            [
                "rag_summarize",
                "get_weather",
                "get_user_location",
                "get_user_id",
                "get_current_month",
                "fetch_external_data",
                "fill_context_for_report",
            ]
        )

    def decide(
        self,
        tenant_id: str,
        user_role: str,
        scene: str,
        tool_name: str,
        args: Dict[str, Any],
    ) -> PolicyDecision:
        spec = self.tool_registry.get_spec(tool_name)
        if spec is None or tool_name not in self.tool_registry.allowed_tools:
            return PolicyDecision(
                action=PolicyAction.DENY,
                reason=f"tool not allowed for tenant={tenant_id}: {tool_name}",
            )

        redactions = [
            key for key in args
            if any(marker in key.lower() for marker in ("token", "secret", "api_key"))
        ]
        if redactions:
            return PolicyDecision(
                action=PolicyAction.NEED_REDACTION,
                reason="tool arguments contain sensitive fields",
                redactions=redactions,
            )

        if tool_name in self.SENSITIVE_TOOLS:
            if scene not in self.REPORT_SCENES:
                return PolicyDecision(
                    action=PolicyAction.DENY,
                    reason="sensitive usage data is only allowed in report scenes",
                )
            if user_role not in self.ADMIN_ROLES:
                return PolicyDecision(
                    action=PolicyAction.NEED_APPROVAL,
                    reason=f"tool requires approval: {tool_name}",
                )

        if spec.requires_approval and user_role not in self.ADMIN_ROLES:
            return PolicyDecision(
                action=PolicyAction.NEED_APPROVAL,
                reason=f"tool requires approval: {tool_name}",
            )

        return PolicyDecision(action=PolicyAction.ALLOW, reason="allowed")


@dataclass
class ValidationResult:
    valid: bool
    errors: List[str] = field(default_factory=list)


class PlanValidator:
    VALID_KINDS = {"rag_qa", "weather", "report", "generic"}

    def validate(self, plan: Iterable[SubTask], max_steps: int = 8) -> ValidationResult:
        tasks = list(plan)
        errors: List[str] = []
        ids = [task.id for task in tasks]
        if len(ids) != len(set(ids)):
            errors.append("duplicate_task_id")
        if len(tasks) > max_steps:
            errors.append("plan_exceeds_budget")
        known_ids = set(ids)
        for task in tasks:
            if task.kind not in self.VALID_KINDS:
                errors.append(f"invalid_task_kind:{task.kind}")
            missing = [dep for dep in task.depends_on if dep not in known_ids]
            if missing:
                errors.append(f"missing_dependency:{task.id}:{','.join(missing)}")
        return ValidationResult(valid=not errors, errors=errors)


class Replanner:
    def replan(self, query: str, failed_task: SubTask, failure_reason: str) -> List[SubTask]:
        return [
            SubTask(
                id="fallback-1",
                kind="generic",
                description=(
                    "原计划失败后走默认回答，"
                    f"失败任务={failed_task.id}，原因={failure_reason}"
                ),
                args={"query": query, "failure_reason": failure_reason},
            )
        ]
