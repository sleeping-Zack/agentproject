from __future__ import annotations

import re
from collections import defaultdict, deque
from copy import deepcopy
from dataclasses import dataclass, field
from datetime import datetime, time, timezone
from enum import Enum
from pathlib import Path
from threading import RLock
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Protocol
from zoneinfo import ZoneInfo, ZoneInfoNotFoundError

import yaml

from agent.planner import SubTask
from agent.tools.registry import ToolRegistry, ToolSpec, build_default_tool_registry
from safety.security import redact_sensitive
from utils.path_tool import get_abs_path


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
    reason_code: str = "legacy_decision"
    policy_version: str = "legacy"
    matched_rule_id: str = ""
    redacted_args: Dict[str, Any] = field(default_factory=dict)


@dataclass(frozen=True)
class PolicyContext:
    tenant_id: str
    principal_id: str
    role: str
    scene: str
    tool: str
    args: Dict[str, Any] = field(default_factory=dict)

    @property
    def user_role(self) -> str:
        return self.role

    @property
    def tool_name(self) -> str:
        return self.tool


class PolicyAuditSink(Protocol):
    def record(self, event: Dict[str, Any]) -> None: ...


_SENSITIVE_KEY_MARKERS = (
    "api_key",
    "authorization",
    "cookie",
    "credential",
    "password",
    "passwd",
    "private_key",
    "secret",
    "token",
)


def _redact_policy_args(value: Any, path: str = "") -> tuple[Any, List[str]]:
    """Return a copy suitable for decisions and audit events, plus redacted paths."""
    redactions: List[str] = []
    if isinstance(value, Mapping):
        result: Dict[str, Any] = {}
        for raw_key, item in value.items():
            key = str(raw_key)
            item_path = f"{path}.{key}" if path else key
            normalized = key.lower().replace("-", "_")
            if any(marker in normalized for marker in _SENSITIVE_KEY_MARKERS):
                result[key] = "<redacted>"
                redactions.append(item_path)
                continue
            redacted_item, nested = _redact_policy_args(item, item_path)
            result[key] = redacted_item
            redactions.extend(nested)
        return result, redactions
    if isinstance(value, list):
        result = []
        for index, item in enumerate(value):
            item_path = f"{path}[{index}]"
            redacted_item, nested = _redact_policy_args(item, item_path)
            result.append(redacted_item)
            redactions.extend(nested)
        return result, redactions

    redacted = redact_sensitive(value)
    if redacted != value:
        redactions.append(path or "$value")
    return redacted, redactions


class ToolPolicy:
    """Versioned, deny-by-default tool authorization policy.

    Rules are evaluated by descending ``priority``; file order breaks ties and
    the first matching rule wins. Registry membership is a hard boundary that
    YAML cannot override. Rule rate limits are deliberately process-local; a
    distributed deployment must replace this instance state with Redis or an
    equivalent shared atomic counter.
    """

    _SELECTORS = {
        "tenants": "tenant_id",
        "roles": "role",
        "scenes": "scene",
        "tools": "tool",
    }
    _SINGULAR_SELECTORS = {
        "tenants": "tenant",
        "roles": "role",
        "scenes": "scene",
        "tools": "tool",
        "data_scopes": "data_scope",
        "risk_levels": "risk_level",
    }
    _RATE_KEY_FIELDS = {
        "tenant_id",
        "principal_id",
        "role",
        "scene",
        "tool",
        "data_scope",
        "risk_level",
    }
    _RULE_KEYS = {
        "action",
        "id",
        "match",
        "priority",
        "rate_limit",
        "reason",
        "reason_code",
        "requires_approval",
        "time_window",
    }
    _MATCH_KEYS = {
        "argument_constraints",
        "data_scope",
        "data_scopes",
        "risk_level",
        "risk_levels",
        "role",
        "roles",
        "scene",
        "scenes",
        "tenant",
        "tenants",
        "time_window",
        "tool",
        "tools",
    }
    _ARGUMENT_OPERATORS = {
        "equals",
        "equals_context",
        "exists",
        "max",
        "min",
        "not_equals",
        "not_in",
        "one_of",
        "regex",
    }

    def __init__(
        self,
        tool_registry: Optional[ToolRegistry] = None,
        config_path: Optional[str | Path] = None,
        audit_sink: Optional[PolicyAuditSink | Callable[[Dict[str, Any]], None]] = None,
        clock: Optional[Callable[[], datetime]] = None,
    ) -> None:
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
        self.audit_sink = audit_sink
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        self._rate_windows: Dict[tuple[str, ...], deque[float]] = defaultdict(deque)
        self._rate_lock = RLock()

        path = Path(config_path) if config_path is not None else Path(
            get_abs_path("config/tool_policy.yml")
        )
        self.config_path = path
        self._config = self._load_config(path)
        self.policy_version = str(self._config["version"])
        self._default = self._config.get("default") or self._config.get("defaults") or {}
        self._validate_default(self._default)
        self._rules = self._prepare_rules(self._config.get("rules", []))

    def decide(
        self,
        tenant_id: str,
        user_role: str,
        scene: str,
        tool_name: str,
        args: Dict[str, Any],
        principal_id: Optional[str] = None,
    ) -> PolicyDecision:
        if not isinstance(args, dict):
            raise TypeError("policy args must be a dictionary")
        return self.decide_context(
            PolicyContext(
                tenant_id=tenant_id,
                principal_id=principal_id or f"{user_role}:{tenant_id}",
                role=user_role,
                scene=scene,
                tool=tool_name,
                args=dict(args),
            )
        )

    def decide_context(self, context: PolicyContext) -> PolicyDecision:
        now = self._now()
        redacted_args, redactions = _redact_policy_args(context.args)
        spec = self.tool_registry.get_spec(context.tool)

        if spec is None or context.tool not in self.tool_registry.allowed_tools:
            return self._finish(
                context,
                spec,
                now,
                PolicyDecision(
                    action=PolicyAction.DENY,
                    reason=f"tool not allowed for tenant={context.tenant_id}: {context.tool}",
                    reason_code="tool_not_registered_or_allowed",
                    policy_version=self.policy_version,
                    matched_rule_id="registry-boundary",
                    redacted_args=redacted_args,
                    redactions=redactions,
                ),
            )

        if redactions:
            return self._finish(
                context,
                spec,
                now,
                PolicyDecision(
                    action=PolicyAction.NEED_REDACTION,
                    reason="tool arguments contain sensitive fields",
                    reason_code="sensitive_arguments",
                    policy_version=self.policy_version,
                    matched_rule_id="sensitive-argument-boundary",
                    redacted_args=redacted_args,
                    redactions=redactions,
                ),
            )

        for rule in self._rules:
            if not self._rule_matches(rule, context, spec, now):
                continue
            if not self._consume_rate_limit(rule, context, spec, now):
                return self._finish(
                    context,
                    spec,
                    now,
                    PolicyDecision(
                        action=PolicyAction.DENY,
                        reason="tool policy rate limit exceeded",
                        reason_code="rate_limit_exceeded",
                        policy_version=self.policy_version,
                        matched_rule_id=rule["id"],
                        redacted_args=redacted_args,
                    ),
                )

            configured_action = PolicyAction(rule.get("action", PolicyAction.DENY.value))
            requires_approval = rule.get("requires_approval")
            if requires_approval is None:
                requires_approval = spec.requires_approval
            action = (
                PolicyAction.NEED_APPROVAL
                if configured_action == PolicyAction.ALLOW and requires_approval
                else configured_action
            )
            reason = rule.get("reason") or self._default_reason(action, context.tool)
            reason_code = rule.get("reason_code") or f"rule_{action.value}"
            return self._finish(
                context,
                spec,
                now,
                PolicyDecision(
                    action=action,
                    reason=reason,
                    reason_code=reason_code,
                    policy_version=self.policy_version,
                    matched_rule_id=rule["id"],
                    redacted_args=redacted_args,
                ),
            )

        return self._finish(
            context,
            spec,
            now,
            PolicyDecision(
                action=PolicyAction.DENY,
                reason=str(self._default.get("reason", "no matching policy rule")),
                reason_code=str(self._default.get("reason_code", "default_deny")),
                policy_version=self.policy_version,
                matched_rule_id="default-deny",
                redacted_args=redacted_args,
            ),
        )

    @staticmethod
    def _load_config(path: Path) -> Dict[str, Any]:
        with path.open("r", encoding="utf-8") as handle:
            loaded = yaml.safe_load(handle)
        if not isinstance(loaded, dict):
            raise ValueError(f"tool policy config must be a mapping: {path}")
        if not loaded.get("version"):
            raise ValueError("tool policy config requires a non-empty version")
        return loaded

    @staticmethod
    def _validate_default(default: Mapping[str, Any]) -> None:
        if default.get("action", PolicyAction.DENY.value) != PolicyAction.DENY.value:
            raise ValueError("tool policy default action must be deny")

    def _prepare_rules(self, raw_rules: Any) -> List[Dict[str, Any]]:
        if not isinstance(raw_rules, list):
            raise ValueError("tool policy rules must be a list")
        seen_ids = set()
        prepared = []
        for order, raw_rule in enumerate(raw_rules):
            if not isinstance(raw_rule, dict):
                raise ValueError("each tool policy rule must be a mapping")
            rule = deepcopy(raw_rule)
            rule_id = rule.get("id")
            if not isinstance(rule_id, str) or not rule_id.strip():
                raise ValueError("each tool policy rule requires a non-empty id")
            if rule_id in seen_ids:
                raise ValueError(f"duplicate tool policy rule id: {rule_id}")
            seen_ids.add(rule_id)
            unknown_rule_keys = set(rule) - self._RULE_KEYS
            if unknown_rule_keys:
                raise ValueError(
                    f"unknown keys for rule {rule_id}: {sorted(unknown_rule_keys)}"
                )
            try:
                PolicyAction(rule.get("action", PolicyAction.DENY.value))
            except ValueError as exc:
                raise ValueError(f"invalid action for rule {rule_id}") from exc
            if "requires_approval" in rule and not isinstance(rule["requires_approval"], bool):
                raise ValueError(f"requires_approval must be boolean for rule {rule_id}")
            rule["priority"] = int(rule.get("priority", 0))
            rule["_order"] = order
            self._validate_rule_conditions(rule)
            prepared.append(rule)
        return sorted(prepared, key=lambda item: (-item["priority"], item["_order"]))

    def _validate_rule_conditions(self, rule: Mapping[str, Any]) -> None:
        match = rule.get("match", {})
        if not isinstance(match, dict):
            raise ValueError(f"match must be a mapping for rule {rule['id']}")
        unknown_match_keys = set(match) - self._MATCH_KEYS
        if unknown_match_keys:
            raise ValueError(
                f"unknown match keys for rule {rule['id']}: {sorted(unknown_match_keys)}"
            )
        for selector in self._SINGULAR_SELECTORS:
            singular = self._SINGULAR_SELECTORS[selector]
            for key in (selector, singular):
                values = match.get(key)
                if values is not None and not isinstance(values, (str, list, tuple, set)):
                    raise ValueError(f"{key} must be a string or list for rule {rule['id']}")
        constraints = match.get("argument_constraints")
        if constraints is not None and not isinstance(constraints, dict):
            raise ValueError(f"argument_constraints must be a mapping for rule {rule['id']}")
        if constraints:
            unknown_constraint_keys = set(constraints) - {"required", "forbidden", "fields"}
            if unknown_constraint_keys:
                raise ValueError(
                    f"unknown argument constraint keys for rule {rule['id']}: "
                    f"{sorted(unknown_constraint_keys)}"
                )
            for key in ("required", "forbidden"):
                if not isinstance(constraints.get(key, []), list):
                    raise ValueError(f"argument {key} must be a list for rule {rule['id']}")
            fields = constraints.get("fields", {})
            if not isinstance(fields, dict):
                raise ValueError(f"argument fields must be a mapping for rule {rule['id']}")
            for field_name, condition in fields.items():
                if not isinstance(condition, dict):
                    continue
                unknown_operators = set(condition) - self._ARGUMENT_OPERATORS
                if unknown_operators:
                    raise ValueError(
                        f"unknown operators for {field_name} in rule {rule['id']}: "
                        f"{sorted(unknown_operators)}"
                    )
                if "regex" in condition:
                    try:
                        re.compile(str(condition["regex"]))
                    except re.error as exc:
                        raise ValueError(
                            f"invalid regex for {field_name} in rule {rule['id']}"
                        ) from exc
                if "equals_context" in condition and condition["equals_context"] not in {
                    "tenant_id",
                    "principal_id",
                    "role",
                    "scene",
                    "tool",
                }:
                    raise ValueError(
                        f"invalid equals_context for {field_name} in rule {rule['id']}"
                    )
        window = match.get("time_window", rule.get("time_window"))
        if window is not None:
            if not isinstance(window, dict):
                raise ValueError(f"time_window must be a mapping for rule {rule['id']}")
            try:
                ZoneInfo(str(window.get("timezone", "UTC")))
                self._parse_clock_time(str(window.get("start", "00:00")))
                self._parse_clock_time(str(window.get("end", "00:00")))
                self._normalize_weekdays(window.get("days", []))
            except (ValueError, ZoneInfoNotFoundError) as exc:
                raise ValueError(f"invalid time_window for rule {rule['id']}") from exc
        rate = rule.get("rate_limit")
        if rate is not None:
            if not isinstance(rate, dict):
                raise ValueError(f"rate_limit must be a mapping for rule {rule['id']}")
            if int(rate.get("max_calls", 0)) <= 0 or float(rate.get("window_seconds", 0)) <= 0:
                raise ValueError(f"invalid rate_limit for rule {rule['id']}")
            key_by = rate.get("key_by", ["tenant_id", "principal_id", "tool"])
            if not isinstance(key_by, list) or any(
                key not in self._RATE_KEY_FIELDS and not str(key).startswith("arg.")
                for key in key_by
            ):
                raise ValueError(f"invalid rate_limit key_by for rule {rule['id']}")

    def _rule_matches(
        self,
        rule: Mapping[str, Any],
        context: PolicyContext,
        spec: ToolSpec,
        now: datetime,
    ) -> bool:
        match = rule.get("match", {})
        for selector, context_field in self._SELECTORS.items():
            if not self._matches_selector(
                match,
                selector,
                str(getattr(context, context_field)),
            ):
                return False
        if not self._matches_selector(match, "data_scopes", spec.scope):
            return False
        if not self._matches_selector(match, "risk_levels", spec.risk_level):
            return False
        if not self._arguments_match(match.get("argument_constraints"), context):
            return False
        window = match.get("time_window", rule.get("time_window"))
        return self._time_window_matches(window, now)

    def _matches_selector(
        self,
        match: Mapping[str, Any],
        plural_name: str,
        actual: str,
    ) -> bool:
        expected = match.get(plural_name)
        if expected is None:
            expected = match.get(self._SINGULAR_SELECTORS[plural_name])
        if expected is None:
            return True
        values = [expected] if isinstance(expected, str) else list(expected)
        normalized = {str(value) for value in values}
        return "*" in normalized or actual in normalized

    @staticmethod
    def _arguments_match(
        constraints: Optional[Mapping[str, Any]],
        context: PolicyContext,
    ) -> bool:
        if not constraints:
            return True
        args = context.args
        required = constraints.get("required", [])
        forbidden = constraints.get("forbidden", [])
        if any(field not in args for field in required):
            return False
        if any(field in args for field in forbidden):
            return False

        context_values = {
            "tenant_id": context.tenant_id,
            "principal_id": context.principal_id,
            "role": context.role,
            "scene": context.scene,
            "tool": context.tool,
        }
        for field_name, condition in constraints.get("fields", {}).items():
            if not isinstance(condition, dict):
                condition = {"equals": condition}
            exists = field_name in args
            if condition.get("exists") is False:
                if exists:
                    return False
                continue
            if not exists:
                return False
            value = args[field_name]
            if "equals" in condition and value != condition["equals"]:
                return False
            if "not_equals" in condition and value == condition["not_equals"]:
                return False
            if "one_of" in condition and value not in condition["one_of"]:
                return False
            if "not_in" in condition and value in condition["not_in"]:
                return False
            if "regex" in condition and re.fullmatch(str(condition["regex"]), str(value)) is None:
                return False
            try:
                if "min" in condition and value < condition["min"]:
                    return False
                if "max" in condition and value > condition["max"]:
                    return False
            except TypeError:
                return False
            if "equals_context" in condition:
                expected = context_values.get(str(condition["equals_context"]))
                if expected is None or value != expected:
                    return False
        return True

    @staticmethod
    def _parse_clock_time(value: str) -> time:
        return datetime.strptime(value, "%H:%M").time()

    @staticmethod
    def _normalize_weekdays(configured_days: Iterable[Any]) -> set[int]:
        day_names = {
            "mon": 0,
            "tue": 1,
            "wed": 2,
            "thu": 3,
            "fri": 4,
            "sat": 5,
            "sun": 6,
        }
        result = set()
        for day in configured_days:
            if isinstance(day, int):
                if day not in range(7):
                    raise ValueError(f"invalid weekday: {day}")
                result.add(day)
                continue
            key = str(day).lower()[:3]
            if key not in day_names:
                raise ValueError(f"invalid weekday: {day}")
            result.add(day_names[key])
        return result

    def _time_window_matches(
        self,
        window: Optional[Mapping[str, Any]],
        now: datetime,
    ) -> bool:
        if not window:
            return True
        local_now = now.astimezone(ZoneInfo(str(window.get("timezone", "UTC"))))
        configured_days = window.get("days")
        if configured_days:
            allowed_days = self._normalize_weekdays(configured_days)
            if local_now.weekday() not in allowed_days:
                return False
        start = self._parse_clock_time(str(window.get("start", "00:00")))
        end = self._parse_clock_time(str(window.get("end", "00:00")))
        current = local_now.time().replace(tzinfo=None)
        if start == end:
            return True
        if start < end:
            return start <= current < end
        return current >= start or current < end

    def _consume_rate_limit(
        self,
        rule: Mapping[str, Any],
        context: PolicyContext,
        spec: ToolSpec,
        now: datetime,
    ) -> bool:
        rate = rule.get("rate_limit")
        if not rate:
            return True
        key_values = {
            "tenant_id": context.tenant_id,
            "principal_id": context.principal_id,
            "role": context.role,
            "scene": context.scene,
            "tool": context.tool,
            "data_scope": spec.scope,
            "risk_level": spec.risk_level,
        }
        key_parts = [str(rule["id"])]
        for key in rate.get("key_by", ["tenant_id", "principal_id", "tool"]):
            if str(key).startswith("arg."):
                key_parts.append(str(context.args.get(str(key)[4:], "<missing>")))
            else:
                key_parts.append(str(key_values[str(key)]))
        rate_key = tuple(key_parts)
        max_calls = int(rate["max_calls"])
        window_seconds = float(rate["window_seconds"])
        timestamp = now.timestamp()
        with self._rate_lock:
            calls = self._rate_windows[rate_key]
            while calls and timestamp - calls[0] >= window_seconds:
                calls.popleft()
            if len(calls) >= max_calls:
                return False
            calls.append(timestamp)
        return True

    def _now(self) -> datetime:
        value = self._clock()
        if not isinstance(value, datetime):
            raise TypeError("policy clock must return datetime")
        return value if value.tzinfo is not None else value.replace(tzinfo=timezone.utc)

    @staticmethod
    def _default_reason(action: PolicyAction, tool: str) -> str:
        if action == PolicyAction.ALLOW:
            return "allowed"
        if action == PolicyAction.NEED_APPROVAL:
            return f"tool requires approval: {tool}"
        if action == PolicyAction.NEED_REDACTION:
            return "tool arguments require redaction"
        return "denied by policy"

    def _finish(
        self,
        context: PolicyContext,
        spec: Optional[ToolSpec],
        now: datetime,
        decision: PolicyDecision,
    ) -> PolicyDecision:
        event = {
            "timestamp": now.isoformat(),
            "tenant_id": context.tenant_id,
            "principal_id": context.principal_id,
            "role": context.role,
            "scene": context.scene,
            "tool": context.tool,
            "data_scope": spec.scope if spec else None,
            "risk_level": spec.risk_level if spec else None,
            "action": decision.action.value,
            "reason": decision.reason,
            "reason_code": decision.reason_code,
            "policy_version": decision.policy_version,
            "matched_rule_id": decision.matched_rule_id,
            "redacted_args": deepcopy(decision.redacted_args),
            "redactions": list(decision.redactions),
        }
        if self.audit_sink is not None:
            record = getattr(self.audit_sink, "record", None)
            if callable(record):
                record(event)
            elif callable(self.audit_sink):
                self.audit_sink(event)
            else:
                raise TypeError("audit_sink must be callable or expose record(event)")
        return decision


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
