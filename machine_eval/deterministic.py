from __future__ import annotations

import re
from typing import Any, Dict, Mapping, Sequence


def evaluate_deterministic(item: Mapping[str, Any]) -> Dict[str, Any]:
    expected = item.get("expected") or {}
    answer = str(item.get("agent_answer") or "")
    actual_status = str(item.get("status") or "").strip()
    expected_outcome = str(expected.get("outcome") or "").strip()
    expected_tools = list(expected.get("tools") or [])
    actual_calls = _tool_calls(item)
    actual_tool_names = [call["name"] for call in actual_calls]
    expected_tool_names = [str(tool.get("name") or "") for tool in expected_tools]

    tool_name_hits = sum(name in actual_tool_names for name in expected_tool_names)
    tool_recall = _ratio(tool_name_hits, len(expected_tool_names), empty=1.0)
    extra_tools = [name for name in actual_tool_names if name not in expected_tool_names]
    tool_precision = _ratio(
        sum(name in expected_tool_names for name in actual_tool_names),
        len(actual_tool_names),
        empty=1.0 if not expected_tool_names else 0.0,
    )
    parameter_checks = []
    for expected_tool in expected_tools:
        name = str(expected_tool.get("name") or "")
        candidates = [call for call in actual_calls if call["name"] == name]
        parameter_checks.append(
            any(_arguments_match(expected_tool, candidate["arguments"]) for candidate in candidates)
        )
    parameter_accuracy = _ratio(
        sum(parameter_checks), len(parameter_checks), empty=1.0
    )

    expected_facts = [str(value) for value in expected.get("facts") or []]
    forbidden_facts = [str(value) for value in expected.get("forbidden_facts") or []]
    fact_hits = [fact for fact in expected_facts if fact and fact in answer]
    forbidden_mentions = [fact for fact in forbidden_facts if fact and fact in answer]
    forbidden_hits = [
        fact for fact in forbidden_mentions if not _only_negated_mentions(answer, fact)
    ]
    fact_coverage = _ratio(len(fact_hits), len(expected_facts), empty=1.0)

    citations = list(item.get("citations") or [])
    if not citations:
        citations = re.findall(r"(?:\[[0-9]+\]|【[^】]+】)", answer)
    citation_validity = _citation_validity(citations)
    citation_required = bool(expected.get("requires_citation"))
    citation_requirement_met = not citation_required or (
        bool(citations) and citation_validity == 1.0
    )
    artifacts = list(item.get("artifacts") or [])
    artifact_required = bool(expected.get("requires_artifact"))
    artifact_requirement_met = not artifact_required or bool(artifacts)
    outcome_match = _outcome_matches(expected_outcome, actual_status)

    failures = []
    if not outcome_match:
        failures.append("outcome_mismatch")
    if tool_recall < 1.0:
        failures.append("required_tool_missing")
    if tool_precision < 1.0:
        failures.append("unexpected_tool_called")
    if parameter_accuracy < 1.0:
        failures.append("tool_parameter_mismatch")
    if fact_coverage < 1.0:
        failures.append("required_fact_missing")
    if forbidden_hits:
        failures.append("forbidden_fact_emitted")
    if not citation_requirement_met:
        failures.append("citation_requirement_failed")
    if not artifact_requirement_met:
        failures.append("artifact_requirement_failed")

    return {
        "status": "evaluated",
        "outcome_match": outcome_match,
        "expected_outcome": expected_outcome,
        "actual_status": actual_status,
        "tool_recall": round(tool_recall, 4),
        "tool_precision": round(tool_precision, 4),
        "tool_selection_accuracy": round(min(tool_recall, tool_precision), 4),
        "parameter_accuracy": round(parameter_accuracy, 4),
        "expected_tools": expected_tool_names,
        "actual_tools": actual_tool_names,
        "extra_tools": extra_tools,
        "fact_coverage": round(fact_coverage, 4),
        "matched_facts": fact_hits,
        "missing_facts": [fact for fact in expected_facts if fact not in fact_hits],
        "forbidden_fact_mentions": forbidden_mentions,
        "forbidden_fact_hits": forbidden_hits,
        "citation_required": citation_required,
        "citation_count": len(citations),
        "citation_validity": round(citation_validity, 4),
        "citation_requirement_met": citation_requirement_met,
        "artifact_required": artifact_required,
        "artifact_count": len(artifacts),
        "artifact_requirement_met": artifact_requirement_met,
        "passed": not failures,
        "failures": failures,
    }


def _tool_calls(item: Mapping[str, Any]) -> list[Dict[str, Any]]:
    raw_calls: Sequence[Any] = item.get("tool_calls") or item.get("trace") or []
    calls = []
    for entry in raw_calls:
        if not isinstance(entry, Mapping):
            continue
        if entry.get("category") not in {None, "tool"} and not any(
            key in entry for key in ("tool_name", "tool")
        ):
            continue
        metadata = entry.get("metadata") or {}
        if not isinstance(metadata, Mapping):
            metadata = {}
        name = entry.get("tool_name") or entry.get("tool") or metadata.get("tool")
        if not name and entry.get("category") == "tool":
            name = entry.get("name")
        if not isinstance(name, str) or not name.strip():
            continue
        arguments = entry.get(
            "arguments",
            entry.get("args", metadata.get("redacted_args", metadata.get("args", {}))),
        )
        calls.append(
            {
                "name": name.strip(),
                "arguments": dict(arguments) if isinstance(arguments, Mapping) else {},
            }
        )
    return calls


def _arguments_match(expected_tool: Mapping[str, Any], actual: Mapping[str, Any]) -> bool:
    expected = expected_tool.get("arguments") or {}
    if not isinstance(expected, Mapping):
        return False
    mode = expected_tool.get("argument_match", "contains")
    if mode == "exact":
        return dict(actual) == dict(expected)
    return all(actual.get(key) == value for key, value in expected.items())


def _citation_validity(citations: Sequence[Any]) -> float:
    if not citations:
        return 0.0
    valid = 0
    for citation in citations:
        if isinstance(citation, Mapping):
            valid += citation.get("valid") is not False and bool(
                citation.get("id") or citation.get("uri") or citation.get("source")
            )
        else:
            valid += bool(str(citation).strip())
    return valid / len(citations)


def _only_negated_mentions(answer: str, fact: str) -> bool:
    negation_markers = (
        "不提供",
        "不能提供",
        "不会提供",
        "无法提供",
        "拒绝提供",
        "禁止提供",
        "不得提供",
        "不披露",
        "不能披露",
        "不会披露",
        "不包含",
        "没有提供",
        "未提供",
    )
    starts = [match.start() for match in re.finditer(re.escape(fact), answer)]
    if not starts:
        return False
    for start in starts:
        prefix = answer[max(0, start - 24) : start]
        clause = re.split(r"[。！？；;\n]", prefix)[-1]
        if not any(marker in clause for marker in negation_markers):
            return False
    return True


def _outcome_matches(expected: str, actual: str) -> bool:
    """Compare dataset semantics with the Runner's smaller lifecycle vocabulary."""
    compatible_statuses = {
        "completed": {"completed"},
        "clarify": {"completed"},
        "refuse": {"rejected"},
        "graceful_failure": {"blocked", "failed", "rejected"},
        "completed_with_degradation": {"completed"},
        "partial_completion": {"completed"},
        "pending_approval": {"pending_approval"},
        "blocked": {"blocked"},
    }
    return bool(expected and actual in compatible_statuses.get(expected, {expected}))


def _ratio(numerator: int, denominator: int, *, empty: float) -> float:
    return numerator / denominator if denominator else empty
