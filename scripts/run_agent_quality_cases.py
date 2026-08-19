from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Sequence
from uuid import uuid4

from agent.policies import ToolPolicy
from agent.react_agent import ReactAgent
from agent.runner import (
    AgentRunner,
    AgentTask,
    AutoRoutingBackend,
    PlannerAgentBackend,
    ReactAgentBackend,
)
from agent.tools.agent_tools import tool_registry
from observability.tracing import trace_recorder
from scripts.prepare_human_eval_batch import load_dataset, load_jsonl
from services.factories import create_approval_store, create_artifact_store
from utils.config_handler import rag_conf
from utils.prompt_loader import load_prompt_document


SUPPORTED_CASE_CONTEXT_KEYS = frozenset(
    {"tenant_id", "principal_id", "data_user_id", "user_role"}
)


def unsupported_case_features(case: Mapping[str, Any]) -> list[str]:
    context = case.get("context") or {}
    if not isinstance(context, Mapping):
        return ["context:invalid"]
    unsupported = [
        f"context:{key}" for key in sorted(set(context) - SUPPORTED_CASE_CONTEXT_KEYS)
    ]
    if case.get("turns"):
        unsupported.append("turns")
    return unsupported


def select_cases(
    dataset: Mapping[str, Mapping[str, Any]],
    *,
    split: str,
    case_ids: Sequence[str],
    limit: int | None,
    runnable_only: bool = False,
) -> list[Mapping[str, Any]]:
    requested = set(case_ids)
    unknown = sorted(requested - set(dataset))
    if unknown:
        raise ValueError(f"unknown case IDs: {unknown[:5]}")
    wrong_split = sorted(
        case_id
        for case_id in requested
        if str(dataset[case_id].get("split")) != split
    )
    if wrong_split:
        raise ValueError(f"case IDs do not belong to split {split!r}: {wrong_split[:5]}")

    matching = [
        case
        for case_id, case in sorted(dataset.items())
        if str(case.get("split")) == split and (not requested or case_id in requested)
    ]
    unsupported = [
        (str(case["case_id"]), unsupported_case_features(case))
        for case in matching
        if unsupported_case_features(case)
    ]
    if requested and runnable_only and unsupported:
        raise ValueError(
            "explicitly requested cases require unsupported runtime fixtures: "
            f"{unsupported[:5]}"
        )
    if runnable_only:
        matching = [case for case in matching if not unsupported_case_features(case)]
    selected = matching
    if limit is not None:
        if limit <= 0:
            raise ValueError("limit must be positive")
        selected = selected[:limit]
    if not runnable_only:
        selected_unsupported = [
            (str(case["case_id"]), unsupported_case_features(case))
            for case in selected
            if unsupported_case_features(case)
        ]
        if selected_unsupported:
            raise ValueError(
                "selected cases require runtime fixtures this runner cannot reproduce: "
                f"{selected_unsupported[:5]}; use --runnable-only or implement the fixtures"
            )
    if not selected:
        raise ValueError("no cases matched the requested selection")
    return selected


def run_case(
    runner: Any,
    case: Mapping[str, Any],
    *,
    variant: str,
    model_snapshot: str | Mapping[str, Any],
    tenant_id: str,
    user_id: str,
) -> Dict[str, Any]:
    case_id = str(case["case_id"])
    context = case.get("context") or {}
    if not isinstance(context, Mapping):
        context = {}
    run_tenant_id = str(context.get("tenant_id") or tenant_id)
    run_user_id = str(context.get("principal_id") or user_id)
    data_user_id = str(context.get("data_user_id") or run_user_id)
    user_role = str(context.get("user_role") or "user")
    expected = case.get("expected") or {}
    expected_tools = expected.get("tools") or [] if isinstance(expected, Mapping) else []
    required_tools = tuple(
        dict.fromkeys(
            str(item.get("name") or "").strip()
            for item in expected_tools
            if isinstance(item, Mapping) and str(item.get("name") or "").strip()
        )
    )
    request_id = str(uuid4())
    started = time.perf_counter()
    try:
        result = runner.run(
            AgentTask(
                query=str(case["query"]),
                session_id=f"agent-quality:{variant}:{case_id}",
                request_id=request_id,
                tenant_id=run_tenant_id,
                user_id=run_user_id,
                data_user_id=data_user_id,
                user_role=user_role,
                scene="default",
                required_tools=required_tools,
            )
        )
        trace = trace_recorder.export_trace(request_id).get("events") or []
        tokens_in, tokens_out, cost, cost_mode, observed_model = _usage(trace)
        if cost_mode == "not_available":
            cost = None
            tokens_in = None
            tokens_out = None
        approval_records = []
        approval_ids = {
            str(value)
            for value in (
                getattr(result, "approval_id", None),
                getattr(result.state, "approval_id", None),
                *(item.approval_id for item in result.state.tool_calls),
            )
            if value
        }
        approval_store = getattr(runner, "approval_store", None)
        for approval_id in sorted(approval_ids):
            if approval_store is None:
                break
            try:
                approval_records.append(asdict(approval_store.get(approval_id)))
            except KeyError:
                continue
        return {
            "case_id": case_id,
            "agent_answer": result.answer,
            "status": result.state.status,
            "trace": trace,
            "tool_calls": [asdict(item) for item in result.state.tool_calls],
            "planner_steps": list(result.state.plan),
            "approval_records": approval_records,
            "evidence": [dict(item.metadata) for item in result.state.observations],
            "citations": [],
            "artifacts": [asdict(item) for item in result.artifacts],
            "policy_context": {
                "tenant_id": run_tenant_id,
                "principal_id": run_user_id,
                "data_user_id": data_user_id,
                "user_role": user_role,
                "scene": result.state.scene,
            },
            "model_metadata": {
                "model_snapshot": model_snapshot,
                "observed_model": observed_model,
                "variant": variant,
            },
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "estimated_cost": cost,
            "cost_mode": cost_mode,
            "tokens_in": tokens_in,
            "tokens_out": tokens_out,
            "request_id": request_id,
            "error": result.state.error,
        }
    except Exception as exc:
        trace = _trace_or_empty(request_id)
        return {
            "case_id": case_id,
            "agent_answer": "",
            "status": "failed",
            "trace": trace,
            "tool_calls": [],
            "planner_steps": [],
            "approval_records": [],
            "evidence": [],
            "citations": [],
            "artifacts": [],
            "policy_context": {
                "tenant_id": run_tenant_id,
                "principal_id": run_user_id,
                "data_user_id": data_user_id,
                "user_role": user_role,
                "scene": "default",
            },
            "model_metadata": {
                "model_snapshot": model_snapshot,
                "observed_model": None,
                "variant": variant,
            },
            "latency_ms": round((time.perf_counter() - started) * 1000, 3),
            "estimated_cost": None,
            "cost_mode": "not_available",
            "tokens_in": None,
            "tokens_out": None,
            "request_id": request_id,
            "error": f"{type(exc).__name__}: {exc}",
        }


def run_cases(
    cases: Sequence[Mapping[str, Any]],
    *,
    runner: Any,
    variant: str,
    model_snapshot: str | Mapping[str, Any],
    tenant_id: str,
    user_id: str,
    existing_case_ids: set[str] | None = None,
    on_result: Callable[[Dict[str, Any]], None] | None = None,
) -> list[Dict[str, Any]]:
    completed = existing_case_ids or set()
    output = []
    for case in cases:
        if str(case["case_id"]) in completed:
            continue
        result = run_case(
            runner,
            case,
            variant=variant,
            model_snapshot=model_snapshot,
            tenant_id=tenant_id,
            user_id=user_id,
        )
        output.append(result)
        if on_result is not None:
            on_result(result)
    return output


def build_runner() -> AgentRunner:
    react_agent = ReactAgent()
    policy = ToolPolicy(tool_registry=tool_registry)
    return AgentRunner(
        backend=AutoRoutingBackend(
            react_backend=ReactAgentBackend(agent=react_agent),
            planner_backend=PlannerAgentBackend(agent=react_agent),
            tool_policy=policy,
        ),
        policy=policy,
        approval_store=create_approval_store(),
        artifact_store=create_artifact_store(),
        conversation_memory=react_agent.memory,
    )


def runtime_snapshot(requested: str) -> Dict[str, Any]:
    prompt = load_prompt_document("main")
    prompt_path = Path("prompts/main_prompt.txt")
    try:
        commit = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        ).stdout.strip()
    except (OSError, subprocess.SubprocessError):
        commit = None
    return {
        "snapshot_id": requested,
        "model_provider": str(rag_conf.get("model_provider") or ""),
        "model_name": str(rag_conf.get("chat_model_name") or ""),
        "prompt_version": prompt.version if prompt else "unknown",
        "prompt_sha256": (
            hashlib.sha256(prompt_path.read_bytes()).hexdigest()
            if prompt_path.exists()
            else None
        ),
        "git_commit": commit,
    }


def _usage(
    events: Sequence[Mapping[str, Any]],
) -> tuple[int | None, int | None, float | None, str, str | None]:
    model_usage_events = []
    runner_usage_events = []
    for event in events:
        metadata = event.get("metadata") or {}
        if event.get("category") != "diagnostic" or not (
            metadata.get("tokens_in")
            or metadata.get("tokens_out")
            or metadata.get("cost")
        ):
            continue
        if metadata.get("type") == "model_usage":
            model_usage_events.append(metadata)
        elif metadata.get("type") == "verifier":
            runner_usage_events.append(metadata)
    # AgentRunner repeats backend usage in its verifier diagnostic. Prefer the
    # underlying model events and use verifier diagnostics only for estimated
    # usage produced by backends without model-usage instrumentation.
    usage_events = model_usage_events or runner_usage_events
    if not usage_events:
        return None, None, None, "not_available", None
    tokens_in = sum(int(item.get("tokens_in") or 0) for item in usage_events)
    tokens_out = sum(int(item.get("tokens_out") or 0) for item in usage_events)
    cost = round(sum(float(item.get("cost") or 0.0) for item in usage_events), 6)
    modes = [str(item.get("cost_mode")) for item in usage_events if item.get("cost_mode")]
    models = [str(item.get("model_name")) for item in usage_events if item.get("model_name")]
    return tokens_in, tokens_out, cost, (modes[-1] if modes else "not_available"), (
        models[-1] if models else None
    )


def _trace_or_empty(request_id: str) -> list[Dict[str, Any]]:
    try:
        return list(trace_recorder.export_trace(request_id).get("events") or [])
    except KeyError:
        return []


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run Agent-quality cases and persist auditable run-result JSONL."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("evals/agent_quality/v1"))
    parser.add_argument("--split", choices=["dev", "regression", "test"], default="dev")
    parser.add_argument("--allow-test", action="store_true")
    parser.add_argument(
        "--runnable-only",
        action="store_true",
        help=(
            "select only cases whose context and conversation state this runner can "
            "faithfully reproduce"
        ),
    )
    parser.add_argument("--case-id", action="append", default=[])
    parser.add_argument("--limit", type=int)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--model-snapshot", required=True)
    parser.add_argument("--tenant", default="eval")
    parser.add_argument("--user-id", default="1005")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.split == "test" and not args.allow_test:
        raise ValueError("test split requires explicit --allow-test")
    dataset = load_dataset(args.dataset_dir)
    snapshot = runtime_snapshot(args.model_snapshot)
    cases = select_cases(
        dataset,
        split=args.split,
        case_ids=args.case_id,
        limit=args.limit,
        runnable_only=args.runnable_only,
    )
    existing = set()
    if args.output.exists():
        if not args.resume:
            raise ValueError(f"output already exists; use --resume: {args.output}")
        existing_rows = load_jsonl(args.output)
        existing = {str(item["case_id"]) for item in existing_rows}
        incompatible = [
            str(item["case_id"])
            for item in existing_rows
            if item.get("model_metadata", {}).get("variant") != args.variant
            or item.get("model_metadata", {}).get("model_snapshot") != snapshot
        ]
        if incompatible:
            raise ValueError(
                "resume output contains a different variant or runtime snapshot: "
                f"{incompatible[:5]}"
            )
    args.output.parent.mkdir(parents=True, exist_ok=True)

    def append_result(result: Dict[str, Any]) -> None:
        with args.output.open("a", encoding="utf-8", newline="\n") as handle:
            handle.write(json.dumps(result, ensure_ascii=False, sort_keys=True) + "\n")
        print(
            json.dumps(
                {"case_id": result["case_id"], "status": result["status"]},
                ensure_ascii=False,
            )
        )

    generated = run_cases(
        cases,
        runner=build_runner(),
        variant=args.variant,
        model_snapshot=snapshot,
        tenant_id=args.tenant,
        user_id=args.user_id,
        existing_case_ids=existing,
        on_result=append_result,
    )
    print(
        json.dumps(
            {
                "output": str(args.output),
                "selected": len(cases),
                "already_present": len(existing & {str(case['case_id']) for case in cases}),
                "generated": len(generated),
                "variant": args.variant,
                "model_snapshot": snapshot,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
