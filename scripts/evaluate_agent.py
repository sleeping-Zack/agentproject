"""端到端 Agent 评测：跑 evals/agent_golden.jsonl，输出通过率与工具命中率。

评测维度：
    1. 工具命中率：Agent 是否调用了 expected_tools 中列出的工具（按名匹配）
       工具调用顺序/参数完全一致是 strict 命中，仅名字命中是 soft 命中
    2. 关键词命中率：最终回答是否包含 expected_keywords
    3. 拒绝率：expected_rejection=true 的 case 是否被安全模块挡下
    4. 总体通过：工具 ≥ 0.5 且关键词 ≥ 0.5 视为 PASS（阈值可调）

调用模式：
    --quiet            只输出最终 JSON 汇总，便于 prompt_diff.py 抓取
    --smoke            只跑前 N 条，CI 用（CI_SMOKE_LIMIT 环境变量也可控）
    --report path.json 写一份机读评测报告

为什么不直接接 LLM-as-judge：判分依据可解释 + 不消耗额外配额。LLM judge 留给
线上质量复核脚本 evaluate_judge.py。
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
import time
import traceback
from datetime import datetime, timezone
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from observability.tracing import trace_recorder
from rag.eval_gate import EvalGate, EvalThresholds
from utils.evaluation_gate_config import (
    DEFAULT_GATE_CONFIG,
    load_gate_profile,
    policy_value,
)
from utils.streaming import get_final_response


@dataclass
class CaseResult:
    id: str
    passed: bool
    tool_recall: float
    keyword_recall: float
    rejected: Optional[bool]
    parameter_accuracy: float = 1.0
    citation_validity: float = 1.0
    artifact_saved: bool = True
    bucket: str = "general"
    risk_tier: str = "standard"
    tool_applicable: bool = False
    keyword_applicable: bool = False
    parameter_applicable: bool = False
    citation_applicable: bool = False
    artifact_applicable: bool = False
    latency_ms: float = 0.0
    error_type: Optional[str] = None
    error: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)


def load_golden(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        cases = [json.loads(line) for line in f if line.strip()]
    if not cases:
        raise ValueError("agent golden must not be empty")
    seen = set()
    for case in cases:
        case_id = case.get("id")
        if not isinstance(case_id, str) or not case_id or case_id in seen:
            raise ValueError(f"invalid or duplicate agent case id: {case_id}")
        seen.add(case_id)
    return cases


def apply_gate_metadata(
    cases: List[Dict], policy: Dict[str, Any], *, strict: bool = True
) -> None:
    """Attach audited risk tiers while rejecting unclassified gated buckets."""
    risk_by_bucket = policy.get("bucket_risk_tiers") or {}
    for case in cases:
        bucket = str(case.get("bucket") or _infer_bucket(case))
        risk_tier = case.get("risk_tier") or risk_by_bucket.get(bucket)
        if risk_tier not in {"standard", "high"} and strict:
            raise ValueError(
                f"agent case {case['id']} bucket {bucket!r} has no valid risk tier"
            )
        case["_gate_risk_tier"] = risk_tier or "standard"


def _case_dimensions(case: Dict) -> Dict[str, bool | str]:
    expected_tools = list(case.get("expected_tools") or [])
    bucket = str(case.get("bucket") or _infer_bucket(case))
    expected_status = case.get(
        "expected_status", "rejected" if case.get("expected_rejection") else "completed"
    )
    return {
        "risk_tier": str(case.get("_gate_risk_tier") or case.get("risk_tier") or "standard"),
        "tool_applicable": bool(expected_tools),
        "keyword_applicable": bool(case.get("expected_keywords")),
        "parameter_applicable": any(tool.get("args") is not None for tool in expected_tools),
        "citation_applicable": bool(
            case.get("citation_applicable", bucket in {"rag", "citation"})
        ),
        "artifact_applicable": bool(
            case.get("expect_artifact", expected_status == "completed")
        ),
    }


def _case_query(case: Dict) -> str:
    return case["turns"][-1]["content"] if case.get("turns") else case["query"]


class _OfflineBackend:
    """Golden 驱动的确定性 backend；完整 Runner 仍负责策略、预算、验证与产物。"""

    def __init__(self, cases: List[Dict]) -> None:
        self._cases = {_case_query(case): case for case in cases}

    def __call__(self, task, state):
        from agent.runner import AgentBackendResult

        case = self._cases[task.query]
        fixture = case.get("mock_result") or {}
        if fixture.get("error"):
            raise RuntimeError(str(fixture["error"]))
        return AgentBackendResult(
            answer=str(fixture.get("answer", "")),
            evidence=list(fixture.get("evidence") or []),
            tool_results=list(fixture.get("tool_results") or []),
            model_name="scripted-mock",
            tokens_in=int(fixture.get("tokens_in", 24)),
            tokens_out=int(fixture.get("tokens_out", 16)),
            cost=float(fixture.get("cost", 0.00004)),
            cost_mode="estimated",
        )


class _OfflineRunnerFactory:
    def __init__(self, cases: List[Dict]) -> None:
        self.backend = _OfflineBackend(cases)

    def build(self, case: Dict):
        from agent.runner import AgentRunner

        options = dict(case.get("runner_options") or {})
        options.setdefault("max_verification_retries", 0)
        return AgentRunner(backend=self.backend, **options)


def _tools_actually_called(request_id: str) -> List[str]:
    try:
        events = trace_recorder.export_trace(request_id)["events"]
    except KeyError:
        return []
    return [e["name"] for e in events if e["category"] == "tool"]


def _evaluate_case(agent, case: Dict) -> CaseResult:
    expected_tools = [t.get("name") for t in case.get("expected_tools", [])]
    expected_keywords = case.get("expected_keywords", [])
    expected_rejection = case.get("expected_rejection", False)
    bucket = case.get("bucket", _infer_bucket(case))
    dimensions = _case_dimensions(case)

    from uuid import uuid4
    request_id = str(uuid4())
    started = time.perf_counter()

    try:
        if case.get("turns"):
            # 多轮：先按 turns 喂入历史，再用最后一条 user 触发
            for turn in case["turns"][:-1]:
                agent.memory.add_message(
                    case["id"], turn["role"], turn["content"], tenant_id="eval"
                )
            query = case["turns"][-1]["content"]
        else:
            query = case["query"]

        chunks = list(agent.execute_stream(
            query, session_id=case["id"], request_id=request_id, tenant_id="eval"
        ))
        answer = get_final_response(chunks)
    except Exception as exc:
        return CaseResult(
            id=case["id"], passed=False, tool_recall=0.0, keyword_recall=0.0,
            parameter_accuracy=0.0, citation_validity=0.0, artifact_saved=False,
            rejected=None, bucket=bucket, **dimensions, latency_ms=_elapsed_ms(started),
            error_type="exception", error=str(exc),
            detail={"trace": traceback.format_exc()[-500:]},
        )

    rejected = answer.startswith("请求未执行") or "请求未执行" in answer

    if expected_rejection:
        return CaseResult(
            id=case["id"], passed=rejected, tool_recall=1.0 if rejected else 0.0,
            keyword_recall=1.0, rejected=rejected, bucket=bucket,
            **dimensions,
            latency_ms=_elapsed_ms(started),
            error_type=None if rejected else "expected_rejection_not_triggered",
            detail={"answer_preview": answer[:120]},
        )

    actual_tools = _tools_actually_called(request_id)
    if expected_tools:
        hits = sum(1 for tool in expected_tools if tool in actual_tools)
        tool_recall = hits / len(expected_tools)
    else:
        tool_recall = 1.0

    if expected_keywords:
        kw_hits = sum(1 for kw in expected_keywords if kw in answer)
        keyword_recall = kw_hits / len(expected_keywords)
    else:
        keyword_recall = 1.0

    passed = tool_recall >= 0.5 and keyword_recall >= 0.5

    return CaseResult(
        id=case["id"], passed=passed, tool_recall=tool_recall,
        keyword_recall=keyword_recall, rejected=False, bucket=bucket,
        **dimensions,
        latency_ms=_elapsed_ms(started),
        error_type=None if passed else _failure_type(tool_recall, keyword_recall),
        detail={
            "actual_tools": actual_tools,
            "expected_tools": expected_tools,
            "answer_preview": answer[:200],
        },
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="evals/agent_golden.jsonl")
    parser.add_argument("--smoke", action="store_true", help="只跑前 N 条，N 由 CI_SMOKE_LIMIT 控制")
    parser.add_argument("--smoke-limit", type=int,
                        default=int(os.getenv("CI_SMOKE_LIMIT", "3")))
    parser.add_argument("--quiet", action="store_true")
    parser.add_argument("--report", help="写一份机读 JSON 报告到该路径")
    parser.add_argument("--mode", choices=["react", "harness"], default="harness")
    parser.add_argument("--offline", action="store_true",
                        help="使用 golden 内 mock_result 跑完整、确定性的 AgentRunner")
    parser.add_argument("--dry-run", action="store_true",
                        help="不实际跑 Agent，只校验 golden 文件格式（CI 默认）")
    parser.add_argument("--gate", action="store_true", help="启用质量门禁，未达阈值返回非 0")
    parser.add_argument("--gate-config", default=str(DEFAULT_GATE_CONFIG))
    parser.add_argument("--gate-profile", choices=["offline_fixture", "online"])
    parser.add_argument("--min-pass-rate", type=float)
    parser.add_argument("--min-tool-recall", type=float)
    parser.add_argument("--min-keyword-recall", type=float)
    parser.add_argument("--min-parameter-accuracy", type=float)
    parser.add_argument("--min-citation-validity", type=float)
    parser.add_argument("--min-standard-tool-recall", type=float)
    parser.add_argument("--min-high-risk-pass-rate", type=float)
    parser.add_argument("--min-high-risk-tool-recall", type=float)
    parser.add_argument(
        "--max-offline-harness-p95-ms",
        "--max-p95-latency-ms",
        dest="max_offline_harness_p95_ms",
        type=float,
    )
    parser.add_argument("--max-avg-cost", type=float)
    parser.add_argument("--min-case-count", type=int)
    parser.add_argument("--baseline", help="批准的 Agent 评测基线 JSON")
    args = parser.parse_args()

    gate_profile = args.gate_profile or ("offline_fixture" if args.offline else "online")
    try:
        gate_policy = load_gate_profile(args.gate_config, "agent", gate_profile)
        cases = load_golden(Path(args.golden))
        apply_gate_metadata(cases, gate_policy, strict=args.gate)
    except (OSError, ValueError) as exc:
        parser.error(str(exc))
    if args.smoke:
        cases = cases[: args.smoke_limit]

    evaluation_mode = "offline_harness" if args.offline else "online_harness"
    if gate_policy.get("evaluation_mode") != evaluation_mode:
        parser.error(
            f"gate profile {gate_profile!r} expects "
            f"{gate_policy.get('evaluation_mode')!r}, got {evaluation_mode!r}"
        )

    if args.dry_run:
        report = {
            "case_count": len(cases),
            "dry_run": True,
            "mode": args.mode,
            "gate_profile": gate_profile,
            "ids": [c["id"] for c in cases],
        }
        print(json.dumps(report, ensure_ascii=False))
        return

    if args.offline and args.mode != "harness":
        parser.error("--offline only supports --mode harness")
    if args.offline:
        missing_fixtures = [case["id"] for case in cases if "mock_result" not in case]
        if missing_fixtures:
            parser.error(f"offline cases missing mock_result: {missing_fixtures[:5]}")
        agent = _OfflineRunnerFactory(cases)
    elif args.mode == "react":
        from agent.react_agent import ReactAgent
        agent = ReactAgent()
    else:
        from agent.runner import AgentRunner
        agent = AgentRunner()

    started = time.time()
    results: List[CaseResult] = []
    for case in cases:
        result = (
            _evaluate_case(agent, case)
            if args.mode == "react"
            else _evaluate_case_harness(agent, case)
        )
        results.append(result)
        if not args.quiet:
            print(json.dumps(
                {"id": result.id, "passed": result.passed,
                 "tool_recall": round(result.tool_recall, 2),
                 "keyword_recall": round(result.keyword_recall, 2),
                 "error": result.error,
                 "answer_preview": result.detail.get("answer_preview", "")[:80]},
                ensure_ascii=False))

    aggregate = _summarize_results(results)
    aggregate["duration_s"] = round(time.time() - started, 2)
    evaluation_runtime = {
        "scope": (
            "offline_harness_control_plane" if args.offline else "online_evaluation_case"
        ),
        "p50_ms": _percentile([r.latency_ms for r in results], 50),
        "p95_ms": _percentile([r.latency_ms for r in results], 95),
        "is_end_to_end_performance_gate": False,
    }
    offline_harness_latency = evaluation_runtime if args.offline else None
    cost = _summarize_cost(results)
    case_payload = [r.__dict__ for r in results]
    buckets = _summarize_buckets(results)
    risk_tiers = _summarize_risk_tiers(results)
    applicable_case_counts = {
        "tool": aggregate["tool_case_count"],
        "keyword": aggregate["keyword_case_count"],
        "parameter": aggregate["parameter_case_count"],
        "citation": aggregate["citation_case_count"],
        "artifact": aggregate["artifact_case_count"],
    }

    thresholds = EvalThresholds(
        min_pass_rate=_setting(
            args.min_pass_rate,
            gate_policy,
            "metrics.pass_rate.minimum",
            0.85,
            "AGENT_EVAL_MIN_PASS_RATE",
        ),
        min_tool_recall=_setting(
            args.min_tool_recall,
            gate_policy,
            "metrics.tool_recall.minimum",
            0.75,
            "AGENT_EVAL_MIN_TOOL_RECALL",
        ),
        min_keyword_recall=_setting(
            args.min_keyword_recall,
            gate_policy,
            "metrics.keyword_recall.minimum",
            0.75,
            "AGENT_EVAL_MIN_KEYWORD_RECALL",
        ),
        min_parameter_accuracy=_setting(
            args.min_parameter_accuracy,
            gate_policy,
            "metrics.parameter_accuracy.minimum",
            0.9,
        ),
        min_citation_validity=_setting(
            args.min_citation_validity,
            gate_policy,
            "metrics.citation_validity.minimum",
            0.9,
        ),
        min_standard_tool_recall=_setting(
            args.min_standard_tool_recall,
            gate_policy,
            "standard_tool_recall.minimum",
            0.9,
        ),
        min_high_risk_pass_rate=_setting(
            args.min_high_risk_pass_rate,
            gate_policy,
            "hard_constraints.high_risk_pass_rate.minimum",
            1.0,
        ),
        min_high_risk_tool_recall=_setting(
            args.min_high_risk_tool_recall,
            gate_policy,
            "hard_constraints.high_risk_tool_recall.minimum",
            1.0,
        ),
        min_high_risk_parameter_accuracy=_setting(
            None,
            gate_policy,
            "hard_constraints.high_risk_parameter_accuracy.minimum",
            1.0,
        ),
        min_high_risk_citation_validity=_setting(
            None,
            gate_policy,
            "hard_constraints.high_risk_citation_validity.minimum",
            1.0,
        ),
        min_case_count=int(
            args.min_case_count
            if args.min_case_count is not None
            else gate_policy.get("minimum_case_count", 1)
        ),
        minimum_bucket_case_counts={
            str(name): int(count)
            for name, count in (
                gate_policy.get("minimum_bucket_case_counts") or {}
            ).items()
        },
        minimum_risk_case_counts={
            str(name): int(count)
            for name, count in (
                gate_policy.get("minimum_risk_case_counts") or {}
            ).items()
        },
        minimum_applicable_case_counts={
            str(name): int(count)
            for name, count in (
                gate_policy.get("minimum_applicable_case_counts") or {}
            ).items()
        },
        max_offline_harness_p95_ms=_optional_setting(
            args.max_offline_harness_p95_ms,
            gate_policy,
            "offline_harness_latency.p95_ms.maximum",
            "AGENT_EVAL_MAX_P95_LATENCY_MS",
        ),
        max_avg_cost=_setting(
            args.max_avg_cost,
            gate_policy,
            "cost.average_case.maximum",
            0.2,
            "AGENT_EVAL_MAX_AVG_COST",
        ),
    )
    gate_input = {
        "aggregate": aggregate,
        "offline_harness_latency": offline_harness_latency,
        "cost": cost,
        "buckets": buckets,
        "risk_tiers": risk_tiers,
        "applicable_case_counts": applicable_case_counts,
        "cases": case_payload,
    }
    gate_result = EvalGate(
        thresholds
    ).evaluate(gate_input)
    baseline_result = _compare_baseline(
        aggregate, offline_harness_latency, args.baseline
    )
    if gate_policy.get("baseline_required") and baseline_result is None:
        gate_result.failures.append("baseline_required")
    if baseline_result and not baseline_result["passed"]:
        gate_result.failures.extend(baseline_result["failures"])
    gate_result.failures = list(dict.fromkeys(gate_result.failures))
    gate_result.passed = not gate_result.failures
    print(json.dumps(aggregate, ensure_ascii=False))

    report_payload = {
        "schema_version": 2,
        "aggregate": aggregate,
        "evaluation_runtime": evaluation_runtime,
        "offline_harness_latency": offline_harness_latency,
        "cost": cost,
        "mode": args.mode,
        "offline": args.offline,
        "evaluation_mode": evaluation_mode,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_commit": _current_commit(),
        "baseline": baseline_result,
        "gate_policy": gate_policy,
        "buckets": buckets,
        "risk_tiers": risk_tiers,
        "applicable_case_counts": applicable_case_counts,
        "gate": gate_result.__dict__,
        "cases": case_payload,
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(report_payload, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )

    if aggregate["pass_rate"] < float(os.getenv("AGENT_EVAL_PASS_THRESHOLD", "0.0")):
        sys.exit(1)
    if args.gate and not gate_result.passed:
        print(json.dumps({"gate": gate_result.__dict__}, ensure_ascii=False))
        sys.exit(1)


def _avg(seq) -> float:
    seq = list(seq)
    if not seq:
        return 0.0
    return round(sum(float(x) for x in seq) / len(seq), 3)


def _setting(
    explicit: Optional[float],
    policy: Dict[str, Any],
    path: str,
    fallback: float,
    environment_name: Optional[str] = None,
) -> float:
    if explicit is not None:
        return float(explicit)
    configured = policy_value(policy, path)
    if configured is not None:
        return float(configured)
    if environment_name and os.getenv(environment_name) is not None:
        return float(os.environ[environment_name])
    return float(fallback)


def _optional_setting(
    explicit: Optional[float],
    policy: Dict[str, Any],
    path: str,
    environment_name: Optional[str] = None,
) -> Optional[float]:
    if explicit is not None:
        return float(explicit)
    configured = policy_value(policy, path)
    if configured is not None:
        return float(configured)
    if environment_name and os.getenv(environment_name) is not None:
        return float(os.environ[environment_name])
    return None


def _applicable_average(
    results: List[CaseResult], value_field: str, applicable_field: str
) -> tuple[Optional[float], int]:
    applicable = [row for row in results if getattr(row, applicable_field)]
    if not applicable:
        return None, 0
    return _avg(getattr(row, value_field) for row in applicable), len(applicable)


def _summarize_results(results: List[CaseResult]) -> Dict[str, Any]:
    tool_recall, tool_count = _applicable_average(
        results, "tool_recall", "tool_applicable"
    )
    keyword_recall, keyword_count = _applicable_average(
        results, "keyword_recall", "keyword_applicable"
    )
    parameter_accuracy, parameter_count = _applicable_average(
        results, "parameter_accuracy", "parameter_applicable"
    )
    citation_validity, citation_count = _applicable_average(
        results, "citation_validity", "citation_applicable"
    )
    artifact_save_rate, artifact_count = _applicable_average(
        results, "artifact_saved", "artifact_applicable"
    )
    return {
        "case_count": len(results),
        "pass_rate": _avg(row.passed for row in results),
        "tool_recall": tool_recall,
        "tool_case_count": tool_count,
        "keyword_recall": keyword_recall,
        "keyword_case_count": keyword_count,
        "parameter_accuracy": parameter_accuracy,
        "parameter_case_count": parameter_count,
        "citation_validity": citation_validity,
        "citation_case_count": citation_count,
        "artifact_save_rate": artifact_save_rate,
        "artifact_case_count": artifact_count,
    }


def _evaluate_case_harness(runner, case: Dict) -> CaseResult:
    expected_tools = [t.get("name") for t in case.get("expected_tools", [])]
    expected_keywords = case.get("expected_keywords", [])
    expected_rejection = case.get("expected_rejection", False)
    bucket = case.get("bucket", _infer_bucket(case))
    dimensions = _case_dimensions(case)

    from agent.runner import AgentTask
    from uuid import uuid4

    request_id = str(uuid4())
    query = _case_query(case)
    if isinstance(runner, _OfflineRunnerFactory):
        runner = runner.build(case)
    started = time.perf_counter()
    try:
        result = runner.run(
            AgentTask(
                query=query,
                session_id=case["id"],
                request_id=request_id,
                tenant_id=case.get("tenant_id", "eval"),
                user_role=case.get("user_role", "user"),
                scene=case.get(
                    "scene",
                    bucket if bucket in {"rag", "report", "general"} else "general",
                ),
            )
        )
        answer = result.answer
    except Exception as exc:
        return CaseResult(
            id=case["id"], passed=False, tool_recall=0.0, keyword_recall=0.0,
            parameter_accuracy=0.0, citation_validity=0.0, artifact_saved=False,
            rejected=None, bucket=bucket, **dimensions, latency_ms=_elapsed_ms(started),
            error_type="exception", error=str(exc),
            detail={"trace": traceback.format_exc()[-500:], "request_id": request_id},
        )

    rejected = result.state.status in {"rejected", "blocked", "pending_approval"} or (
        answer.startswith("请求未执行") or "请求未执行" in answer
    )
    expected_status = case.get("expected_status", "rejected" if expected_rejection else "completed")
    status_matches = result.state.status == expected_status
    actual_tools = [call.tool_name for call in result.state.tool_calls]
    if expected_tools:
        hits = sum(1 for tool in expected_tools if tool in actual_tools)
        tool_recall = hits / len(expected_tools)
    else:
        tool_recall = 1.0

    parameter_accuracy = _tool_parameter_accuracy(
        case.get("expected_tools", []), result.state.tool_calls
    )

    if expected_keywords:
        kw_hits = sum(1 for kw in expected_keywords if kw in answer)
        keyword_recall = kw_hits / len(expected_keywords)
    else:
        keyword_recall = 1.0
    if expected_status != "completed" and not expected_keywords:
        keyword_recall = 1.0

    verifier_quality = getattr(result.verifier, "quality", {}) if result.verifier else {}
    measured_citation_validity = float(verifier_quality.get("citation_validity", 1.0))
    # A negative citation case is successful when the verifier rejects it.
    # Gate metrics should measure unsafe citations that escaped, not expected rejections.
    citation_validity = (
        1.0
        if expected_status != "completed" and status_matches
        else measured_citation_validity
    )
    artifact_saved = bool(result.artifacts) if case.get("expect_artifact", expected_status == "completed") else True
    passed = (
        status_matches
        and tool_recall >= 1.0
        and keyword_recall >= 1.0
        and parameter_accuracy >= 1.0
        and citation_validity >= 1.0
        and artifact_saved
    )
    if expected_rejection and not rejected:
        passed = False
    return CaseResult(
        id=case["id"], passed=passed, tool_recall=tool_recall,
        keyword_recall=keyword_recall, rejected=rejected,
        parameter_accuracy=parameter_accuracy,
        citation_validity=citation_validity,
        artifact_saved=artifact_saved,
        bucket=bucket,
        **dimensions,
        latency_ms=_elapsed_ms(started),
        error_type=None if passed else _harness_failure_type(
            status_matches, tool_recall, keyword_recall, parameter_accuracy,
            citation_validity, artifact_saved,
        ),
        detail={
            "actual_tools": actual_tools,
            "expected_tools": expected_tools,
            "answer_preview": answer[:200],
            "request_id": request_id,
            "status": result.state.status,
            "expected_status": expected_status,
            "measured_citation_validity": measured_citation_validity,
        },
    )


def _tool_parameter_accuracy(expected_tools, actual_calls) -> float:
    expected_with_args = [tool for tool in expected_tools if tool.get("args") is not None]
    if not expected_with_args:
        return 1.0
    matches = 0
    for expected in expected_with_args:
        for actual in actual_calls:
            if actual.tool_name != expected["name"]:
                continue
            if all(actual.args.get(key) == value for key, value in expected["args"].items()):
                matches += 1
                break
    return matches / len(expected_with_args)


def _harness_failure_type(
    status_matches: bool,
    tool_recall: float,
    keyword_recall: float,
    parameter_accuracy: float,
    citation_validity: float,
    artifact_saved: bool,
) -> str:
    if not status_matches:
        return "status_mismatch"
    if tool_recall < 1.0:
        return "tool_miss"
    if parameter_accuracy < 1.0:
        return "parameter_miss"
    if keyword_recall < 1.0:
        return "keyword_miss"
    if citation_validity < 1.0:
        return "invalid_citation"
    if not artifact_saved:
        return "artifact_missing"
    return "failed"


def _summarize_buckets(results: List[CaseResult]) -> Dict[str, Dict[str, Any]]:
    buckets: Dict[str, List[CaseResult]] = {}
    for result in results:
        buckets.setdefault(result.bucket, []).append(result)
    return {name: _summarize_results(rows) for name, rows in sorted(buckets.items())}


def _summarize_risk_tiers(results: List[CaseResult]) -> Dict[str, Dict[str, Any]]:
    tiers: Dict[str, List[CaseResult]] = {}
    for result in results:
        tiers.setdefault(result.risk_tier, []).append(result)
    return {name: _summarize_results(rows) for name, rows in sorted(tiers.items())}


def _current_commit() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _compare_baseline(
    aggregate: Dict,
    offline_harness_latency: Optional[Dict],
    baseline_path: Optional[str],
):
    if not baseline_path:
        return None
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    expected = baseline.get("aggregate") or {}
    allowed = baseline.get("allowed_regression") or {}
    deltas = {}
    failures = []
    for metric in (
        "pass_rate",
        "tool_recall",
        "keyword_recall",
        "parameter_accuracy",
        "citation_validity",
        "artifact_save_rate",
    ):
        delta = round(float(aggregate.get(metric, 0.0)) - float(expected.get(metric, 0.0)), 4)
        deltas[metric] = delta
        if delta < -float(allowed.get(metric, 0.0)):
            failures.append(f"{metric}_regressed:{delta}")
    baseline_latency = baseline.get(
        "offline_harness_p95_latency_ms", baseline.get("p95_latency_ms")
    )
    if offline_harness_latency is not None and baseline_latency is not None:
        latency_delta = round(
            float(offline_harness_latency.get("p95_ms", 0.0))
            - float(baseline_latency),
            3,
        )
        deltas["offline_harness_p95_latency_ms"] = latency_delta
        latency_tolerance = allowed.get(
            "offline_harness_p95_latency_ms", allowed.get("p95_latency_ms", 0.0)
        )
        if latency_delta > float(latency_tolerance):
            failures.append(f"offline_harness_p95_latency_regressed:{latency_delta}")
    return {
        "passed": not failures,
        "baseline_commit": baseline.get("baseline_commit"),
        "deltas": deltas,
        "failures": failures,
    }


def _summarize_cost(results: List[CaseResult]) -> Dict[str, Any]:
    total_cost = 0.0
    total_tokens = 0
    has_usage = False
    for result in results:
        request_id = result.detail.get("request_id")
        if not request_id:
            continue
        try:
            events = trace_recorder.export_trace(request_id)["events"]
        except KeyError:
            continue
        model_usage_events = [
            event
            for event in events
            if (event.get("metadata") or {}).get("type") == "model_usage"
        ]
        # Compatibility backends report usage on verifier diagnostics.  Use
        # those only when no per-model events exist, otherwise the same model
        # call would be counted twice.
        usage_events = model_usage_events or [
            event
            for event in events
            if (event.get("metadata") or {}).get("type") == "verifier"
        ]
        for event in usage_events:
            metadata = event.get("metadata", {})
            cost = float(metadata.get("cost") or 0.0)
            tokens = int(metadata.get("tokens_in") or 0) + int(metadata.get("tokens_out") or 0)
            total_cost += cost
            total_tokens += tokens
            has_usage = has_usage or cost > 0 or tokens > 0
    if not has_usage:
        return {"avg": 0.0, "mode": "disabled"}
    return {
        "avg": round(total_cost / len(results), 6) if results else 0.0,
        "mode": "estimated",
        "tokens_avg": round(total_tokens / len(results), 3) if results else 0.0,
    }


def _elapsed_ms(started: float) -> float:
    return round((time.perf_counter() - started) * 1000, 3)


def _percentile(values: List[float], percentile: int) -> float:
    if not values:
        return 0.0
    ordered = sorted(values)
    index = min(len(ordered) - 1, round((percentile / 100) * (len(ordered) - 1)))
    return round(ordered[index], 3)


def _infer_bucket(case: Dict) -> str:
    expected_tools = {tool.get("name") for tool in case.get("expected_tools", [])}
    if "fetch_external_data" in expected_tools or "报告" in case.get("query", ""):
        return "report"
    if "rag_summarize" in expected_tools:
        return "rag"
    if expected_tools:
        return "tool"
    if case.get("expected_rejection"):
        return "safety"
    return "general"


def _failure_type(tool_recall: float, keyword_recall: float) -> str:
    if tool_recall < 0.5:
        return "tool_miss"
    if keyword_recall < 0.5:
        return "keyword_miss"
    return "failed"


if __name__ == "__main__":
    main()
