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
    latency_ms: float = 0.0
    error_type: Optional[str] = None
    error: Optional[str] = None
    detail: Dict[str, Any] = field(default_factory=dict)


def load_golden(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


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
        answer = "".join(chunks)
    except Exception as exc:
        return CaseResult(
            id=case["id"], passed=False, tool_recall=0.0, keyword_recall=0.0,
            rejected=None, bucket=bucket, latency_ms=_elapsed_ms(started),
            error_type="exception", error=str(exc),
            detail={"trace": traceback.format_exc()[-500:]},
        )

    rejected = answer.startswith("请求未执行") or "请求未执行" in answer

    if expected_rejection:
        return CaseResult(
            id=case["id"], passed=rejected, tool_recall=1.0 if rejected else 0.0,
            keyword_recall=1.0, rejected=rejected, bucket=bucket,
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
    parser.add_argument("--min-pass-rate", type=float,
                        default=float(os.getenv("AGENT_EVAL_MIN_PASS_RATE", "0.85")))
    parser.add_argument("--min-tool-recall", type=float,
                        default=float(os.getenv("AGENT_EVAL_MIN_TOOL_RECALL", "0.75")))
    parser.add_argument("--min-keyword-recall", type=float,
                        default=float(os.getenv("AGENT_EVAL_MIN_KEYWORD_RECALL", "0.75")))
    parser.add_argument("--max-p95-latency-ms", type=float,
                        default=float(os.getenv("AGENT_EVAL_MAX_P95_LATENCY_MS", "5000")))
    parser.add_argument("--max-avg-cost", type=float,
                        default=float(os.getenv("AGENT_EVAL_MAX_AVG_COST", "0.2")))
    parser.add_argument("--min-parameter-accuracy", type=float, default=0.9)
    parser.add_argument("--min-citation-validity", type=float, default=0.9)
    parser.add_argument("--min-case-count", type=int, default=1)
    parser.add_argument("--baseline", help="批准的 Agent 评测基线 JSON")
    args = parser.parse_args()

    cases = load_golden(Path(args.golden))
    if args.smoke:
        cases = cases[: args.smoke_limit]

    if args.dry_run:
        report = {
            "case_count": len(cases),
            "dry_run": True,
            "mode": args.mode,
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

    aggregate = {
        "case_count": len(results),
        "pass_rate": _avg(r.passed for r in results),
        "tool_recall": _avg(r.tool_recall for r in results),
        "keyword_recall": _avg(r.keyword_recall for r in results),
        "parameter_accuracy": _avg(r.parameter_accuracy for r in results),
        "citation_validity": _avg(r.citation_validity for r in results),
        "artifact_save_rate": _avg(r.artifact_saved for r in results),
        "duration_s": round(time.time() - started, 2),
    }
    latency = {
        "p50_ms": _percentile([r.latency_ms for r in results], 50),
        "p95_ms": _percentile([r.latency_ms for r in results], 95),
    }
    cost = _summarize_cost(results)
    case_payload = [r.__dict__ for r in results]
    gate_result = EvalGate(
        EvalThresholds(
            min_pass_rate=args.min_pass_rate,
            min_tool_recall=args.min_tool_recall,
            min_keyword_recall=args.min_keyword_recall,
            max_p95_latency_ms=args.max_p95_latency_ms,
            max_avg_cost=args.max_avg_cost,
        )
    ).evaluate({
        "aggregate": aggregate,
        "latency": latency,
        "cost": cost,
        "cases": case_payload,
    })
    baseline_result = _compare_baseline(aggregate, latency, args.baseline)
    if aggregate["case_count"] < args.min_case_count:
        gate_result.failures.append("case_count_below_threshold")
    if aggregate["parameter_accuracy"] < args.min_parameter_accuracy:
        gate_result.failures.append("parameter_accuracy_below_threshold")
    if aggregate["citation_validity"] < args.min_citation_validity:
        gate_result.failures.append("citation_validity_below_threshold")
    if baseline_result and not baseline_result["passed"]:
        gate_result.failures.extend(baseline_result["failures"])
    gate_result.failures = list(dict.fromkeys(gate_result.failures))
    gate_result.passed = not gate_result.failures
    print(json.dumps(aggregate, ensure_ascii=False))

    report_payload = {
        "aggregate": aggregate,
        "latency": latency,
        "cost": cost,
        "mode": args.mode,
        "offline": args.offline,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_commit": _current_commit(),
        "baseline": baseline_result,
        "buckets": _summarize_buckets(results),
        "gate": gate_result.__dict__,
        "cases": case_payload,
    }
    if args.report:
        Path(args.report).write_text(
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


def _evaluate_case_harness(runner, case: Dict) -> CaseResult:
    expected_tools = [t.get("name") for t in case.get("expected_tools", [])]
    expected_keywords = case.get("expected_keywords", [])
    expected_rejection = case.get("expected_rejection", False)
    bucket = case.get("bucket", _infer_bucket(case))

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
            rejected=None, bucket=bucket, latency_ms=_elapsed_ms(started),
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
    citation_validity = float(verifier_quality.get("citation_validity", 1.0))
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


def _summarize_buckets(results: List[CaseResult]) -> Dict[str, Dict[str, float]]:
    buckets: Dict[str, List[CaseResult]] = {}
    for result in results:
        buckets.setdefault(result.bucket, []).append(result)
    return {
        name: {
            "case_count": len(rows),
            "pass_rate": _avg(row.passed for row in rows),
            "tool_recall": _avg(row.tool_recall for row in rows),
            "keyword_recall": _avg(row.keyword_recall for row in rows),
        }
        for name, rows in sorted(buckets.items())
    }


def _current_commit() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"], check=True, capture_output=True,
            text=True, timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def _compare_baseline(aggregate: Dict, latency: Dict, baseline_path: Optional[str]):
    if not baseline_path:
        return None
    baseline = json.loads(Path(baseline_path).read_text(encoding="utf-8"))
    expected = baseline.get("aggregate") or {}
    allowed = baseline.get("allowed_regression") or {}
    deltas = {}
    failures = []
    for metric in ("pass_rate", "tool_recall", "keyword_recall", "parameter_accuracy"):
        delta = round(float(aggregate.get(metric, 0.0)) - float(expected.get(metric, 0.0)), 4)
        deltas[metric] = delta
        if delta < -float(allowed.get(metric, 0.0)):
            failures.append(f"{metric}_regressed:{delta}")
    latency_delta = round(
        float(latency.get("p95_ms", 0.0)) - float(baseline.get("p95_latency_ms", 0.0)), 3
    )
    deltas["p95_latency_ms"] = latency_delta
    if latency_delta > float(allowed.get("p95_latency_ms", 0.0)):
        failures.append(f"p95_latency_regressed:{latency_delta}")
    return {
        "passed": not failures,
        "baseline_commit": baseline.get("baseline_commit"),
        "deltas": deltas,
        "failures": failures,
    }


def _summarize_cost(results: List[CaseResult]) -> Dict[str, Any]:
    costs: List[float] = []
    token_totals: List[int] = []
    for result in results:
        request_id = result.detail.get("request_id")
        if not request_id:
            continue
        try:
            events = trace_recorder.export_trace(request_id)["events"]
        except KeyError:
            continue
        for event in events:
            metadata = event.get("metadata", {})
            cost = float(metadata.get("cost") or 0.0)
            tokens = int(metadata.get("tokens_in") or 0) + int(metadata.get("tokens_out") or 0)
            if cost > 0:
                costs.append(cost)
            if tokens > 0:
                token_totals.append(tokens)
    if not costs and not token_totals:
        return {"avg": 0.0, "mode": "disabled"}
    return {
        "avg": round(sum(costs) / len(results), 6) if results else 0.0,
        "mode": "estimated",
        "tokens_avg": round(sum(token_totals) / len(results), 3) if results else 0.0,
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
