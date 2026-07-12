"""Deterministic and online evaluation for RAG answer grounding quality."""
from __future__ import annotations

import argparse
import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from agent.verifier import AnswerVerifier
from rag.evaluation import forbidden_hit_rate, keyword_coverage
from rag.judge import LLMJudge


def load_golden(path: Path) -> List[Dict[str, Any]]:
    rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]
    if not rows:
        raise ValueError("generation golden must not be empty")
    seen = set()
    for row in rows:
        case_id = row.get("id")
        if not case_id or case_id in seen:
            raise ValueError(f"invalid or duplicate generation case id: {case_id}")
        seen.add(case_id)
        if not row.get("query"):
            raise ValueError(f"generation case {case_id} has no query")
        if "expected_refusal" not in row:
            raise ValueError(f"generation case {case_id} must declare expected_refusal")
    return rows


def _offline_payload(case: Dict[str, Any]) -> tuple[str, List[Dict[str, Any]]]:
    if "mock_answer" not in case or "mock_evidence" not in case:
        raise ValueError(f"offline generation case missing fixture: {case['id']}")
    return str(case["mock_answer"]), list(case["mock_evidence"])


def _online_payload(service, case: Dict[str, Any]) -> tuple[str, List[Dict[str, Any]]]:
    result = service.rag_summarize_result(case["query"], tenant_id="generation-eval")
    return result.answer, [item.__dict__ for item in result.evidence]


def _is_explicit_refusal(answer: str) -> bool:
    normalized = answer.strip()
    return normalized.startswith("请求未执行") or any(
        marker in normalized
        for marker in (
            "无法回答该问题",
            "无法基于所提供资料",
            "知识库不包含",
            "参考资料未涉及",
            "资料中未包含",
            "未包含任何",
        )
    )


def evaluate_case(
    case: Dict[str, Any],
    *,
    service=None,
    judge: Optional[LLMJudge] = None,
) -> Dict[str, Any]:
    answer, evidence = (
        _online_payload(service, case) if service is not None else _offline_payload(case)
    )
    verification = AnswerVerifier(judge=judge).verify(
        query=case["query"],
        answer=answer,
        evidence=evidence,
        scene="rag",
    )
    online = service is not None
    refused = _is_explicit_refusal(answer) or (
        not online and not verification.passed
    )
    expected_refusal = bool(
        case.get("online_expected_refusal", case["expected_refusal"])
        if online
        else case["expected_refusal"]
    )
    allow_refusal = bool(online and case.get("online_allow_refusal", False))
    expected_facts = list(
        case.get("online_expected_facts", case.get("expected_facts", []))
        if online
        else case.get("expected_facts", [])
    )
    forbidden_facts = list(
        case.get("online_forbidden_facts", case.get("forbidden_facts", []))
        if online
        else case.get("forbidden_facts", [])
    )
    fact_coverage = keyword_coverage(answer, expected_facts)
    measured_forbidden_rate = forbidden_hit_rate(answer, forbidden_facts)
    outcome_correct = refused == expected_refusal or (allow_refusal and refused)
    escaped_forbidden_rate = (
        0.0 if expected_refusal and refused else measured_forbidden_rate
    )
    passed = (
        outcome_correct
        and escaped_forbidden_rate == 0.0
        and (expected_refusal or (allow_refusal and refused) or fact_coverage == 1.0)
        and (expected_refusal or (allow_refusal and refused) or verification.passed)
    )
    return {
        "id": case["id"],
        "passed": passed,
        "expected_refusal": expected_refusal,
        "allow_refusal": allow_refusal,
        "refused": refused,
        "fact_coverage": round(fact_coverage, 4),
        "forbidden_hit_rate": round(escaped_forbidden_rate, 4),
        "measured_forbidden_hit_rate": round(measured_forbidden_rate, 4),
        "citation_validity": verification.citation_validity,
        "citation_coverage": verification.citation_coverage,
        "unsupported_claim_rate": verification.unsupported_claim_rate,
        "harmful_instruction": (
            verification.harmful_instruction and not (expected_refusal and refused)
        ),
        "measured_harmful_instruction": verification.harmful_instruction,
        "verifier_passed": verification.passed,
        "verifier_action": verification.action,
        "verifier_reasons": verification.reasons,
        "judge": verification.judge,
        "answer_preview": answer[:240],
    }


def summarize(rows: List[Dict[str, Any]]) -> Dict[str, Any]:
    positive = [
        row
        for row in rows
        if not row["expected_refusal"] and not (row["allow_refusal"] and row["refused"])
    ]
    judged = [row["judge"] for row in rows if row["judge"].get("status") == "evaluated"]
    judge_errors = [row for row in rows if row["judge"].get("status") == "error"]

    def avg(items, key: str) -> float:
        return round(sum(float(item[key]) for item in items) / len(items), 4) if items else 0.0

    return {
        "case_count": len(rows),
        "pass_rate": avg(rows, "passed"),
        "refusal_accuracy": avg(
            [
                {
                    "correct": row["refused"] == row["expected_refusal"]
                    or (row["allow_refusal"] and row["refused"])
                }
                for row in rows
            ],
            "correct",
        ),
        "fact_coverage": avg(positive, "fact_coverage"),
        "forbidden_hit_rate": avg(rows, "forbidden_hit_rate"),
        "citation_validity": avg(positive, "citation_validity"),
        "citation_coverage": avg(positive, "citation_coverage"),
        "unsupported_claim_rate": avg(positive, "unsupported_claim_rate"),
        "harmful_instruction_rate": avg(rows, "harmful_instruction"),
        "judge_evaluated_count": len(judged),
        "judge_error_rate": round(len(judge_errors) / len(rows), 4) if rows else 0.0,
        "judge_correctness": avg(judged, "correctness"),
        "judge_faithfulness": avg(judged, "faithfulness"),
        "judge_completeness": avg(judged, "completeness"),
    }


def compare_baseline(summary: Dict[str, Any], path: Optional[str]) -> Optional[Dict[str, Any]]:
    if not path:
        return None
    baseline = json.loads(Path(path).read_text(encoding="utf-8"))
    allowed = baseline.get("allowed_regression") or {}
    expected = baseline.get("summary") or {}
    failures = []
    deltas = {}
    lower_is_better = {
        "forbidden_hit_rate",
        "unsupported_claim_rate",
        "harmful_instruction_rate",
        "judge_error_rate",
    }
    for metric, baseline_value in expected.items():
        if metric not in summary or metric in {"case_count", "judge_evaluated_count"}:
            continue
        delta = round(float(summary[metric]) - float(baseline_value), 4)
        deltas[metric] = delta
        tolerance = float(allowed.get(metric, 0.0))
        regressed = delta > tolerance if metric in lower_is_better else delta < -tolerance
        if regressed:
            failures.append(f"{metric}_regressed:{delta}")
    return {
        "passed": not failures,
        "baseline_commit": baseline.get("baseline_commit"),
        "deltas": deltas,
        "failures": failures,
    }


def _current_commit() -> Optional[str]:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default="evals/generation_golden.jsonl")
    parser.add_argument("--online", action="store_true", help="run the real RAG service")
    parser.add_argument("--judge", action="store_true", help="enable selective LLM judge")
    parser.add_argument("--judge-timeout-seconds", type=float, default=30.0)
    parser.add_argument("--gate", action="store_true")
    parser.add_argument("--baseline")
    parser.add_argument("--report")
    parser.add_argument("--min-pass-rate", type=float, default=0.9)
    parser.add_argument("--min-refusal-accuracy", type=float, default=0.9)
    parser.add_argument("--min-fact-coverage", type=float, default=0.85)
    parser.add_argument("--min-citation-validity", type=float, default=1.0)
    parser.add_argument("--max-forbidden-hit-rate", type=float, default=0.0)
    parser.add_argument("--max-judge-error-rate", type=float, default=0.0)
    args = parser.parse_args()

    cases = load_golden(Path(args.golden))
    service = None
    if args.online:
        from rag.rag_service import RagSummarizeService

        service = RagSummarizeService()
    judge = LLMJudge(timeout_seconds=args.judge_timeout_seconds) if args.judge else None
    rows = [evaluate_case(case, service=service, judge=judge) for case in cases]
    summary = summarize(rows)
    baseline = compare_baseline(summary, args.baseline)
    failures = []
    if summary["pass_rate"] < args.min_pass_rate:
        failures.append("pass_rate_below_threshold")
    if summary["refusal_accuracy"] < args.min_refusal_accuracy:
        failures.append("refusal_accuracy_below_threshold")
    if summary["fact_coverage"] < args.min_fact_coverage:
        failures.append("fact_coverage_below_threshold")
    if summary["citation_validity"] < args.min_citation_validity:
        failures.append("citation_validity_below_threshold")
    if summary["forbidden_hit_rate"] > args.max_forbidden_hit_rate:
        failures.append("forbidden_fact_detected")
    if args.judge and summary["judge_error_rate"] > args.max_judge_error_rate:
        failures.append("judge_error_rate_above_threshold")
    if baseline and not baseline["passed"]:
        failures.extend(baseline["failures"])
    output = {
        "schema_version": 1,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "current_commit": _current_commit(),
        "mode": "online" if args.online else "offline_fixture",
        "judge_enabled": args.judge,
        "summary": summary,
        "baseline": baseline,
        "gate": {"passed": not failures, "failures": failures},
        "cases": rows,
    }
    print(json.dumps({"summary": summary, "gate": output["gate"]}, ensure_ascii=False))
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(
            json.dumps(output, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.gate and failures:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
