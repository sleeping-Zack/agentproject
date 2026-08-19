from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

from machine_eval.judge import AgentRubricJudge
from machine_eval.pipeline import MachineEvalPipeline
from scripts.prepare_human_eval_batch import load_dataset, load_jsonl


def build_evaluation_items(
    dataset: Mapping[str, Mapping[str, Any]],
    run_results: Sequence[Mapping[str, Any]],
    *,
    splits: Sequence[str],
) -> list[Dict[str, Any]]:
    selected_splits = set(splits)
    unknown = sorted(
        str(result.get("case_id") or "")
        for result in run_results
        if str(result.get("case_id") or "") not in dataset
    )
    if unknown:
        raise ValueError(f"run results contain unknown case_id values: {unknown[:5]}")
    items = []
    for result in run_results:
        case = dataset[str(result["case_id"])]
        if case.get("split") not in selected_splits:
            continue
        answer = result.get("agent_answer")
        if not isinstance(answer, str):
            raise ValueError(f"{case['case_id']}: agent_answer must be a string")
        labels = case.get("labels") or {}
        performance = result.get("performance") or {}
        model_metadata = result.get("model_metadata") or {}
        items.append(
            {
                "case_id": case["case_id"],
                "family_id": case.get("family_id") or case["case_id"],
                "dataset_version": case.get("dataset_version"),
                "split": case.get("split"),
                "category": case.get("category"),
                "scene": case.get("scene"),
                "risk_level": labels.get("risk_level"),
                "capability_tags": labels.get("capability_tags") or [],
                "query": case.get("query"),
                "turns": case.get("turns") or [],
                "context": case.get("context") or {},
                "references": case.get("references") or [],
                "expected": case.get("expected") or {},
                "agent_answer": answer,
                "status": result.get("status"),
                "trace": result.get("trace") or [],
                "tool_calls": result.get("tool_calls") or [],
                "evidence": result.get("evidence") or [],
                "citations": result.get("citations") or [],
                "artifacts": result.get("artifacts") or [],
                "policy_context": result.get("policy_context") or {},
                "model_metadata": model_metadata,
                "latency_ms": result.get("latency_ms", performance.get("latency_ms")),
                "estimated_cost": result.get(
                    "estimated_cost", performance.get("estimated_cost")
                ),
                "cost_mode": result.get("cost_mode", performance.get("cost_mode")),
                "tokens_in": result.get("tokens_in", performance.get("tokens_in")),
                "tokens_out": result.get("tokens_out", performance.get("tokens_out")),
            }
        )
    if not items:
        raise ValueError("no run results matched the requested dataset splits")
    items.sort(key=lambda item: item["case_id"])
    return items


def _load_json(path: str | None) -> Dict[str, Any] | None:
    if not path:
        return None
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def _current_commit() -> str | None:
    try:
        result = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            capture_output=True,
            check=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return result.stdout.strip() or None


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _case_set_sha256(items: Sequence[Mapping[str, Any]]) -> str:
    payload = "\n".join(sorted(str(item["case_id"]) for item in items))
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def _dataset_sha256(dataset_dir: Path) -> str | None:
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        return None
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    value = manifest.get("dataset_sha256")
    return str(value) if isinstance(value, str) and value else None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run hybrid deterministic + Rubric-aligned Agent machine evaluation."
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("evals/agent_quality/v1"))
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["dev", "test", "regression"],
        default=["dev"],
    )
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-id", default="configured-chat-model")
    parser.add_argument("--judge-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--human-export")
    parser.add_argument("--baseline")
    parser.add_argument("--run-id")
    parser.add_argument("--variant", default="candidate")
    parser.add_argument("--report", type=Path)
    parser.add_argument("--gate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset_dir)
    run_results = load_jsonl(args.results)
    items = build_evaluation_items(dataset, run_results, splits=args.splits)
    judge = (
        AgentRubricJudge(
            timeout_seconds=args.judge_timeout_seconds,
            judge_id=args.judge_id,
        )
        if args.judge
        else None
    )
    report = MachineEvalPipeline(judge=judge).evaluate(
        items,
        human_export=_load_json(args.human_export),
        baseline=_load_json(args.baseline),
        run_metadata={
            "run_id": args.run_id or args.results.stem,
            "variant": args.variant,
            "current_commit": _current_commit(),
            "result_file": str(args.results),
            "result_sha256": _sha256(args.results),
            "dataset_sha256": _dataset_sha256(args.dataset_dir),
            "case_set_sha256": _case_set_sha256(items),
            "splits": args.splits,
        },
    )
    print(
        json.dumps(
            {
                "summary": report["summary"],
                "human_alignment": report["human_alignment"],
                "production_gate": report["production_gate"],
            },
            ensure_ascii=False,
        )
    )
    if args.report:
        args.report.parent.mkdir(parents=True, exist_ok=True)
        args.report.write_text(
            json.dumps(report, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    if args.gate and report["production_gate"].get("passed") is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
