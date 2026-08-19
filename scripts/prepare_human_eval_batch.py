from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Sequence

from human_eval.service import HumanEvalService
from services.human_eval_store import SQLiteHumanEvalStore


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    records = []
    seen = set()
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_number}: invalid JSON") from exc
            case_id = str(record.get("case_id") or "").strip()
            if not case_id:
                raise ValueError(f"{path}:{line_number}: case_id is required")
            if case_id in seen:
                raise ValueError(f"{path}:{line_number}: duplicate case_id {case_id}")
            seen.add(case_id)
            records.append(record)
    return records


def load_dataset(dataset_dir: Path) -> Dict[str, Dict[str, Any]]:
    cases: Dict[str, Dict[str, Any]] = {}
    paths = sorted(dataset_dir.glob("*.jsonl"))
    if not paths:
        raise ValueError(f"no JSONL dataset files found in {dataset_dir}")
    for path in paths:
        for case in load_jsonl(path):
            case_id = case["case_id"]
            if case_id in cases:
                raise ValueError(f"duplicate dataset case_id {case_id}")
            cases[case_id] = case
    return cases


def build_items(
    dataset: Mapping[str, Mapping[str, Any]],
    run_results: Iterable[Mapping[str, Any]],
    *,
    splits: Sequence[str],
    sample_size: int | None,
    seed: int,
) -> List[Dict[str, Any]]:
    results = list(run_results)
    unknown = sorted(
        str(result.get("case_id") or "")
        for result in results
        if str(result.get("case_id") or "") not in dataset
    )
    if unknown:
        raise ValueError(f"run results contain unknown case_id values: {unknown[:5]}")
    selected_splits = set(splits)
    matched = [
        (dataset[str(result["case_id"])], result)
        for result in results
        if dataset[str(result["case_id"])].get("split") in selected_splits
    ]
    if not matched:
        raise ValueError("no run results matched the requested dataset splits")
    if sample_size is not None:
        if sample_size <= 0:
            raise ValueError("sample_size must be positive")
        if sample_size > len(matched):
            raise ValueError(
                f"sample_size {sample_size} exceeds {len(matched)} matched run results"
            )
        matched = random.Random(seed).sample(matched, sample_size)
    matched.sort(key=lambda pair: str(pair[0]["case_id"]))

    items = []
    for case, result in matched:
        answer = result.get("agent_answer")
        if not isinstance(answer, str):
            raise ValueError(f"{case['case_id']}: agent_answer must be a string")
        trace = result.get("trace") or []
        evidence = result.get("evidence") or []
        if not isinstance(trace, list) or not isinstance(evidence, list):
            raise ValueError(f"{case['case_id']}: trace and evidence must be arrays")
        labels = case.get("labels") or {}
        items.append(
            {
                "case_id": case["case_id"],
                "query": case["query"],
                "turns": case.get("turns") or [],
                "scene": case.get("scene") or "",
                "risk_level": labels.get("risk_level") or "",
                "capability_tags": labels.get("capability_tags") or [],
                "context": case.get("context") or {},
                "agent_answer": answer,
                "trace": trace,
                "approval_records": result.get("approval_records") or [],
                "planner_steps": result.get("planner_steps") or [],
                "tool_calls": result.get("tool_calls") or [],
                "evidence": evidence,
                "references": case.get("references") or [],
                "policy_context": result.get("policy_context") or {},
                "oracle": {
                    "expected": case.get("expected") or {},
                    "provenance": case.get("provenance") or {},
                    "run_metadata": result.get("model_metadata") or {},
                },
            }
        )
    return items


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create a double-blind human-evaluation batch from run-result JSONL."
    )
    parser.add_argument(
        "--dataset-dir",
        type=Path,
        default=Path("evals/agent_quality/v1"),
    )
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("storage/human_eval.db"))
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--created-by", required=True)
    parser.add_argument("--name", required=True)
    parser.add_argument("--dataset-version", default="agent-quality-v1")
    parser.add_argument("--reviewers", nargs="+", required=True)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["dev", "test", "regression"],
        default=["dev"],
    )
    parser.add_argument("--sample-size", type=int)
    parser.add_argument("--qc-rate", type=float, default=0.10)
    parser.add_argument("--seed", type=int, default=20260811)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    dataset = load_dataset(args.dataset_dir)
    run_results = load_jsonl(args.results)
    items = build_items(
        dataset,
        run_results,
        splits=args.splits,
        sample_size=args.sample_size,
        seed=args.seed,
    )
    service = HumanEvalService(SQLiteHumanEvalStore(args.db))
    progress = service.create_batch(
        tenant_id=args.tenant,
        created_by=args.created_by,
        name=args.name,
        dataset_version=args.dataset_version,
        items=items,
        reviewer_ids=args.reviewers,
        qc_rate=args.qc_rate,
        seed=args.seed,
    )
    print(json.dumps(progress, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
