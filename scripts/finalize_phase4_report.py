from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

from human_eval.service import HumanEvalService
from machine_eval.judge import AgentRubricJudge
from machine_eval.pipeline import MachineEvalPipeline
from scripts.evaluate_agent_quality import (
    _case_set_sha256,
    _current_commit,
    _dataset_sha256,
    _load_json,
    _sha256,
    build_evaluation_items,
)
from scripts.prepare_human_eval_batch import load_dataset, load_jsonl
from services.human_eval_store import SQLiteHumanEvalStore


def _validate_human_export(
    human_export: Mapping[str, Any],
    items: Sequence[Mapping[str, Any]],
) -> None:
    if human_export.get("batch_status") != "closed":
        raise ValueError("human-evaluation batch must be closed before phase-four reporting")
    pending = int(human_export.get("pending_final_count") or 0)
    if pending:
        raise ValueError(
            f"human-evaluation batch has {pending} unresolved final label(s)"
        )

    item_by_case = {str(item["case_id"]): item for item in items}
    records = human_export.get("records") or []
    record_case_ids = {str(record.get("case_id") or "") for record in records}
    if record_case_ids != set(item_by_case):
        missing_from_human = sorted(set(item_by_case) - record_case_ids)
        missing_from_results = sorted(record_case_ids - set(item_by_case))
        raise ValueError(
            "human-evaluation and run-result case sets must match exactly; "
            f"missing_from_human={missing_from_human[:5]}, "
            f"missing_from_results={missing_from_results[:5]}"
        )
    for record in records:
        case_id = str(record.get("case_id") or "")
        payload = record.get("payload") or {}
        item = item_by_case[case_id]
        for field in ("query", "agent_answer"):
            if payload.get(field) != item.get(field):
                raise ValueError(
                    f"{case_id}: human-evaluation {field} does not match the run results"
                )


def _human_export_path(report_path: Path, requested: Path | None) -> Path:
    return requested or report_path.with_name(
        f"{report_path.stem}.human-export.json"
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export a closed human-evaluation batch and generate its phase-four "
            "machine-evaluation report."
        )
    )
    parser.add_argument("--dataset-dir", type=Path, default=Path("evals/agent_quality/v1"))
    parser.add_argument("--results", type=Path, required=True)
    parser.add_argument("--db", type=Path, default=Path("storage/human_eval.db"))
    parser.add_argument("--batch-id", required=True)
    parser.add_argument("--tenant", required=True)
    parser.add_argument("--variant", required=True)
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--human-export-out", type=Path)
    parser.add_argument(
        "--splits",
        nargs="+",
        choices=["dev", "test", "regression"],
        default=["dev"],
    )
    parser.add_argument("--judge", action="store_true")
    parser.add_argument("--judge-id")
    parser.add_argument("--judge-timeout-seconds", type=float, default=45.0)
    parser.add_argument("--baseline")
    parser.add_argument("--run-id")
    parser.add_argument("--gate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if args.judge and not args.judge_id:
        raise ValueError("--judge requires an explicit pinned --judge-id")
    service = HumanEvalService(SQLiteHumanEvalStore(args.db))
    progress = service.progress(batch_id=args.batch_id, tenant_id=args.tenant)
    if progress["status"] != "closed":
        raise ValueError(
            f"human-evaluation batch must be closed; current status is {progress['status']!r}"
        )
    human_export = service.export(batch_id=args.batch_id, tenant_id=args.tenant)

    dataset = load_dataset(args.dataset_dir)
    run_results = load_jsonl(args.results)
    items = build_evaluation_items(dataset, run_results, splits=args.splits)
    _validate_human_export(human_export, items)

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
        human_export=human_export,
        baseline=_load_json(args.baseline),
        run_metadata={
            "run_id": args.run_id or args.results.stem,
            "variant": args.variant,
            "evaluation_mode": "production_candidate" if args.judge else "diagnostic",
            "current_commit": _current_commit(),
            "result_file": str(args.results),
            "result_sha256": _sha256(args.results),
            "dataset_sha256": _dataset_sha256(args.dataset_dir),
            "case_set_sha256": _case_set_sha256(items),
            "splits": args.splits,
        },
    )

    export_path = _human_export_path(args.report, args.human_export_out)
    args.report.parent.mkdir(parents=True, exist_ok=True)
    export_path.parent.mkdir(parents=True, exist_ok=True)
    export_path.write_text(
        json.dumps(human_export, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    args.report.write_text(
        json.dumps(report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    print(
        json.dumps(
            {
                "report": str(args.report),
                "human_export": str(export_path),
                "variant": args.variant,
                "evaluation_mode": report["run_metadata"]["evaluation_mode"],
                "production_gate": report["production_gate"],
            },
            ensure_ascii=False,
        )
    )
    if args.gate and report["production_gate"].get("passed") is not True:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
