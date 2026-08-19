from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from evaluation_analysis.reporting import write_markdown
from evaluation_analysis.service import EvaluationAnalysisService
from services.evaluation_analysis_store import SQLiteEvaluationAnalysisStore


def _load_object(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"{path}: expected a JSON object")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze a paired baseline/candidate Agent evaluation experiment."
    )
    parser.add_argument("--baseline", type=Path, required=True)
    parser.add_argument("--candidate", type=Path, required=True)
    parser.add_argument("--experiment-id", required=True)
    parser.add_argument("--mode", choices=["diagnostic", "promotion"], required=True)
    parser.add_argument("--hypothesis", required=True)
    parser.add_argument("--change", required=True)
    parser.add_argument("--baseline-approval", type=Path)
    parser.add_argument("--report-id")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--markdown", type=Path)
    parser.add_argument("--store-db", type=Path)
    parser.add_argument("--tenant")
    parser.add_argument("--gate", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    service = EvaluationAnalysisService()
    report = service.analyze(
        _load_object(args.baseline),
        _load_object(args.candidate),
        experiment={
            "experiment_id": args.experiment_id,
            "mode": args.mode,
            "hypothesis": args.hypothesis,
            "change": args.change,
            "primary_metric": "pass_rate",
        },
        baseline_approval=(
            _load_object(args.baseline_approval) if args.baseline_approval else None
        ),
        report_id=args.report_id,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(
        json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if args.markdown:
        write_markdown(report, args.markdown)
    if args.store_db:
        if not args.tenant:
            raise ValueError("--tenant is required with --store-db")
        SQLiteEvaluationAnalysisStore(args.store_db).save_report(args.tenant, report)
    print(
        json.dumps(
            {
                "report_id": report["report_id"],
                "comparability": report["comparability"]["status"],
                "evaluator_gate": report["evaluator_gate"]["status"],
                "evidence": report["evidence"]["status"],
                "release_decision": report["release_decision"],
            },
            ensure_ascii=False,
        )
    )
    if args.gate and report["release_decision"]["status"] != "eligible_for_human_approval":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
