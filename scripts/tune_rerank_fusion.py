"""Tune Hybrid/Rerank rank-fusion weights on a reviewed dev set only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from scripts.evaluate_retrieval import load_golden, mrr, ndcg_at_k, recall_at_k


DEFAULT_GOLDEN = "evals/retrieval_dev_golden.jsonl"
DEFAULT_REPORT = "reports/retrieval-rerank-dev-shadow.json"
DEFAULT_OUTPUT = "reports/rerank-fusion-tuning.json"


def require_dev_cases(cases: Sequence[Mapping[str, Any]], *, min_cases: int) -> None:
    if min_cases <= 0:
        raise ValueError("min_cases must be greater than zero")
    if len(cases) < min_cases:
        raise ValueError(f"at least {min_cases} dev cases are required; got {len(cases)}")
    invalid = [str(case.get("id")) for case in cases if case.get("split") != "dev"]
    if invalid:
        raise ValueError(f"fusion tuning accepts dev cases only: {invalid}")


def _positive_rank(value: Any, *, field: str, case_id: str) -> int:
    if type(value) is not int or value <= 0:
        raise ValueError(f"report case {case_id}: {field} must be a positive integer")
    return value


def rank_for_weight(
    report_case: Mapping[str, Any],
    *,
    model_weight: float,
    fusion_k: int,
) -> List[str]:
    if not 0 <= model_weight <= 1:
        raise ValueError("model_weight must be between zero and one")
    if fusion_k < 0:
        raise ValueError("fusion_k must be non-negative")
    case_id = str(report_case.get("id"))
    retrieved = report_case.get("retrieved")
    if not isinstance(retrieved, list) or not retrieved:
        raise ValueError(f"report case {case_id}: retrieved candidates are required")

    candidates = [dict(item) for item in retrieved if isinstance(item, Mapping)]
    if len(candidates) != len(retrieved):
        raise ValueError(f"report case {case_id}: invalid candidate payload")
    for candidate in candidates:
        if not isinstance(candidate.get("doc_id"), str) or not candidate["doc_id"]:
            raise ValueError(f"report case {case_id}: candidate doc_id is required")
        _positive_rank(candidate.get("hybrid_rank"), field="hybrid_rank", case_id=case_id)
    candidates.sort(key=lambda item: (item["hybrid_rank"], item["doc_id"]))
    hybrid_ranks = [item["hybrid_rank"] for item in candidates]
    if len(hybrid_ranks) != len(set(hybrid_ranks)):
        raise ValueError(f"report case {case_id}: duplicate hybrid_rank")

    evaluated = [item for item in candidates if item.get("rerank_evaluated") is True]
    if model_weight == 0 or not evaluated:
        return [str(item["doc_id"]) for item in candidates]
    expected_head = candidates[: len(evaluated)]
    if {item["doc_id"] for item in evaluated} != {item["doc_id"] for item in expected_head}:
        raise ValueError(f"report case {case_id}: rerank window is not a Hybrid prefix")

    rerank_ranks = []
    for candidate in evaluated:
        rerank_ranks.append(
            _positive_rank(
                candidate.get("rerank_rank"),
                field="rerank_rank",
                case_id=case_id,
            )
        )
    if len(rerank_ranks) != len(set(rerank_ranks)):
        raise ValueError(f"report case {case_id}: duplicate rerank_rank")

    hybrid_weight = 1.0 - model_weight
    for candidate in expected_head:
        candidate["tuned_score"] = (
            hybrid_weight / (fusion_k + int(candidate["hybrid_rank"]))
            + model_weight / (fusion_k + int(candidate["rerank_rank"]))
        )
    ranked_head = sorted(
        expected_head,
        key=lambda item: (-item["tuned_score"], item["hybrid_rank"], item["doc_id"]),
    )
    return [str(item["doc_id"]) for item in ranked_head + candidates[len(evaluated) :]]


def evaluate_weight_grid(
    cases: Sequence[Mapping[str, Any]],
    report_cases: Sequence[Mapping[str, Any]],
    *,
    model_weights: Sequence[float],
    k: int,
    fusion_k: int,
) -> Dict[str, Any]:
    if k <= 0:
        raise ValueError("k must be greater than zero")
    if not model_weights:
        raise ValueError("at least one model weight is required")
    if 0.0 not in model_weights:
        raise ValueError("weight grid must include 0.0 as the Hybrid baseline")
    golden_by_id = {str(case["id"]): case for case in cases}
    report_by_id = {str(case.get("id")): case for case in report_cases}
    if set(golden_by_id) != set(report_by_id):
        raise ValueError("report case ids do not match dev golden")

    results = []
    per_case_recall: Dict[float, Dict[str, float]] = {}
    for weight in sorted(set(model_weights)):
        recalls = []
        reciprocal_ranks = []
        ndcgs = []
        case_recalls = {}
        for case_id in sorted(golden_by_id):
            golden = golden_by_id[case_id]
            ranked = rank_for_weight(
                report_by_id[case_id],
                model_weight=weight,
                fusion_k=fusion_k,
            )
            relevant = list(golden["relevant_doc_ids"])
            case_recall = recall_at_k(ranked, relevant, k)
            case_recalls[case_id] = case_recall
            recalls.append(case_recall)
            reciprocal_ranks.append(mrr(ranked, relevant, k=10))
            ndcgs.append(
                ndcg_at_k(ranked, relevant, k, relevance=golden.get("relevance"))
            )
        per_case_recall[weight] = case_recalls
        results.append(
            {
                "model_weight": weight,
                "hybrid_weight": round(1.0 - weight, 6),
                "recall_at_k": round(sum(recalls) / len(recalls), 4),
                "mrr": round(sum(reciprocal_ranks) / len(reciprocal_ranks), 4),
                "ndcg_at_k": round(sum(ndcgs) / len(ndcgs), 4),
            }
        )

    baseline = next(row for row in results if row["model_weight"] == 0.0)
    baseline_case_recall = per_case_recall[0.0]
    for row in results:
        weight = float(row["model_weight"])
        regressed = [
            case_id
            for case_id, value in per_case_recall[weight].items()
            if value < baseline_case_recall[case_id]
        ]
        row["recall_regressed_case_ids"] = regressed
        row["eligible"] = (
            row["recall_at_k"] >= baseline["recall_at_k"] and not regressed
        )

    eligible = [row for row in results if row["eligible"]]
    recommended = max(
        eligible,
        key=lambda row: (
            row["ndcg_at_k"],
            row["mrr"],
            -float(row["model_weight"]),
        ),
    )
    return {
        "case_count": len(cases),
        "k": k,
        "fusion_k": fusion_k,
        "selection_policy": "no per-case recall regression; maximize nDCG, then MRR",
        "baseline": baseline,
        "recommended": recommended,
        "weights": results,
    }


def _parse_weights(raw: str) -> List[float]:
    try:
        weights = [float(value.strip()) for value in raw.split(",") if value.strip()]
    except ValueError:
        raise ValueError("weights must be comma-separated numbers") from None
    if not weights or any(weight < 0 or weight > 1 for weight in weights):
        raise ValueError("weights must contain values between zero and one")
    return weights


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--golden", default=DEFAULT_GOLDEN)
    parser.add_argument("--report", default=DEFAULT_REPORT)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--weights", default="0,0.05,0.1,0.15,0.2,0.25,0.3")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--fusion-k", type=int, default=10)
    parser.add_argument("--min-cases", type=int, default=20)
    args = parser.parse_args()

    try:
        cases = load_golden(Path(args.golden))
        require_dev_cases(cases, min_cases=args.min_cases)
        report = json.loads(Path(args.report).read_text(encoding="utf-8"))
        strategy = (report.get("strategies") or {}).get("hybrid_rerank")
        if not isinstance(strategy, Mapping):
            raise ValueError("report has no hybrid_rerank strategy")
        result = evaluate_weight_grid(
            cases,
            strategy.get("per_case") or [],
            model_weights=_parse_weights(args.weights),
            k=args.k,
            fusion_k=args.fusion_k,
        )
        result["source_report"] = args.report
        result["source_golden"] = args.golden
        output = Path(args.output)
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(
            json.dumps(result, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
    except (OSError, json.JSONDecodeError, ValueError) as exc:
        parser.error(str(exc))

    print(json.dumps({"status": "completed", "output": args.output, **result["recommended"]}))


if __name__ == "__main__":
    main()
