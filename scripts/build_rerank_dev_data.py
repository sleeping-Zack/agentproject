"""Build a leak-free rerank dev golden set and explicit hard-negative rows."""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

from scripts.generate_retrieval_golden import normalize_query
from scripts.split_retrieval_labels import load_reviewed_labels, write_jsonl_atomic


DEFAULT_CANDIDATES = "evals/annotations/retrieval_dev_candidates_v1.jsonl"
DEFAULT_LABELS = "evals/annotations/retrieval_dev_labels_v1.jsonl"
DEFAULT_LOCKED_GOLDEN = "evals/retrieval_golden.jsonl"
DEFAULT_LOCKED_QUERY_ALIASES = "evals/retrieval_test_query_aliases_v1.jsonl"
DEFAULT_DEV_GOLDEN = "evals/retrieval_dev_golden.jsonl"
DEFAULT_HARD_NEGATIVES = "evals/training/rerank_hard_negatives_v1.jsonl"


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError:
        raise ValueError(f"file not found: {path}") from None

    rows: List[Dict[str, Any]] = []
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from None
            if not isinstance(row, dict):
                raise ValueError(f"row at {path}:{line_number} must be an object")
            rows.append(row)
    if not rows:
        raise ValueError(f"file must contain at least one row: {path}")
    return rows


def _index_dev_candidates(
    candidate_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    indexed: Dict[str, Mapping[str, Any]] = {}
    for row in candidate_rows:
        case_id = row.get("case_id")
        if not isinstance(case_id, str) or not case_id:
            raise ValueError("candidate row has no case_id")
        if case_id in indexed:
            raise ValueError(f"duplicate candidate case_id: {case_id}")
        if row.get("split") != "dev":
            raise ValueError(f"candidate case {case_id}: split must be dev")
        query = row.get("query")
        candidates = row.get("candidates")
        if not isinstance(query, str) or not query.strip():
            raise ValueError(f"candidate case {case_id}: query is required")
        if not isinstance(candidates, list) or not candidates:
            raise ValueError(f"candidate case {case_id}: candidates are required")
        doc_ids = [item.get("doc_id") for item in candidates if isinstance(item, Mapping)]
        if len(doc_ids) != len(candidates) or any(
            not isinstance(doc_id, str) or not doc_id for doc_id in doc_ids
        ):
            raise ValueError(f"candidate case {case_id}: every candidate needs a doc_id")
        if len(doc_ids) != len(set(doc_ids)):
            raise ValueError(f"candidate case {case_id}: duplicate candidate doc_id")
        indexed[case_id] = row
    return indexed


def _locked_queries(rows: Sequence[Mapping[str, Any]]) -> set[str]:
    queries = set()
    for row in rows:
        query = row.get("query")
        if not isinstance(query, str) or not query.strip():
            raise ValueError("locked golden row has no query")
        queries.add(normalize_query(query))
    return queries


def validate_dev_annotations(
    labelled_cases: Sequence[Mapping[str, Any]],
    candidate_rows: Sequence[Mapping[str, Any]],
    locked_rows: Sequence[Mapping[str, Any]],
) -> Dict[str, Mapping[str, Any]]:
    candidates_by_case = _index_dev_candidates(candidate_rows)
    locked = _locked_queries(locked_rows)
    for row in labelled_cases:
        case_id = str(row["case_id"])
        candidate_row = candidates_by_case.get(case_id)
        if candidate_row is None:
            raise ValueError(f"label case {case_id}: no matching candidate row")
        if row.get("split", "dev") != "dev":
            raise ValueError(f"label case {case_id}: split must be dev")
        query = str(row["query"])
        if normalize_query(query) != normalize_query(str(candidate_row["query"])):
            raise ValueError(f"label case {case_id}: query does not match candidates")
        if normalize_query(query) in locked:
            raise ValueError(f"label case {case_id}: query overlaps locked test data")
        candidate_ids = {str(item["doc_id"]) for item in candidate_row["candidates"]}
        labelled_ids = {str(label["doc_id"]) for label in row["labels"]}
        unknown = sorted(labelled_ids - candidate_ids)
        if unknown:
            raise ValueError(f"label case {case_id}: doc ids absent from candidates: {unknown}")
    return candidates_by_case


def build_dev_golden(labelled_cases: Sequence[Mapping[str, Any]]) -> List[Dict[str, Any]]:
    result = []
    for row in sorted(labelled_cases, key=lambda item: str(item["case_id"])):
        labels = sorted(row["labels"], key=lambda item: str(item["doc_id"]))
        result.append(
            {
                "id": str(row["case_id"]),
                "query": str(row["query"]).strip(),
                "relevant_doc_ids": [
                    str(label["doc_id"]) for label in labels if int(label["grade"]) > 0
                ],
                "relevance": {
                    str(label["doc_id"]): int(label["grade"]) for label in labels
                },
                "split": "dev",
            }
        )
    return result


def _hardness_key(candidate: Mapping[str, Any]) -> tuple[int, int, str]:
    ranks = []
    for field in ("rerank_rank", "hybrid_rank", "dense_rank", "bm25_rank"):
        value = candidate.get(field)
        if type(value) is int and value > 0:
            ranks.append(value)
    best_rank = min(ranks, default=10**9)
    rerank_rank = candidate.get("rerank_rank")
    rerank_order = rerank_rank if type(rerank_rank) is int and rerank_rank > 0 else 10**9
    return rerank_order, best_rank, str(candidate["doc_id"])


def mine_hard_negatives(
    labelled_cases: Sequence[Mapping[str, Any]],
    candidates_by_case: Mapping[str, Mapping[str, Any]],
    *,
    max_negatives: int,
) -> List[Dict[str, Any]]:
    if max_negatives <= 0:
        raise ValueError("max_negatives must be greater than zero")
    rows = []
    for labelled in sorted(labelled_cases, key=lambda item: str(item["case_id"])):
        case_id = str(labelled["case_id"])
        candidate_by_id = {
            str(item["doc_id"]): item
            for item in candidates_by_case[case_id]["candidates"]
        }
        grades = {str(item["doc_id"]): int(item["grade"]) for item in labelled["labels"]}
        positive_ids = [doc_id for doc_id, grade in grades.items() if grade >= 2]
        negative_ids = [doc_id for doc_id, grade in grades.items() if grade == 0]
        if not positive_ids or not negative_ids:
            continue
        negative_ids.sort(key=lambda doc_id: _hardness_key(candidate_by_id[doc_id]))

        def passage(doc_id: str) -> Dict[str, Any]:
            candidate = candidate_by_id[doc_id]
            return {
                "doc_id": doc_id,
                "text": str(candidate.get("chunk_text") or ""),
                "source": candidate.get("source"),
                "hybrid_rank": candidate.get("hybrid_rank"),
                "rerank_rank": candidate.get("rerank_rank"),
            }

        rows.append(
            {
                "id": case_id,
                "query": str(labelled["query"]).strip(),
                "split": "dev",
                "positives": [passage(doc_id) for doc_id in positive_ids],
                "hard_negatives": [
                    passage(doc_id) for doc_id in negative_ids[:max_negatives]
                ],
                "reviewed_by": labelled["reviewed_by"],
            }
        )
    return rows


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates", default=DEFAULT_CANDIDATES)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--locked-golden", default=DEFAULT_LOCKED_GOLDEN)
    parser.add_argument("--locked-query-aliases", default=DEFAULT_LOCKED_QUERY_ALIASES)
    parser.add_argument("--dev-golden", default=DEFAULT_DEV_GOLDEN)
    parser.add_argument("--hard-negatives", default=DEFAULT_HARD_NEGATIVES)
    parser.add_argument("--max-negatives", type=int, default=5)
    parser.add_argument("--min-cases", type=int, default=20)
    args = parser.parse_args()

    try:
        labelled = load_reviewed_labels(Path(args.labels))
        if len(labelled) < args.min_cases:
            raise ValueError(
                f"at least {args.min_cases} reviewed dev cases are required; got {len(labelled)}"
            )
        candidate_rows = _load_jsonl(Path(args.candidates))
        locked_rows = _load_jsonl(Path(args.locked_golden))
        locked_rows.extend(_load_jsonl(Path(args.locked_query_aliases)))
        candidates_by_case = validate_dev_annotations(labelled, candidate_rows, locked_rows)
        golden = build_dev_golden(labelled)
        hard_negatives = mine_hard_negatives(
            labelled,
            candidates_by_case,
            max_negatives=args.max_negatives,
        )
        write_jsonl_atomic(Path(args.dev_golden), golden)
        write_jsonl_atomic(Path(args.hard_negatives), hard_negatives)
    except ValueError as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "status": "completed",
                "dev_cases": len(golden),
                "hard_negative_cases": len(hard_negatives),
                "dev_golden": args.dev_golden,
                "hard_negatives": args.hard_negatives,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
