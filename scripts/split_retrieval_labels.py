"""Validate reviewed four-grade labels and build a deterministic dev/test golden set."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Mapping


DEFAULT_LABELS = "evals/annotations/retrieval_labels_v1.jsonl"
DEFAULT_MANIFEST = "evals/retrieval_manifest_v1.json"
DEFAULT_OUTPUT = "evals/retrieval_golden.jsonl"
VALID_GRADES = frozenset(range(4))


def _load_manifest(path: Path) -> Dict[str, Any]:
    try:
        manifest = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"manifest not found: {path}") from None
    except json.JSONDecodeError:
        raise ValueError(f"manifest is not valid JSON: {path}") from None
    if not isinstance(manifest, dict):
        raise ValueError("manifest root must be an object")
    seed = manifest.get("split_seed")
    if type(seed) is not int or seed < 0:
        raise ValueError("manifest.split_seed must be a non-negative integer")
    label_scale = manifest.get("label_scale")
    if not isinstance(label_scale, Mapping) or set(label_scale) != {"0", "1", "2", "3"}:
        raise ValueError("manifest.label_scale must define grades 0, 1, 2 and 3")
    return manifest


def load_reviewed_labels(path: Path) -> List[Dict[str, Any]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError:
        raise ValueError(f"labels not found: {path}") from None

    rows: List[Dict[str, Any]] = []
    seen_case_ids = set()
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from None
            if not isinstance(row, dict):
                raise ValueError(f"label row at {path}:{line_number} must be an object")

            case_id = row.get("case_id")
            query = row.get("query")
            labels = row.get("labels")
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError(f"label row at {path}:{line_number} has no case_id")
            if case_id in seen_case_ids:
                raise ValueError(f"duplicate label case_id: {case_id}")
            seen_case_ids.add(case_id)
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"case {case_id}: query is required")
            if row.get("review_status") != "reviewed":
                raise ValueError(f"case {case_id}: review_status must be reviewed")
            if not isinstance(row.get("reviewed_by"), str) or not row["reviewed_by"].strip():
                raise ValueError(f"case {case_id}: reviewed_by is required")
            if not isinstance(labels, list) or not labels:
                raise ValueError(f"case {case_id}: labels must be a non-empty list")

            seen_doc_ids = set()
            positive_count = 0
            for index, label in enumerate(labels, start=1):
                if not isinstance(label, dict):
                    raise ValueError(f"case {case_id}: label {index} must be an object")
                doc_id = label.get("doc_id")
                grade = label.get("grade")
                rationale = label.get("rationale")
                if not isinstance(doc_id, str) or not doc_id.strip():
                    raise ValueError(f"case {case_id}: label {index} has no doc_id")
                if doc_id in seen_doc_ids:
                    raise ValueError(f"case {case_id}: duplicate labelled doc_id {doc_id}")
                seen_doc_ids.add(doc_id)
                if type(grade) is not int or grade not in VALID_GRADES:
                    raise ValueError(f"case {case_id}: grade must be an integer from 0 to 3")
                if not isinstance(rationale, str) or not rationale.strip():
                    raise ValueError(f"case {case_id}: label {index} has no rationale")
                positive_count += int(grade > 0)
            if positive_count == 0:
                raise ValueError(f"case {case_id}: at least one positive label is required")
            rows.append(row)
    if len(rows) < 2:
        raise ValueError("at least two reviewed cases are required for dev/test splitting")
    return rows


def _seeded_order(case_id: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{case_id}".encode("utf-8")).hexdigest()


def build_golden(
    labelled_cases: List[Mapping[str, Any]],
    *,
    seed: int,
    dev_ratio: float = 0.5,
) -> List[Dict[str, Any]]:
    if not 0 < dev_ratio < 1:
        raise ValueError("dev_ratio must be between zero and one")
    ordered_for_split = sorted(
        labelled_cases,
        key=lambda row: (_seeded_order(str(row["case_id"]), seed), str(row["case_id"])),
    )
    dev_count = int(len(ordered_for_split) * dev_ratio + 0.5)
    dev_count = min(max(dev_count, 1), len(ordered_for_split) - 1)
    dev_ids = {str(row["case_id"]) for row in ordered_for_split[:dev_count]}

    golden = []
    for row in sorted(labelled_cases, key=lambda item: str(item["case_id"])):
        labels = sorted(row["labels"], key=lambda item: str(item["doc_id"]))
        relevant_doc_ids = [str(label["doc_id"]) for label in labels if label["grade"] > 0]
        relevance = {str(label["doc_id"]): int(label["grade"]) for label in labels}
        golden.append(
            {
                "id": row["case_id"],
                "query": row["query"].strip(),
                "relevant_doc_ids": relevant_doc_ids,
                "relevance": relevance,
                "split": "dev" if row["case_id"] in dev_ids else "test",
            }
        )
    return golden


def write_jsonl_atomic(path: Path, rows: List[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary_path = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = Path(handle.name)
            for row in rows:
                handle.write(json.dumps(row, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--labels", default=DEFAULT_LABELS)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--dev-ratio", type=float, default=0.5)
    args = parser.parse_args()

    try:
        manifest = _load_manifest(Path(args.manifest))
        labels = load_reviewed_labels(Path(args.labels))
        golden = build_golden(
            labels,
            seed=manifest["split_seed"],
            dev_ratio=args.dev_ratio,
        )
        write_jsonl_atomic(Path(args.output), golden)
    except ValueError as exc:
        parser.error(str(exc))

    split_counts = {
        name: sum(row["split"] == name for row in golden)
        for name in ("dev", "test")
    }
    print(
        json.dumps(
            {
                "status": "completed",
                "output": args.output,
                "seed": manifest["split_seed"],
                **split_counts,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
