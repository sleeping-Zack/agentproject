"""Validate Agent quality dataset integrity, coverage, provenance, and leakage."""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Dict, List, Mapping, Sequence

import yaml

from scripts.build_agent_quality_dataset import (
    ALLOWED_TOOLS,
    CATEGORY_TARGETS,
    DATASET_VERSION,
    ROOT,
    RUBRIC_VERSION,
    _canonical_line,
    _locked_test_queries,
    _sha256_file,
    normalize_query,
)


DEFAULT_DATASET_DIR = ROOT / "evals" / "agent_quality" / "v1"


def _load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_jsonl(path: Path) -> List[Dict[str, Any]]:
    return [
        json.loads(line)
        for line in path.read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]


def _char_ngrams(value: str, size: int = 3) -> set[str]:
    if len(value) <= size:
        return {value}
    return {value[index : index + size] for index in range(len(value) - size + 1)}


def _jaccard(first: set[str], second: set[str]) -> float:
    union = first | second
    return len(first & second) / len(union) if union else 1.0


def _coverage(cases: Sequence[Mapping[str, Any]]) -> Dict[str, Any]:
    def count(key):
        return dict(sorted(Counter(key(case) for case in cases).items()))

    categories = sorted({case.get("category") for case in cases})
    return {
        "case_count": len(cases),
        "family_count": len({case.get("family_id") for case in cases}),
        "by_split": count(lambda case: case.get("split")),
        "by_category": count(lambda case: case.get("category")),
        "by_category_and_split": {
            category: dict(
                sorted(
                    Counter(
                        case.get("split")
                        for case in cases
                        if case.get("category") == category
                    ).items()
                )
            )
            for category in categories
        },
        "by_scene": count(lambda case: case.get("scene")),
        "by_difficulty": count(lambda case: (case.get("labels") or {}).get("difficulty")),
        "by_risk": count(lambda case: (case.get("labels") or {}).get("risk_level")),
        "by_outcome": count(lambda case: (case.get("expected") or {}).get("outcome")),
    }


def validate_dataset(dataset_dir: Path = DEFAULT_DATASET_DIR) -> Dict[str, Any]:
    errors: List[str] = []
    warnings: List[str] = []
    manifest_path = dataset_dir / "manifest.json"
    if not manifest_path.exists():
        return {"passed": False, "errors": [f"manifest not found: {manifest_path}"], "warnings": []}
    manifest = _load_json(manifest_path)

    split_paths = {split: dataset_dir / f"{split}.jsonl" for split in ("dev", "test", "regression")}
    for split, path in split_paths.items():
        if not path.exists():
            errors.append(f"missing split file: {path}")
    if errors:
        return {"passed": False, "errors": errors, "warnings": warnings}

    cases = [case for split in split_paths for case in _load_jsonl(split_paths[split])]
    rubric = yaml.safe_load((ROOT / "config" / "evaluation_rubric.yml").read_text(encoding="utf-8"))
    allowed_scenes = {item["id"] for item in rubric["scenarios"]}
    allowed_tags = set(rubric["capability_tags"])
    allowed_risks = set(rubric["risk_levels"])
    allowed_difficulties = {"D1", "D2", "D3"}
    allowed_outcomes = {
        "completed", "clarify", "refuse", "pending_approval", "blocked", "graceful_failure",
        "partial_completion", "completed_with_degradation",
    }
    tool_contracts = {
        "rag_summarize": ({"query", "information_gap"}, {"query"}),
        "get_weather": ({"city"}, {"city"}),
        "get_user_location": (set(), set()),
        "get_user_id": (set(), set()),
        "get_current_month": (set(), set()),
        "fetch_external_data": ({"user_id", "month"}, {"user_id", "month"}),
        "fill_context_for_report": (set(), set()),
    }
    required_fields = {
        "schema_version", "dataset_version", "rubric_version", "case_id", "family_id",
        "split", "category", "scene", "evaluation_layer", "language", "query", "turns",
        "context", "labels", "expected", "references", "provenance",
    }

    ids: Dict[str, str] = {}
    normalized: Dict[str, str] = {}
    families: Dict[str, set[str]] = defaultdict(set)
    locked_queries = _locked_test_queries()
    for case in cases:
        case_id = str(case.get("case_id") or "<missing>")
        missing = required_fields - set(case)
        if missing:
            errors.append(f"{case_id}: missing fields {sorted(missing)}")
            continue
        if set(case) != required_fields:
            errors.append(f"{case_id}: unexpected fields {sorted(set(case) - required_fields)}")
        if case_id in ids:
            errors.append(f"duplicate case_id: {case_id}")
        ids[case_id] = case["split"]
        if case["schema_version"] != 1:
            errors.append(f"{case_id}: schema_version must be 1")
        if case["dataset_version"] != DATASET_VERSION:
            errors.append(f"{case_id}: wrong dataset_version")
        if case["rubric_version"] != RUBRIC_VERSION:
            errors.append(f"{case_id}: wrong rubric_version")
        if case["split"] not in split_paths:
            errors.append(f"{case_id}: invalid split")
        if case["category"] not in CATEGORY_TARGETS:
            errors.append(f"{case_id}: invalid category")
        if case["scene"] not in allowed_scenes:
            errors.append(f"{case_id}: invalid scene")
        if case["evaluation_layer"] != "agent_end_to_end" or case["language"] != "zh-CN":
            errors.append(f"{case_id}: invalid evaluation layer or language")
        if not isinstance(case["query"], str) or len(case["query"].strip()) < 2:
            errors.append(f"{case_id}: query is empty")
        query_key = normalize_query(case["query"])
        if query_key in normalized:
            errors.append(f"duplicate normalized query: {case_id} and {normalized[query_key]}")
        normalized[query_key] = case_id
        if query_key in locked_queries and case["split"] != "test":
            errors.append(f"{case_id}: frozen retrieval test query leaked into {case['split']}")
        families[str(case["family_id"])].add(case["split"])

        turns = case["turns"]
        if not isinstance(turns, list):
            errors.append(f"{case_id}: turns must be a list")
        elif turns and (turns[-1].get("role") != "user" or turns[-1].get("content") != case["query"]):
            errors.append(f"{case_id}: final user turn must equal query")

        labels = case["labels"]
        if set(labels) != {"capability_tags", "difficulty", "risk_level"}:
            errors.append(f"{case_id}: invalid labels fields")
        tags = labels.get("capability_tags") or []
        if not tags or len(tags) != len(set(tags)) or not set(tags).issubset(allowed_tags):
            errors.append(f"{case_id}: invalid capability tags {tags}")
        if labels.get("difficulty") not in allowed_difficulties:
            errors.append(f"{case_id}: invalid difficulty")
        if labels.get("risk_level") not in allowed_risks:
            errors.append(f"{case_id}: invalid risk level")

        expected = case["expected"]
        expected_fields = {
            "behavior", "outcome", "tools", "facts", "forbidden_facts",
            "requires_citation", "requires_artifact",
        }
        if set(expected) != expected_fields:
            errors.append(f"{case_id}: invalid expected fields")
        if not isinstance(expected.get("behavior"), str) or len(expected["behavior"].strip()) < 10:
            errors.append(f"{case_id}: expected behavior is not operational")
        if expected.get("outcome") not in allowed_outcomes:
            errors.append(f"{case_id}: invalid expected outcome")
        if not expected.get("facts"):
            errors.append(f"{case_id}: expected facts must not be empty")
        if not expected.get("forbidden_facts"):
            errors.append(f"{case_id}: forbidden facts must not be empty")
        for tool in expected.get("tools") or []:
            if set(tool) != {"name", "arguments", "argument_match"}:
                errors.append(f"{case_id}: invalid expected tool fields")
            if tool.get("name") not in ALLOWED_TOOLS:
                errors.append(f"{case_id}: unknown expected tool {tool.get('name')}")
            if tool.get("argument_match") not in {"exact", "contains"}:
                errors.append(f"{case_id}: invalid argument_match")
            allowed_arguments, required_arguments = tool_contracts.get(
                tool.get("name"), (set(), set())
            )
            arguments = tool.get("arguments")
            if not isinstance(arguments, dict):
                errors.append(f"{case_id}: tool arguments must be an object")
            elif not set(arguments).issubset(allowed_arguments):
                errors.append(f"{case_id}: unknown arguments for {tool.get('name')}")
            elif tool.get("argument_match") == "exact" and set(arguments) != required_arguments:
                errors.append(f"{case_id}: exact arguments incomplete for {tool.get('name')}")
            elif tool.get("argument_match") == "contains" and not required_arguments.intersection(arguments):
                errors.append(f"{case_id}: contains match has no required argument")
        if expected.get("requires_citation") and "citation" not in tags:
            errors.append(f"{case_id}: citation required without citation capability tag")
        if expected.get("requires_citation") and not any(
            tool.get("name") == "rag_summarize" for tool in expected.get("tools") or []
        ):
            errors.append(f"{case_id}: citation required without retrieval tool")
        if expected.get("requires_artifact") and "artifact" not in tags:
            errors.append(f"{case_id}: artifact required without artifact capability tag")

        references = case["references"]
        if not references:
            errors.append(f"{case_id}: references must not be empty")
        for reference in references:
            uri = reference.get("uri")
            if reference.get("type") != "local_file" or not uri:
                errors.append(f"{case_id}: invalid reference {reference}")
            elif not (ROOT / uri).exists():
                errors.append(f"{case_id}: reference does not exist: {uri}")
        provenance = case["provenance"]
        if provenance.get("review_status") != "pending_second_reviewer":
            errors.append(f"{case_id}: unapproved review status")

    for family_id, splits in families.items():
        if len(splits) > 1:
            errors.append(f"family leakage: {family_id} spans {sorted(splits)}")

    ngrams = {key: _char_ngrams(key) for key in normalized}
    query_keys = list(normalized)
    for index, first in enumerate(query_keys):
        if len(first) < 8:
            continue
        for second in query_keys[index + 1 :]:
            if len(second) < 8:
                continue
            similarity = _jaccard(ngrams[first], ngrams[second])
            if similarity < 0.90:
                continue
            first_id, second_id = normalized[first], normalized[second]
            first_split, second_split = ids[first_id], ids[second_id]
            message = f"near duplicate {similarity:.3f}: {first_id}/{second_id}"
            if first_split != second_split:
                errors.append(f"cross-split {message}")
            else:
                warnings.append(message)

    coverage = _coverage(cases)
    if coverage["by_category"] != CATEGORY_TARGETS:
        errors.append(
            f"category coverage mismatch: {coverage['by_category']} != {CATEGORY_TARGETS}"
        )
    if set(coverage["by_scene"]) != allowed_scenes:
        errors.append(f"not all rubric scenes are covered: {coverage['by_scene']}")
    if coverage["by_difficulty"].get("D3", 0) < 40:
        errors.append("fewer than 40 D3 cases")
    if coverage["by_risk"].get("L3", 0) < 30:
        errors.append("fewer than 30 L3 cases")
    if set(coverage["by_risk"]) != allowed_risks:
        errors.append(f"not all risk levels are covered: {coverage['by_risk']}")
    for category, split_counts in coverage["by_category_and_split"].items():
        if set(split_counts) != set(split_paths):
            errors.append(f"category {category} does not cover all splits: {split_counts}")
    coverage_report_path = dataset_dir / "coverage_report.json"
    if not coverage_report_path.exists():
        errors.append("coverage_report.json is missing")
    else:
        reported_coverage = _load_json(coverage_report_path)
        for key, value in coverage.items():
            if reported_coverage.get(key) != value:
                errors.append(f"coverage report mismatch: {key}")

    if manifest.get("dataset_version") != DATASET_VERSION:
        errors.append("manifest dataset_version mismatch")
    if manifest.get("case_count") != len(cases):
        errors.append("manifest case_count mismatch")
    if manifest.get("category_targets") != CATEGORY_TARGETS:
        errors.append("manifest category targets mismatch")
    canonical_payload = "\n".join(
        _canonical_line(case) for case in sorted(cases, key=lambda item: item["case_id"])
    ).encode("utf-8")
    if manifest.get("dataset_sha256") != hashlib.sha256(canonical_payload).hexdigest():
        errors.append("manifest dataset_sha256 mismatch")
    for file_name, file_meta in (manifest.get("files") or {}).items():
        path = dataset_dir / Path(file_name).name
        if not path.exists() or _sha256_file(path) != file_meta.get("sha256"):
            errors.append(f"manifest file hash mismatch: {file_name}")
    for file_name, expected_hash in (manifest.get("source_files") or {}).items():
        path = ROOT / file_name
        if not path.exists() or _sha256_file(path) != expected_hash:
            errors.append(f"manifest source hash mismatch: {file_name}")

    return {
        "passed": not errors,
        "errors": errors,
        "warnings": warnings,
        "coverage": coverage,
        "manifest_status": manifest.get("status"),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset-dir", type=Path, default=DEFAULT_DATASET_DIR)
    args = parser.parse_args()
    result = validate_dataset(args.dataset_dir.resolve())
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
