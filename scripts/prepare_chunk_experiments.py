"""Create isolated Chroma/BM25 configs for reproducible chunk-size experiments."""
from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Sequence

import yaml


DEFAULT_CHROMA_CONFIG = "config/chroma.yml"
DEFAULT_MANIFEST = "evals/retrieval_manifest_v1.json"
DEFAULT_OUTPUT_DIR = "reports/chunk-experiments"
DEFAULT_VARIANTS = ("200:20", "350:50", "500:80")


def parse_variant(value: str) -> tuple[int, int]:
    try:
        size_text, overlap_text = value.split(":", maxsplit=1)
        size, overlap = int(size_text), int(overlap_text)
    except ValueError:
        raise ValueError(f"invalid chunk variant {value!r}; expected SIZE:OVERLAP") from None
    if size <= 0 or overlap < 0 or overlap >= size:
        raise ValueError("chunk size must be positive and overlap must satisfy 0 <= overlap < size")
    return size, overlap


def build_experiment_config(
    base_config: Mapping[str, Any],
    *,
    chunk_size: int,
    chunk_overlap: int,
) -> tuple[str, Dict[str, Any]]:
    if chunk_size <= 0 or chunk_overlap < 0 or chunk_overlap >= chunk_size:
        raise ValueError("invalid chunk size/overlap")
    slug = f"chunk-{chunk_size}-{chunk_overlap}"
    config = copy.deepcopy(dict(base_config))
    base_version = str(config.get("chunk_version") or "chunk")
    config["collection_name"] = f"{config.get('collection_name', 'agent')}-{slug}"
    config["chunk_version"] = f"{base_version}-cs{chunk_size}-co{chunk_overlap}"
    config["chunk_size"] = chunk_size
    config["chunk_overlap"] = chunk_overlap
    storage_root = f"storage/experiments/{slug}"
    config["persist_directory"] = f"{storage_root}/chroma"
    config["md5_hex_store"] = f"{storage_root}/md5.text"
    retrieval = config.setdefault("retrieval", {})
    if not isinstance(retrieval, dict):
        raise ValueError("base retrieval config must be an object")
    retrieval["bm25_index_path"] = f"{storage_root}/bm25_index.pkl"
    retrieval["enable_reranker"] = False
    retrieval["rerank_strategy"] = "shadow"
    return slug, config


def build_experiment_manifest(
    base_manifest: Mapping[str, Any],
    config: Mapping[str, Any],
) -> Dict[str, Any]:
    manifest = copy.deepcopy(dict(base_manifest))
    manifest["chunk_config"] = {
        "chunk_version": config["chunk_version"],
        "chunk_size": config["chunk_size"],
        "chunk_overlap": config["chunk_overlap"],
    }
    return manifest


def prepare_experiments(
    base_config: Mapping[str, Any],
    base_manifest: Mapping[str, Any],
    *,
    variants: Sequence[tuple[int, int]],
    output_dir: Path,
) -> Dict[str, Any]:
    if not variants:
        raise ValueError("at least one chunk variant is required")
    output_dir.mkdir(parents=True, exist_ok=True)
    experiments = []
    seen = set()
    for chunk_size, chunk_overlap in variants:
        slug, config = build_experiment_config(
            base_config,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
        )
        if slug in seen:
            raise ValueError(f"duplicate chunk variant: {slug}")
        seen.add(slug)
        manifest = build_experiment_manifest(base_manifest, config)
        experiment_dir = output_dir / slug
        experiment_dir.mkdir(parents=True, exist_ok=True)
        config_path = experiment_dir / "chroma.yml"
        manifest_path = experiment_dir / "retrieval_manifest.json"
        candidates_path = experiment_dir / "dev_candidates.jsonl"
        config_path.write_text(
            yaml.safe_dump(config, allow_unicode=True, sort_keys=False),
            encoding="utf-8",
        )
        manifest_path.write_text(
            json.dumps(manifest, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        experiments.append(
            {
                "id": slug,
                "chunk_size": chunk_size,
                "chunk_overlap": chunk_overlap,
                "chunk_version": config["chunk_version"],
                "collection_name": config["collection_name"],
                "persist_directory": config["persist_directory"],
                "config": str(config_path.resolve()),
                "manifest": str(manifest_path.resolve()),
                "dev_candidates": str(candidates_path.resolve()),
                "commands": {
                    "index": (
                        f'$env:AGENT_CHROMA_CONFIG_PATH="{config_path.resolve()}"; '
                        ".\\.venv\\Scripts\\python.exe -m rag.vector_store"
                    ),
                    "validate": (
                        ".\\.venv\\Scripts\\python.exe -m scripts.validate_retrieval_manifest "
                        f'--manifest "{manifest_path.resolve()}" '
                        f'--chroma-config "{config_path.resolve()}"'
                    ),
                    "generate_dev_candidates": (
                        f'$env:AGENT_CHROMA_CONFIG_PATH="{config_path.resolve()}"; '
                        ".\\.venv\\Scripts\\python.exe -m scripts.generate_retrieval_golden "
                        f'--manifest "{manifest_path.resolve()}" '
                        f'--output "{candidates_path.resolve()}"'
                    ),
                },
            }
        )
    plan = {
        "schema_version": 1,
        "isolation_policy": "one collection, persist directory, MD5 store and BM25 index per variant",
        "experiments": experiments,
    }
    (output_dir / "plan.json").write_text(
        json.dumps(plan, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    return plan


def _load_yaml(path: Path) -> Dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"YAML root must be an object: {path}")
    return payload


def _load_json(path: Path) -> Dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--chroma-config", default=DEFAULT_CHROMA_CONFIG)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR)
    parser.add_argument(
        "--variant",
        action="append",
        help="repeatable SIZE:OVERLAP; defaults to 200:20, 350:50 and 500:80",
    )
    args = parser.parse_args()

    try:
        variants = [parse_variant(value) for value in (args.variant or DEFAULT_VARIANTS)]
        plan = prepare_experiments(
            _load_yaml(Path(args.chroma_config)),
            _load_json(Path(args.manifest)),
            variants=variants,
            output_dir=Path(args.output_dir),
        )
    except (OSError, json.JSONDecodeError, yaml.YAMLError, ValueError) as exc:
        parser.error(str(exc))

    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(Path(args.output_dir) / "plan.json"),
                "experiments": len(plan["experiments"]),
            }
        )
    )


if __name__ == "__main__":
    main()
