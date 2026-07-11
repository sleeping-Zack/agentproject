"""Validate retrieval-data provenance against source files and live configuration."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any, Dict, Mapping, Optional

import yaml


DEFAULT_MANIFEST = "evals/retrieval_manifest_v1.json"
DEFAULT_CHROMA_CONFIG = "config/chroma.yml"
DEFAULT_RAG_CONFIG = "config/rag.yml"
SHA256_LENGTH = 64


class ManifestValidationError(ValueError):
    pass


def _is_sha256(value: Any) -> bool:
    if not isinstance(value, str) or len(value) != SHA256_LENGTH:
        return False
    try:
        int(value, 16)
    except ValueError:
        return False
    return value == value.lower()


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _required_string(payload: Mapping[str, Any], key: str, errors: list[str]) -> Optional[str]:
    value = payload.get(key)
    if not isinstance(value, str) or not value.strip():
        errors.append(f"{key} must be a non-empty string")
        return None
    return value


def validate_manifest(
    manifest: Mapping[str, Any],
    *,
    project_root: Path,
    chroma_config: Mapping[str, Any],
    rag_config: Mapping[str, Any],
    expected_corpus_hash: Optional[str] = None,
) -> Dict[str, Any]:
    errors: list[str] = []
    if manifest.get("schema_version") != 1:
        errors.append("schema_version must be 1")

    corpus_hash = manifest.get("corpus_hash")
    if not _is_sha256(corpus_hash):
        errors.append("corpus_hash must be a lowercase SHA-256 digest")
    if expected_corpus_hash is not None:
        if not _is_sha256(expected_corpus_hash):
            errors.append("expected corpus hash must be a lowercase SHA-256 digest")
        elif corpus_hash != expected_corpus_hash:
            errors.append("corpus_hash does not match the independently observed corpus")

    corpus_version = _required_string(manifest, "corpus_version", errors)
    if corpus_version is not None and corpus_version != chroma_config.get("corpus_version"):
        errors.append("corpus_version does not match config/chroma.yml")

    chunk_config = manifest.get("chunk_config")
    if not isinstance(chunk_config, Mapping):
        errors.append("chunk_config must be an object")
    else:
        expected_chunk_config = {
            "chunk_version": chroma_config.get("chunk_version"),
            "chunk_size": chroma_config.get("chunk_size"),
            "chunk_overlap": chroma_config.get("chunk_overlap"),
        }
        if dict(chunk_config) != expected_chunk_config:
            errors.append("chunk_config does not match config/chroma.yml")

    embedding_model = _required_string(manifest, "embedding_model", errors)
    if embedding_model is not None and embedding_model != rag_config.get("embedding_model_name"):
        errors.append("embedding_model does not match config/rag.yml")

    retrieval_version = _required_string(manifest, "retrieval_version", errors)
    configured_retrieval = chroma_config.get("retrieval") or {}
    if not isinstance(configured_retrieval, Mapping):
        errors.append("config/chroma.yml retrieval must be an object")
    elif retrieval_version is not None and retrieval_version != configured_retrieval.get("version"):
        errors.append("retrieval_version does not match config/chroma.yml")

    label_scale = manifest.get("label_scale")
    if not isinstance(label_scale, Mapping) or set(label_scale) != {"0", "1", "2", "3"}:
        errors.append("label_scale must define exactly grades 0, 1, 2 and 3")
    if type(manifest.get("split_seed")) is not int or manifest["split_seed"] < 0:
        errors.append("split_seed must be a non-negative integer")

    source_files = manifest.get("source_files")
    verified_sources = 0
    root = project_root.resolve()
    if not isinstance(source_files, Mapping) or not source_files:
        errors.append("source_files must be a non-empty object")
    else:
        for relative_path, expected_hash in sorted(source_files.items()):
            if not isinstance(relative_path, str) or not relative_path:
                errors.append("source_files contains an invalid path")
                continue
            if not _is_sha256(expected_hash):
                errors.append(f"source file hash is not SHA-256: {relative_path}")
                continue
            source_path = (root / relative_path).resolve()
            if not source_path.is_relative_to(root):
                errors.append(f"source file escapes project root: {relative_path}")
                continue
            if not source_path.is_file():
                errors.append(f"source file is missing: {relative_path}")
                continue
            if sha256_file(source_path) != expected_hash:
                errors.append(f"source file hash mismatch: {relative_path}")
                continue
            verified_sources += 1

    if errors:
        raise ManifestValidationError("; ".join(errors))
    return {
        "schema_version": 1,
        "corpus_hash": corpus_hash,
        "source_file_count": verified_sources,
        "corpus_version": corpus_version,
        "chunk_version": chunk_config["chunk_version"],
        "embedding_model": embedding_model,
        "retrieval_version": retrieval_version,
    }


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ManifestValidationError(f"file not found: {path}") from None
    except json.JSONDecodeError:
        raise ManifestValidationError(f"invalid JSON: {path}") from None
    if not isinstance(payload, dict):
        raise ManifestValidationError(f"JSON root must be an object: {path}")
    return payload


def _load_yaml(path: Path) -> Dict[str, Any]:
    try:
        payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ManifestValidationError(f"file not found: {path}") from None
    except yaml.YAMLError:
        raise ManifestValidationError(f"invalid YAML: {path}") from None
    if not isinstance(payload, dict):
        raise ManifestValidationError(f"YAML root must be an object: {path}")
    return payload


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--chroma-config", default=DEFAULT_CHROMA_CONFIG)
    parser.add_argument("--rag-config", default=DEFAULT_RAG_CONFIG)
    parser.add_argument("--project-root", default=".")
    parser.add_argument(
        "--corpus-hash",
        help="independently observed indexed-corpus SHA-256 to compare with the manifest",
    )
    args = parser.parse_args()

    try:
        result = validate_manifest(
            _load_json(Path(args.manifest)),
            project_root=Path(args.project_root),
            chroma_config=_load_yaml(Path(args.chroma_config)),
            rag_config=_load_yaml(Path(args.rag_config)),
            expected_corpus_hash=args.corpus_hash,
        )
    except ManifestValidationError as exc:
        parser.error(str(exc))
    print(json.dumps({"status": "valid", **result}, ensure_ascii=False))


if __name__ == "__main__":
    main()
