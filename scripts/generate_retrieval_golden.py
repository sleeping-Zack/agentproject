"""Generate auditable retrieval candidates for human relevance labelling.

The output is deliberately *not* a golden set.  It is the stable union of the
Dense, BM25 and Hybrid top-k results.  Relevance must be supplied later by a
reviewer; answer keywords are never used to filter or label candidates.
"""
from __future__ import annotations

import argparse
import json
import math
import queue
import sys
import threading
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence


DEFAULT_INPUT = "evals/rag_golden.jsonl"
DEFAULT_MANIFEST = "evals/retrieval_manifest_v1.json"
DEFAULT_OUTPUT = "evals/annotations/retrieval_candidates_v1.jsonl"
ROUTE_ORDER = ("dense", "bm25", "hybrid")
RANK_FIELDS = ("dense_rank", "bm25_rank", "hybrid_rank", "rerank_rank")
SCORE_FIELDS = ("dense_score", "sparse_score", "fusion_score", "rerank_score")


class CandidatePipelineError(RuntimeError):
    """Base error for a candidate-generation run."""


class QueryTimeoutError(CandidatePipelineError):
    """A query exceeded the configured wall-clock timeout."""


class QueryRetrievalError(CandidatePipelineError):
    """A retriever failed; its raw exception text is intentionally not retained."""

    def __init__(self, error_type: str):
        super().__init__("retrieval failed")
        self.error_type = error_type


@dataclass(frozen=True)
class VersionInfo:
    corpus_version: str
    corpus_hash: str
    chunk_version: str
    embedding_model: str
    retrieval_version: str

    @classmethod
    def from_manifest(cls, manifest: Mapping[str, Any]) -> "VersionInfo":
        chunk_config = manifest.get("chunk_config")
        if not isinstance(chunk_config, Mapping):
            raise ValueError("manifest.chunk_config must be an object")

        values = {
            "corpus_version": manifest.get("corpus_version"),
            "corpus_hash": manifest.get("corpus_hash"),
            "chunk_version": chunk_config.get("chunk_version"),
            "embedding_model": manifest.get("embedding_model"),
            "retrieval_version": manifest.get("retrieval_version"),
        }
        missing = [key for key, value in values.items() if not isinstance(value, str) or not value]
        if missing:
            raise ValueError(f"manifest is missing version fields: {', '.join(missing)}")
        return cls(**values)


@dataclass
class RetrievalRuntime:
    dense: Any
    bm25: Any
    reranker: Any = None
    rrf_k: int = 60
    rerank_top_n: int = 20


def _load_json(path: Path) -> Dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except FileNotFoundError:
        raise ValueError(f"file not found: {path}") from None
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON in {path} at line {exc.lineno}") from None
    if not isinstance(payload, dict):
        raise ValueError(f"JSON root must be an object: {path}")
    return payload


def load_cases(path: Path) -> List[Dict[str, Any]]:
    try:
        handle = path.open("r", encoding="utf-8")
    except FileNotFoundError:
        raise ValueError(f"file not found: {path}") from None

    cases: List[Dict[str, Any]] = []
    seen_ids = set()
    with handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"invalid JSON at {path}:{line_number}") from None
            if not isinstance(row, dict):
                raise ValueError(f"case at {path}:{line_number} must be an object")
            query = row.get("query")
            case_id = row.get("id") or row.get("case_id")
            if not isinstance(case_id, str) or not case_id.strip():
                raise ValueError(f"case at {path}:{line_number} has no id")
            if not isinstance(query, str) or not query.strip():
                raise ValueError(f"case {case_id} has no query")
            if case_id in seen_ids:
                raise ValueError(f"duplicate case id: {case_id}")
            seen_ids.add(case_id)
            cases.append({**row, "id": case_id, "query": query.strip()})
    if not cases:
        raise ValueError("input must contain at least one case")
    return cases


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_json_safe(item) for item in value]
    return str(value)


def _stable_candidate_id(candidate: Any) -> str:
    document = getattr(candidate, "document", None)
    metadata = getattr(document, "metadata", None) or {}
    metadata_id = metadata.get("doc_id") if isinstance(metadata, Mapping) else None
    candidate_id = metadata_id or getattr(candidate, "doc_id", None)
    if candidate_id is not None and str(candidate_id).strip():
        return str(candidate_id).strip()

    from rag.schemas import stable_doc_id

    if document is None:
        raise ValueError("retrieval candidate has neither doc_id nor document")
    return stable_doc_id(document)


def _positive_rank(value: Any) -> Optional[int]:
    if isinstance(value, bool):
        return None
    try:
        rank = int(value)
    except (TypeError, ValueError):
        return None
    return rank if rank > 0 else None


def _finite_score(value: Any) -> Optional[float]:
    if value is None or isinstance(value, bool):
        return None
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    return score if math.isfinite(score) else None


def _candidate_payload(
    candidate: Any,
    *,
    route: str,
    route_rank: int,
    versions: VersionInfo,
) -> Dict[str, Any]:
    document = getattr(candidate, "document", None)
    if document is None:
        raise ValueError("retrieval candidate has no document")
    metadata = _json_safe(getattr(document, "metadata", None) or {})
    if not isinstance(metadata, dict):
        raise ValueError("candidate metadata must be an object")
    actual_chunk_version = metadata.get("chunk_version")
    if actual_chunk_version and str(actual_chunk_version) != versions.chunk_version:
        raise ValueError("candidate chunk_version does not match retrieval manifest")

    meta = getattr(candidate, "meta", None) or {}
    if not isinstance(meta, Mapping):
        meta = {}
    source = (
        metadata.get("source_name")
        or metadata.get("source")
        or metadata.get("file")
        or getattr(candidate, "source", None)
        or "unknown"
    )
    payload: Dict[str, Any] = {
        "doc_id": _stable_candidate_id(candidate),
        "chunk_text": str(getattr(document, "page_content", "") or ""),
        "source": str(source),
        "metadata": metadata,
        "retrieved_by": [route],
        "dense_rank": _positive_rank(meta.get("dense_rank")),
        "dense_score": _finite_score(getattr(candidate, "dense_score", None)),
        "bm25_rank": _positive_rank(meta.get("bm25_rank")),
        "sparse_score": _finite_score(getattr(candidate, "sparse_score", None)),
        "hybrid_rank": _positive_rank(meta.get("hybrid_rank")),
        "fusion_score": _finite_score(getattr(candidate, "fusion_score", None)),
        "rerank_rank": _positive_rank(meta.get("rerank_rank")),
        "rerank_score": _finite_score(getattr(candidate, "rerank_score", None)),
        "corpus_version": versions.corpus_version,
        "corpus_hash": versions.corpus_hash,
        "chunk_version": versions.chunk_version,
        "embedding_model": versions.embedding_model,
        "retrieval_version": versions.retrieval_version,
    }
    if route == "dense":
        payload["dense_rank"] = route_rank
    elif route == "bm25":
        payload["bm25_rank"] = route_rank
    elif route == "hybrid":
        payload["hybrid_rank"] = payload["hybrid_rank"] or route_rank
    else:
        raise ValueError(f"unknown retrieval route: {route}")
    return payload


def merge_route_candidates(
    route_candidates: Mapping[str, Sequence[Any]],
    versions: VersionInfo,
) -> List[Dict[str, Any]]:
    """Union three ranked routes by stable doc_id without relevance filtering."""
    merged: Dict[str, Dict[str, Any]] = {}
    for route in ROUTE_ORDER:
        candidates = route_candidates.get(route)
        if candidates is None:
            raise ValueError(f"missing retrieval route: {route}")
        for rank, candidate in enumerate(candidates, start=1):
            incoming = _candidate_payload(
                candidate,
                route=route,
                route_rank=rank,
                versions=versions,
            )
            doc_id = incoming["doc_id"]
            current = merged.get(doc_id)
            if current is None:
                merged[doc_id] = incoming
                continue
            if current["chunk_text"] != incoming["chunk_text"]:
                raise ValueError(f"conflicting content for stable doc_id: {doc_id}")

            current["retrieved_by"] = [
                name
                for name in ROUTE_ORDER
                if name in set(current["retrieved_by"]) | set(incoming["retrieved_by"])
            ]
            for field in RANK_FIELDS:
                left, right = current[field], incoming[field]
                if left is None or (right is not None and right < left):
                    current[field] = right
            for field in SCORE_FIELDS:
                if current[field] is None and incoming[field] is not None:
                    current[field] = incoming[field]
            for key, value in incoming["metadata"].items():
                current["metadata"].setdefault(key, value)
            if current["source"] == "unknown" and incoming["source"] != "unknown":
                current["source"] = incoming["source"]

    def ordering(item: Dict[str, Any]) -> tuple[Any, ...]:
        direct_ranks = [item["dense_rank"], item["bm25_rank"]]
        best_direct = min((rank for rank in direct_ranks if rank is not None), default=10**9)
        hybrid_rank = item["hybrid_rank"]
        return (hybrid_rank is None, hybrid_rank or 10**9, best_direct, item["doc_id"])

    return sorted(merged.values(), key=ordering)


def retrieve_candidate_pool(
    query: str,
    *,
    runtime: RetrievalRuntime,
    top_k: int,
    versions: VersionInfo,
) -> List[Dict[str, Any]]:
    """Run each expensive retrieval branch once, then reuse it for RRF fusion."""
    dense_candidates = list(runtime.dense.retrieve(query, k=top_k))
    is_ready = getattr(runtime.bm25, "is_ready", None)
    if callable(is_ready) and not is_ready():
        raise RuntimeError("BM25 retriever is not ready")
    bm25_candidates = list(runtime.bm25.retrieve(query, k=top_k))

    from rag.retrievers.hybrid_retriever import fuse_rrf

    hybrid_candidates = fuse_rrf([dense_candidates, bm25_candidates], k=runtime.rrf_k)
    if runtime.reranker is not None and hybrid_candidates:
        head_size = min(max(runtime.rerank_top_n, top_k), len(hybrid_candidates))
        head = hybrid_candidates[:head_size]
        reranked = list(runtime.reranker.rerank(query, list(head), top_n=len(head)))
        reranked_ids = {_stable_candidate_id(candidate) for candidate in reranked}
        reranked.extend(candidate for candidate in head if _stable_candidate_id(candidate) not in reranked_ids)
        for rank, candidate in enumerate(reranked, start=1):
            candidate.meta["rerank_rank"] = rank
        hybrid_candidates = reranked + hybrid_candidates[head_size:]

    return merge_route_candidates(
        {
            "dense": dense_candidates[:top_k],
            "bm25": bm25_candidates[:top_k],
            "hybrid": hybrid_candidates[:top_k],
        },
        versions,
    )


def _run_with_timeout(operation: Callable[[], Any], timeout_seconds: float) -> Any:
    outcome: queue.Queue[tuple[bool, Any]] = queue.Queue(maxsize=1)

    def invoke() -> None:
        try:
            outcome.put((True, operation()))
        except Exception as exc:  # the main thread converts this to a non-sensitive error
            outcome.put((False, exc))

    worker = threading.Thread(target=invoke, name="retrieval-candidate-query", daemon=True)
    worker.start()
    worker.join(timeout_seconds)
    if worker.is_alive():
        raise QueryTimeoutError("query retrieval timed out")
    succeeded, value = outcome.get_nowait()
    if not succeeded:
        raise QueryRetrievalError(type(value).__name__) from None
    return value


def _completed_case_ids(path: Path) -> set[str]:
    if not path.exists():
        return set()
    completed = set()
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError:
                raise ValueError(f"invalid resume file at {path}:{line_number}") from None
            case_id = row.get("case_id") if isinstance(row, dict) else None
            if not isinstance(case_id, str) or not case_id:
                raise ValueError(f"resume record at {path}:{line_number} has no case_id")
            if case_id in completed:
                raise ValueError(f"duplicate case_id in resume file: {case_id}")
            completed.add(case_id)
    return completed


def _safe_log(event: str, *, case_id: str, error_type: Optional[str] = None) -> None:
    payload = {"event": event, "case_id": case_id}
    if error_type:
        payload["error_type"] = error_type
    print(json.dumps(payload, ensure_ascii=False), file=sys.stderr)


def generate_candidate_file(
    cases: Iterable[Mapping[str, Any]],
    *,
    output_path: Path,
    candidate_loader: Callable[[str], List[Dict[str, Any]]],
    versions: VersionInfo,
    timeout_seconds: float,
    resume: bool = False,
) -> Dict[str, int]:
    completed = _completed_case_ids(output_path) if resume else set()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mode = "a" if resume else "w"
    written = 0
    skipped = 0
    with output_path.open(mode, encoding="utf-8", newline="\n") as handle:
        for case in cases:
            case_id = str(case["id"])
            if case_id in completed:
                skipped += 1
                continue
            try:
                candidates = _run_with_timeout(
                    lambda query=str(case["query"]): candidate_loader(query),
                    timeout_seconds,
                )
            except QueryTimeoutError:
                _safe_log("query_failed", case_id=case_id, error_type="timeout")
                raise
            except QueryRetrievalError as exc:
                _safe_log("query_failed", case_id=case_id, error_type=exc.error_type)
                raise

            record = {
                "schema_version": 1,
                "case_id": case_id,
                "query": case["query"],
                "review_status": "pending",
                "corpus_version": versions.corpus_version,
                "corpus_hash": versions.corpus_hash,
                "chunk_version": versions.chunk_version,
                "embedding_model": versions.embedding_model,
                "retrieval_version": versions.retrieval_version,
                "candidates": candidates,
            }
            handle.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
            handle.flush()
            written += 1
    return {"written": written, "skipped": skipped}


def build_runtime() -> RetrievalRuntime:
    from rag.retrievers.bm25_retriever import BM25Retriever
    from rag.retrievers.dense_retriever import DenseRetriever
    from rag.vector_store import VectorStoreService
    from utils.config_handler import chroma_conf

    vector_service = VectorStoreService()
    dense = DenseRetriever(vector_service.vector_store)
    bm25: Optional[BM25Retriever] = vector_service.get_bm25_retriever()
    if bm25 is None or not bm25.is_ready():
        raise RuntimeError("BM25 retriever is not ready")

    config = chroma_conf.get("retrieval") or {}
    reranker = None
    if config.get("enable_reranker"):
        from rag.rerankers.bge_reranker import BGEReranker

        reranker = BGEReranker(
            model_name=config.get("reranker_model", "BAAI/bge-reranker-v2-m3")
        )
    return RetrievalRuntime(
        dense=dense,
        bm25=bm25,
        reranker=reranker,
        rrf_k=int(config.get("rrf_k", 60)),
        rerank_top_n=int(config.get("fusion_top_n", 20)),
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rag-golden", default=DEFAULT_INPUT)
    parser.add_argument("--manifest", default=DEFAULT_MANIFEST)
    parser.add_argument("--output", default=DEFAULT_OUTPUT)
    parser.add_argument("--top-k", type=int, default=20)
    parser.add_argument("--max-queries", type=int)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--timeout", type=float, default=60.0, help="seconds per query")
    args = parser.parse_args()

    if args.top_k <= 0:
        parser.error("--top-k must be greater than zero")
    if args.max_queries is not None and args.max_queries <= 0:
        parser.error("--max-queries must be greater than zero")
    if args.timeout <= 0:
        parser.error("--timeout must be greater than zero")

    try:
        cases = load_cases(Path(args.rag_golden))
        if args.max_queries is not None:
            cases = cases[: args.max_queries]
        versions = VersionInfo.from_manifest(_load_json(Path(args.manifest)))
        runtime = build_runtime()
        result = generate_candidate_file(
            cases,
            output_path=Path(args.output),
            candidate_loader=lambda query: retrieve_candidate_pool(
                query,
                runtime=runtime,
                top_k=args.top_k,
                versions=versions,
            ),
            versions=versions,
            timeout_seconds=args.timeout,
            resume=args.resume,
        )
    except (ValueError, CandidatePipelineError) as exc:
        reason = "invalid_input" if isinstance(exc, ValueError) else "retrieval_incomplete"
        print(json.dumps({"status": "failed", "reason": reason}), file=sys.stderr)
        raise SystemExit(2) from None
    except Exception as exc:
        # Never print provider exception text: SDK errors may echo credentials or signed URLs.
        print(
            json.dumps({"status": "failed", "reason": "initialization_failed", "error_type": type(exc).__name__}),
            file=sys.stderr,
        )
        raise SystemExit(2) from None

    print(
        json.dumps(
            {
                "status": "completed",
                "output": str(args.output),
                "input_cases": len(cases),
                **result,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
