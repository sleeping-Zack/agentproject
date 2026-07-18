"""Construct the configured local or remote reranker backend."""
from __future__ import annotations

from typing import Any, Mapping, Optional

from rag.rerankers.base import BaseReranker


def build_reranker(config: Mapping[str, Any]) -> Optional[BaseReranker]:
    if not config.get("enable_reranker"):
        return None
    backend = str(config.get("reranker_backend", "local")).strip().lower()
    common = {
        "model_name": str(
            config.get("reranker_model", "BAAI/bge-reranker-v2-m3")
        ),
        "max_document_chars": int(config.get("rerank_max_document_chars", 1200)),
    }
    if backend == "local":
        from rag.rerankers.bge_reranker import BGEReranker

        return BGEReranker(**common)
    if backend == "remote":
        from rag.rerankers.remote_reranker import RemoteReranker

        return RemoteReranker(
            endpoint=str(config.get("reranker_url") or ""),
            timeout_seconds=float(config.get("reranker_timeout_seconds", 2.0)),
            failure_threshold=int(config.get("reranker_failure_threshold", 3)),
            recovery_timeout=float(config.get("reranker_recovery_seconds", 30.0)),
            **common,
        )
    raise ValueError(f"unsupported reranker backend: {backend}")
