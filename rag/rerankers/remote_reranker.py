"""Timeout-bounded reranker client with circuit-breaker fallback to Hybrid order."""
from __future__ import annotations

import json
import math
import time
import urllib.request
from typing import Any, Callable, Dict, List, Mapping, Optional
from urllib.parse import urlparse

from observability.metrics import metrics_registry
from rag.rerankers.base import BaseReranker
from rag.rerankers.bge_reranker import build_rerank_passage
from rag.schemas import RetrievalCandidate
from services.circuit_breaker import CircuitBreaker


Transport = Callable[[str, Mapping[str, Any], float], Mapping[str, Any]]
MAX_RESPONSE_BYTES = 2 * 1024 * 1024


def _http_transport(
    endpoint: str,
    payload: Mapping[str, Any],
    timeout_seconds: float,
) -> Mapping[str, Any]:
    request = urllib.request.Request(
        endpoint,
        data=json.dumps(payload, ensure_ascii=False).encode("utf-8"),
        headers={"Content-Type": "application/json", "Accept": "application/json"},
        method="POST",
    )
    with urllib.request.urlopen(request, timeout=timeout_seconds) as response:
        body = response.read(MAX_RESPONSE_BYTES + 1)
    if len(body) > MAX_RESPONSE_BYTES:
        raise ValueError("reranker response is too large")
    parsed = json.loads(body.decode("utf-8"))
    if not isinstance(parsed, Mapping):
        raise ValueError("reranker response root must be an object")
    return parsed


def _finite_score(value: Any) -> float:
    if isinstance(value, bool):
        raise ValueError("reranker score must be numeric")
    score = float(value)
    if not math.isfinite(score):
        raise ValueError("reranker score must be finite")
    return score


def _parse_scores(response: Mapping[str, Any], candidate_count: int) -> List[float]:
    raw_scores = response.get("scores")
    if isinstance(raw_scores, list):
        if len(raw_scores) != candidate_count:
            raise ValueError("reranker returned an incomplete score list")
        return [_finite_score(value) for value in raw_scores]

    results = response.get("results")
    if not isinstance(results, list) or len(results) != candidate_count:
        raise ValueError("reranker response must contain one result per document")
    scores: List[Optional[float]] = [None] * candidate_count
    for item in results:
        if not isinstance(item, Mapping):
            raise ValueError("reranker result must be an object")
        index = item.get("index")
        if type(index) is not int or not 0 <= index < candidate_count:
            raise ValueError("reranker result index is invalid")
        if scores[index] is not None:
            raise ValueError("reranker returned a duplicate result index")
        value = item.get("relevance_score", item.get("score"))
        scores[index] = _finite_score(value)
    if any(score is None for score in scores):
        raise ValueError("reranker response omitted a document")
    return [float(score) for score in scores]


class RemoteReranker(BaseReranker):
    def __init__(
        self,
        endpoint: str,
        *,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        timeout_seconds: float = 2.0,
        max_document_chars: int = 1200,
        failure_threshold: int = 3,
        recovery_timeout: float = 30.0,
        transport: Optional[Transport] = None,
    ) -> None:
        parsed = urlparse(endpoint)
        if parsed.scheme not in {"http", "https"} or not parsed.netloc:
            raise ValueError("remote reranker endpoint must be an HTTP(S) URL")
        if parsed.username or parsed.password:
            raise ValueError("remote reranker endpoint must not contain credentials")
        if timeout_seconds <= 0 or max_document_chars <= 0:
            raise ValueError("reranker timeout and max_document_chars must be positive")
        if failure_threshold <= 0 or recovery_timeout <= 0:
            raise ValueError("reranker circuit-breaker values must be positive")
        self.endpoint = endpoint
        self.model_name = model_name
        self.timeout_seconds = timeout_seconds
        self.max_document_chars = max_document_chars
        self._transport = transport or _http_transport
        self._breaker = CircuitBreaker(
            name="remote_reranker",
            failure_threshold=failure_threshold,
            recovery_timeout=recovery_timeout,
        )
        self.last_error: Optional[str] = None
        self.last_latency_ms: Optional[float] = None
        self.successful_calls = 0
        self.failed_calls = 0
        self.short_circuited_calls = 0

    @property
    def is_active(self) -> bool:
        return True

    @property
    def is_operational(self) -> bool:
        return self.successful_calls > 0 and self._breaker.state.value == "closed"

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        top_n: int = 5,
    ) -> List[RetrievalCandidate]:
        if not candidates or top_n <= 0:
            return []
        if not self._breaker.allow():
            self.last_error = "circuit_open"
            self.short_circuited_calls += 1
            metrics_registry.inc_counter(
                "agent_rerank_call_total", {"backend": "remote", "status": "circuit_open"}
            )
            return candidates[:top_n]

        payload: Dict[str, Any] = {
            "model": self.model_name,
            "query": query,
            "documents": [
                build_rerank_passage(candidate, max_chars=self.max_document_chars)
                for candidate in candidates
            ],
            "top_n": len(candidates),
        }
        started = time.perf_counter()
        try:
            response = self._transport(self.endpoint, payload, self.timeout_seconds)
            scores = _parse_scores(response, len(candidates))
        except Exception as exc:
            self.last_latency_ms = (time.perf_counter() - started) * 1000
            self.last_error = type(exc).__name__
            self.failed_calls += 1
            self._breaker.record_failure()
            metrics_registry.inc_counter(
                "agent_rerank_call_total", {"backend": "remote", "status": "failed"}
            )
            metrics_registry.observe_histogram(
                "agent_rerank_latency_ms",
                self.last_latency_ms,
                {"backend": "remote", "status": "failed"},
            )
            return candidates[:top_n]

        self.last_latency_ms = (time.perf_counter() - started) * 1000
        self._breaker.record_success()
        metrics_registry.inc_counter(
            "agent_rerank_call_total", {"backend": "remote", "status": "success"}
        )
        metrics_registry.observe_histogram(
            "agent_rerank_latency_ms",
            self.last_latency_ms,
            {"backend": "remote", "status": "success"},
        )
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = score
        ranked = sorted(
            enumerate(candidates),
            key=lambda item: (-float(item[1].rerank_score), item[0]),
        )
        self.last_error = None
        self.successful_calls += 1
        return [candidate for _, candidate in ranked[:top_n]]
