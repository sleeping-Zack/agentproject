"""Standalone HTTP service for BGE Cross-Encoder inference."""
from __future__ import annotations

import logging
import math
import os
import threading
from contextlib import asynccontextmanager
from typing import Any, List, Protocol, Sequence

from fastapi import FastAPI, HTTPException
from fastapi.concurrency import run_in_threadpool
from pydantic import BaseModel, Field


DEFAULT_MODEL = "BAAI/bge-reranker-v2-m3"
DEFAULT_MAX_DOCUMENTS = 100
DEFAULT_MAX_DOCUMENT_CHARS = 4000
logger = logging.getLogger("reranker_service")


class RerankRequest(BaseModel):
    model: str = Field(..., min_length=1, max_length=256)
    query: str = Field(..., min_length=1, max_length=4096)
    documents: List[str] = Field(..., min_length=1, max_length=DEFAULT_MAX_DOCUMENTS)
    top_n: int = Field(..., ge=1, le=DEFAULT_MAX_DOCUMENTS)


class RerankResponse(BaseModel):
    model: str
    scores: List[float]


class RerankRuntime(Protocol):
    model_name: str

    @property
    def is_loaded(self) -> bool: ...

    def load(self) -> None: ...

    def score(self, query: str, documents: Sequence[str]) -> List[float]: ...


class CrossEncoderRuntime:
    """Load one model per process and serialize inference on that model."""

    def __init__(self, model_name: str = DEFAULT_MODEL) -> None:
        self.model_name = model_name
        self._model: Any = None
        self._load_lock = threading.Lock()
        self._inference_lock = threading.Lock()

    @property
    def is_loaded(self) -> bool:
        return self._model is not None

    def load(self) -> None:
        if self._model is not None:
            return
        with self._load_lock:
            if self._model is not None:
                return
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(self.model_name)

    def score(self, query: str, documents: Sequence[str]) -> List[float]:
        self.load()
        pairs = [(query, document) for document in documents]
        with self._inference_lock:
            raw_scores = self._model.predict(pairs, show_progress_bar=False)
        scores = [float(value) for value in raw_scores]
        if len(scores) != len(documents) or not all(math.isfinite(value) for value in scores):
            raise ValueError("model returned an invalid score vector")
        return scores


def _env_bool(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    normalized = value.strip().lower()
    if normalized not in {"1", "true", "yes", "on", "0", "false", "no", "off"}:
        raise ValueError(f"invalid environment value for {name}: {value}")
    return normalized in {"1", "true", "yes", "on"}


def create_app(
    runtime: RerankRuntime | None = None,
    *,
    preload: bool | None = None,
) -> FastAPI:
    runtime = runtime or CrossEncoderRuntime(os.getenv("AGENT_RERANK_MODEL", DEFAULT_MODEL))
    should_preload = (
        _env_bool("AGENT_RERANK_SERVICE_PRELOAD", True) if preload is None else preload
    )
    max_document_chars = int(
        os.getenv("AGENT_RERANK_SERVICE_MAX_DOCUMENT_CHARS", str(DEFAULT_MAX_DOCUMENT_CHARS))
    )
    if max_document_chars <= 0:
        raise ValueError("AGENT_RERANK_SERVICE_MAX_DOCUMENT_CHARS must be positive")

    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        if should_preload:
            await run_in_threadpool(runtime.load)
        yield

    service = FastAPI(
        title="Sweeper BGE Reranker",
        version="1.0.0",
        lifespan=lifespan,
    )

    @service.get("/health")
    def health() -> dict[str, object]:
        return {
            "status": "ok",
            "ready": runtime.is_loaded,
            "model": runtime.model_name,
        }

    @service.get("/ready")
    def ready() -> dict[str, object]:
        if not runtime.is_loaded:
            raise HTTPException(status_code=503, detail="model is not loaded")
        return {"status": "ready", "model": runtime.model_name}

    @service.post("/rerank", response_model=RerankResponse)
    def rerank(request: RerankRequest) -> RerankResponse:
        if request.model != runtime.model_name:
            raise HTTPException(status_code=400, detail="requested model is not served")
        if request.top_n != len(request.documents):
            raise HTTPException(
                status_code=400,
                detail="top_n must equal the document count so every document receives a score",
            )
        if any(not document.strip() for document in request.documents):
            raise HTTPException(status_code=400, detail="documents must not be blank")
        if any(len(document) > max_document_chars for document in request.documents):
            raise HTTPException(status_code=413, detail="document exceeds the character limit")
        try:
            scores = runtime.score(request.query, request.documents)
        except Exception as exc:
            logger.exception("reranker inference failed")
            raise HTTPException(
                status_code=503,
                detail=f"reranker inference failed: {type(exc).__name__}",
            ) from exc
        return RerankResponse(model=runtime.model_name, scores=scores)

    return service


app = create_app()
