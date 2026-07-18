"""本地 BGE-Reranker，走 sentence-transformers 的 CrossEncoder。

sentence-transformers / torch 都是重依赖，且 CI 不装 rerank extras。
所以：
    - 全部 lazy import：类实例化不 import torch
    - 首次 rerank() 才加载模型；失败则退化为原顺序（不阻断主流程）
"""
from __future__ import annotations

import threading
import time
from typing import List, Optional

from rag.rerankers.base import BaseReranker
from rag.schemas import RetrievalCandidate


def build_rerank_passage(
    candidate: RetrievalCandidate,
    *,
    max_chars: int = 1200,
) -> str:
    """把稳定 metadata 和正文组成 Cross-Encoder 可理解的 passage。"""
    if max_chars <= 0:
        raise ValueError("max_chars must be greater than zero")

    metadata = candidate.document.metadata or {}
    fields = (
        ("文档", ("document_title", "source_name", "source", "file")),
        ("产品型号", ("product_model", "device_model", "model")),
        ("章节", ("section_title", "section", "title")),
        ("文档版本", ("document_version", "version")),
        ("页码", ("page",)),
    )
    lines = []
    seen_values = set()
    for label, keys in fields:
        value = next((metadata.get(key) for key in keys if metadata.get(key) not in (None, "")), None)
        if value is None:
            continue
        text = str(value).strip()
        if not text or text in seen_values:
            continue
        seen_values.add(text)
        lines.append(f"{label}：{text}")

    prefix = "\n".join(lines)
    content = (candidate.document.page_content or "").strip()
    separator = "\n正文：" if prefix else "正文："
    remaining = max_chars - len(prefix) - len(separator)
    if remaining <= 0:
        return prefix[:max_chars]
    return f"{prefix}{separator}{content[:remaining]}"


class BGEReranker(BaseReranker):
    def __init__(
        self,
        model_name: str = "BAAI/bge-reranker-v2-m3",
        *,
        max_document_chars: int = 1200,
    ):
        if max_document_chars <= 0:
            raise ValueError("max_document_chars must be greater than zero")
        self.model_name = model_name
        self.max_document_chars = max_document_chars
        self._model = None
        self._load_failed = False
        self._load_lock = threading.Lock()
        self.last_error: Optional[str] = None
        self.successful_calls = 0
        self.failed_calls = 0
        self.last_latency_ms: Optional[float] = None

    def _ensure_model(self) -> Optional[object]:
        if self._model is not None or self._load_failed:
            return self._model
        with self._load_lock:
            if self._model is not None or self._load_failed:
                return self._model
            try:
                from sentence_transformers import CrossEncoder

                self._model = CrossEncoder(self.model_name)
                self.last_error = None
            except Exception as exc:
                self._load_failed = True
                self._model = None
                self.last_error = str(exc)
        return self._model

    @property
    def is_active(self) -> bool:
        return self._model is not None

    @property
    def is_operational(self) -> bool:
        return self.is_active and self.successful_calls > 0 and self.failed_calls == 0

    def rerank(
        self,
        query: str,
        candidates: List[RetrievalCandidate],
        top_n: int = 5,
    ) -> List[RetrievalCandidate]:
        if not candidates:
            return []
        model = self._ensure_model()
        if model is None:
            return candidates[:top_n]

        pairs = [
            (query, build_rerank_passage(c, max_chars=self.max_document_chars))
            for c in candidates
        ]
        started = time.perf_counter()
        try:
            scores = model.predict(pairs)
        except Exception as exc:
            self.last_latency_ms = (time.perf_counter() - started) * 1000
            self.last_error = str(exc)
            self.failed_calls += 1
            return candidates[:top_n]
        self.last_latency_ms = (time.perf_counter() - started) * 1000

        if len(scores) != len(candidates):
            self.last_error = (
                f"reranker returned {len(scores)} scores for {len(candidates)} candidates"
            )
            self.failed_calls += 1
            return candidates[:top_n]
        for candidate, score in zip(candidates, scores):
            candidate.rerank_score = float(score)
        ranked = list(candidates)
        ranked.sort(key=lambda c: c.rerank_score, reverse=True)
        self.last_error = None
        self.successful_calls += 1
        return ranked[:top_n]
