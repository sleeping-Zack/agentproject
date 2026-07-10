"""BM25 稀疏检索，用 jieba 做中文分词。

设计要点：
    - Chroma 是 source of truth，BM25 pickle 只是派生索引。
    - 索引缺失或 doc 数量对不上 Chroma 就现场从 Chroma dump 全量重建，避免召回失真。
    - 分词器和 rank_bm25 都做 lazy import，CI 无这些依赖时不会崩。
"""
from __future__ import annotations

import hashlib
import json
import os
import pickle
import tempfile
import threading
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.documents import Document

from rag.schemas import RetrievalCandidate, stable_doc_id


def tokenize_chinese(text: str) -> List[str]:
    """面向检索的中英文分词；连续中文不会再退化成单个 token。"""
    import jieba

    tokens = []
    for raw in jieba.cut_for_search((text or "").casefold()):
        token = raw.strip()
        if token and any(char.isalnum() or "\u4e00" <= char <= "\u9fff" for char in token):
            tokens.append(token)
    return tokens


_tokenize = tokenize_chinese


@dataclass
class _BM25Payload:
    doc_ids: List[str]
    documents: List[Document]
    tokenized_corpus: List[List[str]]
    corpus_fingerprint: str


class BM25Retriever:
    INDEX_SCHEMA_VERSION = 2

    def __init__(self, index_path: Optional[str] = None):
        self.index_path = index_path
        self._payload: Optional[_BM25Payload] = None
        self._bm25 = None
        self._lock = threading.RLock()
        self.last_load_error: Optional[str] = None

    def is_ready(self) -> bool:
        return self._payload is not None and self._bm25 is not None

    @property
    def document_count(self) -> int:
        return len(self._payload.doc_ids) if self._payload is not None else 0

    @property
    def corpus_fingerprint(self) -> Optional[str]:
        return self._payload.corpus_fingerprint if self._payload is not None else None

    @staticmethod
    def fingerprint_documents(documents: List[Document]) -> str:
        """对稳定 ID、内容和切块版本做有序摘要，检测同数量的陈旧索引。"""
        rows = []
        for index, document in enumerate(documents):
            metadata = document.metadata or {}
            rows.append(
                {
                    "doc_id": stable_doc_id(document, index),
                    "content": document.page_content,
                    "source": metadata.get("source_name") or metadata.get("source"),
                    "chunk_index": metadata.get("chunk_index", metadata.get("chunk_id")),
                    "chunk_version": metadata.get("chunk_version"),
                    "content_hash": metadata.get("content_hash"),
                }
            )
        encoded = json.dumps(
            sorted(rows, key=lambda row: row["doc_id"]),
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
        ).encode("utf-8")
        return hashlib.sha256(encoded).hexdigest()

    def build(self, documents: List[Document]) -> None:
        with self._lock:
            if not documents:
                self._payload = None
                self._bm25 = None
                return

            from rank_bm25 import BM25Okapi

            doc_ids = [stable_doc_id(doc, idx) for idx, doc in enumerate(documents)]
            if len(doc_ids) != len(set(doc_ids)):
                raise ValueError("duplicate stable doc_id in BM25 corpus")
            tokenized_corpus = [_tokenize(doc.page_content) for doc in documents]
            self._payload = _BM25Payload(
                doc_ids=doc_ids,
                documents=list(documents),
                tokenized_corpus=tokenized_corpus,
                corpus_fingerprint=self.fingerprint_documents(documents),
            )
            self._bm25 = BM25Okapi(tokenized_corpus)

    def save(self) -> None:
        with self._lock:
            if not self.index_path or self._payload is None:
                return
            directory = os.path.dirname(os.path.abspath(self.index_path))
            os.makedirs(directory, exist_ok=True)
            payload = {
                "schema_version": self.INDEX_SCHEMA_VERSION,
                "corpus_fingerprint": self._payload.corpus_fingerprint,
                "doc_ids": self._payload.doc_ids,
                "documents": [
                    {"page_content": d.page_content, "metadata": d.metadata}
                    for d in self._payload.documents
                ],
                "tokenized_corpus": self._payload.tokenized_corpus,
            }
            temp_path = None
            try:
                with tempfile.NamedTemporaryFile(
                    mode="wb", dir=directory, prefix=".bm25-", suffix=".tmp", delete=False
                ) as handle:
                    temp_path = handle.name
                    pickle.dump(payload, handle, protocol=pickle.HIGHEST_PROTOCOL)
                    handle.flush()
                    os.fsync(handle.fileno())
                os.replace(temp_path, self.index_path)
            finally:
                if temp_path and os.path.exists(temp_path):
                    os.unlink(temp_path)

    def load(self, expected_fingerprint: Optional[str] = None) -> bool:
        if not self.index_path or not os.path.exists(self.index_path):
            return False
        with self._lock:
            try:
                # 该 pickle 只从应用自有 index_path 读取；不要将其指向用户上传目录。
                with open(self.index_path, "rb") as f:
                    raw = pickle.load(f)
                if raw.get("schema_version") != self.INDEX_SCHEMA_VERSION:
                    raise ValueError("unsupported BM25 index schema")
                fingerprint = str(raw.get("corpus_fingerprint") or "")
                if expected_fingerprint and fingerprint != expected_fingerprint:
                    raise ValueError("BM25 corpus fingerprint mismatch")
                documents = [
                    Document(page_content=d["page_content"], metadata=d["metadata"])
                    for d in raw["documents"]
                ]
                doc_ids = list(raw["doc_ids"])
                tokenized = list(raw["tokenized_corpus"])
                if not (len(doc_ids) == len(documents) == len(tokenized)):
                    raise ValueError("incomplete BM25 index payload")
                if len(doc_ids) != len(set(doc_ids)):
                    raise ValueError("duplicate doc_id in BM25 index")

                from rank_bm25 import BM25Okapi

                self._payload = _BM25Payload(
                    doc_ids=doc_ids,
                    documents=documents,
                    tokenized_corpus=tokenized,
                    corpus_fingerprint=fingerprint,
                )
                self._bm25 = BM25Okapi(tokenized)
                self.last_load_error = None
                return True
            except Exception as exc:
                self._payload = None
                self._bm25 = None
                self.last_load_error = str(exc)
                return False

    def retrieve(self, query: str, k: int = 20) -> List[RetrievalCandidate]:
        with self._lock:
            if not self.is_ready() or k <= 0:
                return []
            assert self._payload is not None and self._bm25 is not None
            tokens = _tokenize(query)
            if not tokens:
                return []
            scores = self._bm25.get_scores(tokens)
            ranked_idx = sorted(range(len(scores)), key=lambda i: (-scores[i], i))[:k]
            candidates: List[RetrievalCandidate] = []
            for rank, idx in enumerate(ranked_idx, start=1):
                if scores[idx] <= 0:
                    continue
                candidates.append(
                    RetrievalCandidate(
                        doc_id=self._payload.doc_ids[idx],
                        document=self._payload.documents[idx],
                        sparse_score=float(scores[idx]),
                        meta={"bm25_rank": rank, "retrieved_by": ["bm25"]},
                    )
                )
            return candidates
