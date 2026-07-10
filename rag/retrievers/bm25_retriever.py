"""BM25 稀疏检索，用 jieba 做中文分词。

设计要点：
    - Chroma 是 source of truth，BM25 pickle 只是派生索引。
    - 索引缺失或 doc 数量对不上 Chroma 就现场从 Chroma dump 全量重建，避免召回失真。
    - 分词器和 rank_bm25 都做 lazy import，CI 无这些依赖时不会崩。
"""
from __future__ import annotations

import os
import pickle
from dataclasses import dataclass
from typing import List, Optional

from langchain_core.documents import Document

from rag.schemas import RetrievalCandidate, stable_doc_id


def tokenize_chinese(text: str) -> List[str]:
    """jieba 中文分词。lazy import 避免 CI 无该依赖时崩。"""
    import jieba

    return [tok.strip() for tok in jieba.cut(text or "") if tok.strip()]


_tokenize = tokenize_chinese


@dataclass
class _BM25Payload:
    doc_ids: List[str]
    documents: List[Document]
    tokenized_corpus: List[List[str]]


class BM25Retriever:
    def __init__(self, index_path: Optional[str] = None):
        self.index_path = index_path
        self._payload: Optional[_BM25Payload] = None
        self._bm25 = None

    def is_ready(self) -> bool:
        return self._payload is not None and self._bm25 is not None

    def build(self, documents: List[Document]) -> None:
        if not documents:
            self._payload = None
            self._bm25 = None
            return

        from rank_bm25 import BM25Okapi

        doc_ids = [stable_doc_id(doc, idx) for idx, doc in enumerate(documents)]
        tokenized_corpus = [_tokenize(doc.page_content) for doc in documents]
        self._payload = _BM25Payload(
            doc_ids=doc_ids,
            documents=documents,
            tokenized_corpus=tokenized_corpus,
        )
        self._bm25 = BM25Okapi(tokenized_corpus)

    def save(self) -> None:
        if not self.index_path or self._payload is None:
            return
        os.makedirs(os.path.dirname(self.index_path), exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(
                {
                    "doc_ids": self._payload.doc_ids,
                    "documents": [
                        {"page_content": d.page_content, "metadata": d.metadata}
                        for d in self._payload.documents
                    ],
                    "tokenized_corpus": self._payload.tokenized_corpus,
                },
                f,
            )

    def load(self) -> bool:
        if not self.index_path or not os.path.exists(self.index_path):
            return False
        try:
            with open(self.index_path, "rb") as f:
                raw = pickle.load(f)
            from rank_bm25 import BM25Okapi

            documents = [
                Document(page_content=d["page_content"], metadata=d["metadata"])
                for d in raw["documents"]
            ]
            self._payload = _BM25Payload(
                doc_ids=raw["doc_ids"],
                documents=documents,
                tokenized_corpus=raw["tokenized_corpus"],
            )
            self._bm25 = BM25Okapi(raw["tokenized_corpus"])
            return True
        except Exception:
            self._payload = None
            self._bm25 = None
            return False

    def retrieve(self, query: str, k: int = 20) -> List[RetrievalCandidate]:
        if not self.is_ready():
            return []
        assert self._payload is not None and self._bm25 is not None
        tokens = _tokenize(query)
        if not tokens:
            return []
        scores = self._bm25.get_scores(tokens)
        ranked_idx = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:k]
        candidates: List[RetrievalCandidate] = []
        for idx in ranked_idx:
            if scores[idx] <= 0:
                continue
            candidates.append(
                RetrievalCandidate(
                    doc_id=self._payload.doc_ids[idx],
                    document=self._payload.documents[idx],
                    sparse_score=float(scores[idx]),
                )
            )
        return candidates
