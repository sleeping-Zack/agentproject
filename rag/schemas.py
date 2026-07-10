"""RAG 检索层统一数据结构。

RetrievalCandidate 贯穿 Dense / BM25 / Fusion / Rerank 四层，每层只填自己那部分分数。
上游拿到的 candidates 是"每个 doc 只有一条记录"的融合结果，可以按需读取任意分数。
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional

from langchain_core.documents import Document


@dataclass
class RetrievalCandidate:
    doc_id: str
    document: Document
    dense_score: float = 0.0
    sparse_score: float = 0.0
    fusion_score: float = 0.0
    rerank_score: float = 0.0
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        return self.document.page_content

    @property
    def source(self) -> str:
        md = self.document.metadata or {}
        return str(md.get("source_name") or md.get("source") or md.get("file") or "unknown")

    def final_score(self) -> float:
        """返回可用的最强信号：rerank > fusion > dense > sparse。"""
        for value in (self.rerank_score, self.fusion_score, self.dense_score, self.sparse_score):
            if value:
                return float(value)
        return 0.0


def stable_doc_id(document: Document, fallback_index: Optional[int] = None) -> str:
    """从 Document.metadata 抽取稳定 doc_id。缺失时退化到 source#chunk_index。

    fallback_index 只用于兜底命名，不影响正常情况下的 doc_id 稳定性。
    """
    md = document.metadata or {}
    if md.get("doc_id"):
        return str(md["doc_id"])
    source = md.get("source_name") or md.get("source") or md.get("file")
    if source is None:
        source = f"doc-{fallback_index}" if fallback_index is not None else "unknown"
    chunk = md.get("chunk_index")
    if chunk is None:
        chunk = md.get("chunk_id")
    if chunk is None:
        chunk = fallback_index if fallback_index is not None else 0
    return f"{source}#{chunk}"
