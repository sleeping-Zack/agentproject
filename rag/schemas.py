"""RAG 检索层统一数据结构。

RetrievalCandidate 贯穿 Dense / BM25 / Fusion / Rerank 四层，每层只填自己那部分分数。
上游拿到的 candidates 是"每个 doc 只有一条记录"的融合结果，可以按需读取任意分数。
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.documents import Document


@dataclass
class RetrievalCandidate:
    doc_id: str
    document: Document
    dense_score: Optional[float] = None
    sparse_score: Optional[float] = None
    fusion_score: Optional[float] = None
    rerank_score: Optional[float] = None
    ranking_score: Optional[float] = None
    meta: Dict[str, Any] = field(default_factory=dict)

    @property
    def content(self) -> str:
        return self.document.page_content

    @property
    def source(self) -> str:
        md = self.document.metadata or {}
        return str(md.get("source_name") or md.get("source") or md.get("file") or "unknown")

    def final_score(self) -> float:
        """返回最终排序实际使用的分数，并兼容未融合的旧候选。"""
        for value in (
            self.ranking_score,
            self.rerank_score,
            self.fusion_score,
            self.dense_score,
            self.sparse_score,
        ):
            if value is not None:
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
    if source is not None:
        # 绝对路径会随部署目录改变；golden 与索引只保留稳定文件名。
        source = Path(str(source)).name
    chunk = md.get("chunk_index")
    if chunk is None:
        chunk = md.get("chunk_id")
    if source is not None and chunk is not None:
        return f"{source}#{chunk}"

    # 旧文档可能没有 chunk metadata。内容摘要比检索结果中的列表下标稳定，
    # 可以保证 Dense 与 BM25 对同一文档生成相同 key。
    content = document.page_content or ""
    if content:
        digest = hashlib.sha256(content.encode("utf-8")).hexdigest()[:24]
        return f"{source or 'content'}#sha256:{digest}"
    fallback = fallback_index if fallback_index is not None else 0
    return f"{source or 'unknown'}#{fallback}"
