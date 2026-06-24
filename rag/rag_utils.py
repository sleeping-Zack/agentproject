import hashlib
import os
import re
from typing import Dict, Iterable, List, Optional


def build_document_metadata(source_path: str, chunk_version: str) -> Dict[str, str]:
    with open(source_path, "rb") as f:
        content_hash = hashlib.md5(f.read()).hexdigest()
    return {
        "source_path": os.path.abspath(source_path),
        "source_name": os.path.basename(source_path),
        "content_hash": content_hash,
        "chunk_version": chunk_version,
    }


def _tokens(text: str) -> List[str]:
    return [token for token in re.split(r"\s+", text.strip()) if token]


def _keyword_score(query: str, content: str) -> float:
    query_tokens = _tokens(query)
    if not query_tokens:
        return 0
    hits = sum(1 for token in query_tokens if token in content)
    return hits / len(query_tokens)


def hybrid_rank(
    query: str,
    docs: Iterable,
    vector_scores: Optional[Dict[str, float]] = None,
    keyword_weight: float = 0.35,
    top_n: Optional[int] = None,
) -> List:
    vector_scores = vector_scores or {}
    scored = []
    for index, doc in enumerate(docs):
        doc_id = doc.metadata.get("doc_id") or doc.metadata.get("source") or str(index)
        vector_score = vector_scores.get(doc_id, 0.5)
        keyword_score = _keyword_score(query, doc.page_content)
        score = keyword_weight * keyword_score + (1 - keyword_weight) * vector_score
        scored.append((score, doc))
    scored.sort(key=lambda item: item[0], reverse=True)
    ranked = [doc for _, doc in scored]
    return ranked[:top_n] if top_n else ranked


def format_citations(docs: Iterable) -> str:
    parts = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_name") or doc.metadata.get("source") or "unknown"
        page = doc.metadata.get("page")
        suffix = f"#page={page}" if page is not None else ""
        parts.append(f"[{index}] {source}{suffix}")
    return "\n".join(parts)
