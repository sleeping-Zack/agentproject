import hashlib
import os
from typing import Dict, Iterable


def build_document_metadata(source_path: str, chunk_version: str) -> Dict[str, str]:
    with open(source_path, "rb") as f:
        content_hash = hashlib.md5(f.read()).hexdigest()
    return {
        "source_path": os.path.abspath(source_path),
        "source_name": os.path.basename(source_path),
        "content_hash": content_hash,
        "chunk_version": chunk_version,
    }


def format_citations(docs: Iterable) -> str:
    parts = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_name") or doc.metadata.get("source") or "unknown"
        page = doc.metadata.get("page")
        suffix = f"#page={page}" if page is not None else ""
        parts.append(f"[{index}] {source}{suffix}")
    return "\n".join(parts)
