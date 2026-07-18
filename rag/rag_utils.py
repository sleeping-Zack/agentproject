import hashlib
import os
from pathlib import Path
from typing import Dict, Iterable


def build_document_metadata(source_path: str, chunk_version: str) -> Dict[str, str]:
    with open(source_path, "rb") as f:
        content_hash = hashlib.md5(f.read()).hexdigest()
    return {
        "source_path": os.path.abspath(source_path),
        "source_name": os.path.basename(source_path),
        "document_title": Path(source_path).stem,
        "content_hash": content_hash,
        "chunk_version": chunk_version,
    }


def markdown_section_title(content: str) -> str | None:
    """提取 chunk 开头的 Markdown 标题；无法确定时不猜测。"""
    for line in (content or "").splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if stripped.startswith("#"):
            title = stripped.lstrip("#").strip()
            return title or None
        return None
    return None


def format_citations(docs: Iterable) -> str:
    parts = []
    for index, doc in enumerate(docs, start=1):
        source = doc.metadata.get("source_name") or doc.metadata.get("source") or "unknown"
        page = doc.metadata.get("page")
        suffix = f"#page={page}" if page is not None else ""
        parts.append(f"[{index}] {source}{suffix}")
    return "\n".join(parts)
