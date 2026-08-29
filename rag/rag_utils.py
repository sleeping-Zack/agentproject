from __future__ import annotations

import hashlib
import os
import re
from pathlib import Path
from typing import Dict, Iterable, Union

import yaml


_PRICE_COMPARISON_QUERY = re.compile(r"最贵|价格|售价|价位|多少钱|报价")
_NO_RESULT_REASONS = {
    "evidence_required",
    "knowledge_no_results",
    "no_results",
}
_LOW_RELEVANCE_REASONS = {
    "knowledge_irrelevant",
    "low_relevance",
    "retrieval_relevance_below_threshold",
}
_INSUFFICIENT_CONCLUSION_REASONS = {
    "evidence_insufficient_for_conclusion",
    "unsupported_claim_rate_exceeded",
}


def knowledge_gap_answer(query: str, reason: str) -> str:
    """把知识缺口原因转换为诚实、非技术化的用户文案。"""
    if _PRICE_COMPARISON_QUERY.search(str(query or "")):
        return (
            "当前知识库没有收录各型号的具体售价或可比较价格表，"
            "因此无法判断哪款机器人最贵。"
            "你可以提供候选型号及最新报价，我再帮你排序比较。"
        )

    normalised_reason = str(reason or "").strip().lower()
    if normalised_reason in _NO_RESULT_REASONS:
        return (
            "当前知识库没有检索到可用于回答这个问题的内容，"
            "因此暂时无法给出可靠答案。"
        )
    if normalised_reason in _LOW_RELEVANCE_REASONS:
        return (
            "当前知识库检索到的内容与这个问题相关性较低，"
            "无法据此给出可靠答案。"
        )
    if normalised_reason in _INSUFFICIENT_CONCLUSION_REASONS:
        return "当前知识库中的相关信息不足以支持这个结论，因此暂时无法可靠判断。"
    return "当前知识库中的信息不足以可靠回答这个问题。"


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


def load_knowledge_source_metadata(
    manifest_path: str,
    data_root: str,
) -> Dict[str, Dict[str, Union[str, bool]]]:
    """加载官方资料清单，并按规范化后的本地文件路径建立索引。"""
    manifest = Path(manifest_path)
    if not manifest.exists():
        return {}

    root = Path(data_root).resolve()
    payload = yaml.safe_load(manifest.read_text(encoding="utf-8")) or {}
    sources = payload.get("sources") or []
    metadata_by_path: Dict[str, Dict[str, Union[str, bool]]] = {}

    for source in sources:
        local_path = str(source.get("local_path") or "").strip()
        if not local_path:
            continue
        resolved = (root / local_path).resolve()
        if not resolved.is_relative_to(root):
            raise ValueError(f"knowledge source path is outside data root: {local_path}")

        models = source.get("models") or []
        if isinstance(models, str):
            models = [models]
        metadata: Dict[str, Union[str, bool]] = {
            "source_id": str(source.get("id") or ""),
            "vendor": str(source.get("vendor") or ""),
            "models": "|".join(str(model) for model in models),
            "document_type": str(source.get("document_type") or ""),
            "language": str(source.get("language") or ""),
            "region": str(source.get("region") or ""),
            "source_url": str(
                source.get("official_page_url") or source.get("download_url") or ""
            ),
            "download_url": str(source.get("download_url") or ""),
            "redistribute": bool(source.get("redistribute", False)),
            "rights_status": str(source.get("rights_status") or ""),
            "usage_scope": str(source.get("usage_scope") or ""),
            "verified_at": str(source.get("verified_at") or ""),
        }
        metadata_by_path[str(resolved).casefold()] = metadata

    return metadata_by_path


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


def format_citations(
    docs: Iterable,
    evidence_ids: Iterable[str] | None = None,
) -> str:
    documents = list(docs)
    stable_ids = list(evidence_ids) if evidence_ids is not None else None
    if stable_ids is not None and len(stable_ids) != len(documents):
        raise ValueError("evidence_ids must match docs one-to-one")

    parts = []
    for index, doc in enumerate(documents, start=1):
        source = doc.metadata.get("source_name") or doc.metadata.get("source") or "unknown"
        page = doc.metadata.get("page")
        suffix = f"#page={page}" if page is not None else ""
        citation_id = stable_ids[index - 1] if stable_ids is not None else str(index)
        parts.append(f"[{citation_id}] {source}{suffix}")
    return "\n".join(parts)
