import hashlib
import os
import re
from pathlib import Path
from typing import Dict, Iterable


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
