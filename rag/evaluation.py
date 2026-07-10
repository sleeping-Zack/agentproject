"""生成评测的答案侧指标。

只评估"答案-证据"这一层，不再评估"检索到什么"（那属于检索评测，见 scripts/evaluate_retrieval.py）。

- keyword_coverage: expected_facts 命中率
- forbidden_hit_rate: forbidden_facts 命中率（越低越好）
- citation_hit_rate: 期望来源在答案里的出现率
- citation_validity: 答案里的 [n]/来源引用是否都指向传入的 evidence 集合
"""
from __future__ import annotations

import re
from typing import Dict, Iterable, List


def _contains_any(text: str, values: Iterable[str]) -> bool:
    return any(value in text for value in values)


def keyword_coverage(answer: str, expected: List[str]) -> float:
    if not expected:
        return 1.0
    hits = sum(1 for kw in expected if kw in answer)
    return hits / len(expected)


def forbidden_hit_rate(answer: str, forbidden: List[str]) -> float:
    if not forbidden:
        return 0.0
    hits = sum(1 for kw in forbidden if kw in answer)
    return hits / len(forbidden)


def citation_hit_rate(answer: str, expected_sources: List[str]) -> float:
    if not expected_sources:
        return 1.0
    hits = sum(1 for src in expected_sources if src in answer)
    return hits / len(expected_sources)


_CITATION_PATTERN = re.compile(r"\[(\d+)\]")


def citation_validity(answer: str, evidence_count: int) -> float:
    """答案里出现的 [n] 引用是否都落在 evidence 集合内。空引用视作 1.0。"""
    if evidence_count <= 0:
        return 1.0
    matches = _CITATION_PATTERN.findall(answer)
    if not matches:
        return 1.0
    valid = sum(1 for m in matches if 1 <= int(m) <= evidence_count)
    return valid / len(matches)


def evaluate_generation_case(case: Dict, answer: str, evidence_count: int) -> Dict[str, float]:
    expected_keywords = case.get("expected_keywords") or case.get("expected_facts") or []
    forbidden = case.get("forbidden_facts", [])
    expected_sources = case.get("expected_sources", [])

    return {
        "keyword_coverage": round(keyword_coverage(answer, expected_keywords), 4),
        "forbidden_hit_rate": round(forbidden_hit_rate(answer, forbidden), 4),
        "citation_hit_rate": round(citation_hit_rate(answer, expected_sources), 4),
        "citation_validity": round(citation_validity(answer, evidence_count), 4),
    }


def summarize_generation_metrics(per_case: List[Dict[str, float]]) -> Dict[str, float]:
    if not per_case:
        return {
            "case_count": 0,
            "keyword_coverage": 0.0,
            "forbidden_hit_rate": 0.0,
            "citation_hit_rate": 0.0,
            "citation_validity": 0.0,
        }
    return {
        "case_count": len(per_case),
        "keyword_coverage": round(sum(m["keyword_coverage"] for m in per_case) / len(per_case), 4),
        "forbidden_hit_rate": round(
            sum(m["forbidden_hit_rate"] for m in per_case) / len(per_case), 4
        ),
        "citation_hit_rate": round(sum(m["citation_hit_rate"] for m in per_case) / len(per_case), 4),
        "citation_validity": round(sum(m["citation_validity"] for m in per_case) / len(per_case), 4),
    }
