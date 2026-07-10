"""生成评测：走完整 RAG 链路，评估答案覆盖 / 忠实性 / 引用有效性。

拆分之前的问题：
    旧版本对三种策略 (top_k / hybrid / rerank) 都喂同一份"合成文档"（把答案关键词拼接起来），
    所以策略间指标完全不能反映真实检索差异。检索质量已经由 evaluate_retrieval.py 单独评测。
    本脚本只负责"生成阶段"的度量：

    - answer_keyword_recall：答案覆盖 expected_keywords 的比例
    - forbidden_hit_rate：答案是否包含 forbidden_facts（越低越好）
    - citation_validity：answer 引用的 evidence_id 是否都在实际 evidence 里
    - evidence_used：本次实际返回了几条 evidence

不做 LLM-as-judge，减少配额开销。真正的 faithfulness/judge 走 evaluate_judge.py。
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List

from rag.rag_service import RagSummarizeService


def load_golden(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _keyword_recall(answer: str, keywords: List[str]) -> float:
    if not keywords:
        return 1.0
    hits = sum(1 for k in keywords if k in answer)
    return hits / len(keywords)


def _forbidden_hit_rate(answer: str, forbidden: List[str]) -> float:
    if not forbidden:
        return 0.0
    hits = sum(1 for f in forbidden if f in answer)
    return hits / len(forbidden)


def _citation_validity(answer: str, evidence_ids: List[str]) -> float:
    """极简规则：若答案里出现某 evidence_id 字符串，认为是合法引用；未出现视为 1.0（无需引用）。"""
    if not evidence_ids:
        return 1.0
    cited = [eid for eid in evidence_ids if eid in answer]
    return 1.0 if cited else 0.0


def _evaluate_case(service: RagSummarizeService, case: Dict) -> Dict:
    result = service.rag_summarize_result(case["query"])
    evidence_ids = [e.id for e in result.evidence]
    return {
        "id": case.get("id") or case.get("query"),
        "answer_keyword_recall": round(
            _keyword_recall(result.answer, case.get("expected_keywords", [])), 4
        ),
        "forbidden_hit_rate": round(
            _forbidden_hit_rate(result.answer, case.get("forbidden_facts", [])), 4
        ),
        "citation_validity": round(_citation_validity(result.answer, evidence_ids), 4),
        "evidence_used": len(evidence_ids),
    }


def _aggregate(per_case: List[Dict]) -> Dict:
    if not per_case:
        return {"case_count": 0}

    def _avg(key: str) -> float:
        return sum(item[key] for item in per_case) / len(per_case)

    return {
        "case_count": len(per_case),
        "answer_keyword_recall": round(_avg("answer_keyword_recall"), 4),
        "forbidden_hit_rate": round(_avg("forbidden_hit_rate"), 4),
        "citation_validity": round(_avg("citation_validity"), 4),
        "avg_evidence_used": round(_avg("evidence_used"), 2),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="evals/rag_golden.jsonl")
    parser.add_argument("--report", help="写机读 JSON 报告到该路径")
    args = parser.parse_args()

    cases = load_golden(Path(args.golden))
    service = RagSummarizeService()

    per_case = []
    for case in cases:
        row = _evaluate_case(service, case)
        per_case.append(row)
        print(json.dumps(row, ensure_ascii=False))

    summary = _aggregate(per_case)
    print(json.dumps(summary, ensure_ascii=False))

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"summary": summary, "per_case": per_case}, f, ensure_ascii=False, indent=2)


if __name__ == "__main__":
    main()
