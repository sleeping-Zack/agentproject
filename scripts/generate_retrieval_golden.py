"""从现有 evals/rag_golden.jsonl 出发，为每个 query 生成 relevant_doc_ids 候选。

产出：evals/retrieval_golden.candidates.jsonl —— 需要人工审校后 rename 成 retrieval_golden.jsonl。

策略（保守，不做"自动决定 ground truth"）：
    1. 用 Chroma dense retriever 拉 top-10。
    2. 若 case 提供 expected_keywords / expected_sources，用它们过滤明显不相关的候选。
    3. 每条候选都标记 chunk_id / source / 前 120 字预览，方便人工确认。

评审规则（给人看的）：
    - 保留：doc 内容确实回答了 query
    - 删除：doc 只是关键词命中但语义无关
    - 补充：dense 漏掉但你知道相关的 chunk
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, List


def _load_cases(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def _is_plausible(doc_content: str, case: Dict) -> bool:
    keywords = case.get("expected_keywords") or []
    if not keywords:
        return True
    return any(kw in doc_content for kw in keywords)


def _preview(text: str, length: int = 120) -> str:
    text = (text or "").replace("\n", " ").strip()
    return text[:length]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--rag-golden", default="evals/rag_golden.jsonl")
    parser.add_argument("--output", default="evals/retrieval_golden.candidates.jsonl")
    parser.add_argument("--top-k", type=int, default=10)
    args = parser.parse_args()

    from rag.retrievers.dense_retriever import DenseRetriever
    from rag.vector_store import VectorStoreService

    vs = VectorStoreService()
    dense = DenseRetriever(vs.vector_store)

    cases = _load_cases(Path(args.rag_golden))
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with output_path.open("w", encoding="utf-8") as out:
        for case in cases:
            query = case["query"]
            candidates = dense.retrieve(query, k=args.top_k)
            filtered = []
            for cand in candidates:
                doc = cand.document
                if not _is_plausible(doc.page_content, case):
                    continue
                filtered.append({
                    "doc_id": cand.doc_id,
                    "source": cand.source,
                    "dense_score": cand.dense_score,
                    "preview": _preview(doc.page_content),
                })
            record = {
                "id": case.get("id") or query,
                "query": query,
                "expected_keywords": case.get("expected_keywords", []),
                "expected_sources": case.get("expected_sources", []),
                "relevant_doc_ids": [c["doc_id"] for c in filtered],
                "candidates": filtered,
                "review_status": "pending",
            }
            out.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(json.dumps({
        "output": str(output_path),
        "case_count": len(cases),
        "next_step": "人工审校后 rename 成 evals/retrieval_golden.jsonl",
    }, ensure_ascii=False))


if __name__ == "__main__":
    main()
