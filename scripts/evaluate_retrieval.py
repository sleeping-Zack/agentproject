"""检索评测：跑 evals/retrieval_golden.jsonl，输出 Recall@K / MRR / nDCG。

Golden Set 格式：
    {"id": "...", "query": "...", "relevant_doc_ids": ["source#chunk_id", ...]}

评测策略：
    - Dense-only：仅 Chroma similarity search
    - Hybrid：Dense + BM25 + RRF
    - Hybrid + Rerank：加 BGE-Reranker（需 --enable-reranker）

阈值门禁通过 --gate 打开，可用于线上定期基线报告，不建议在 PR CI 上强门禁
（否则每次改 chunk 参数都会破坏历史基线）。
"""
from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path
from typing import Dict, List, Tuple


def load_golden(path: Path) -> List[Dict]:
    with path.open("r", encoding="utf-8") as f:
        return [json.loads(line) for line in f if line.strip()]


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 1.0
    top = set(retrieved[:k])
    hits = sum(1 for r in relevant if r in top)
    return hits / len(relevant)


def mrr(retrieved: List[str], relevant: List[str]) -> float:
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    for rank, doc_id in enumerate(retrieved, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def ndcg_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if not relevant:
        return 0.0
    relevant_set = set(relevant)
    dcg = 0.0
    for rank, doc_id in enumerate(retrieved[:k], start=1):
        if doc_id in relevant_set:
            dcg += 1.0 / math.log2(rank + 1)
    ideal_hits = min(len(relevant), k)
    idcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, ideal_hits + 1))
    return dcg / idcg if idcg else 0.0


def _build_strategies(enable_reranker: bool):
    from rag.retrievers.dense_retriever import DenseRetriever
    from rag.retrievers.hybrid_retriever import HybridRetriever
    from rag.vector_store import VectorStoreService

    vs = VectorStoreService()
    dense = DenseRetriever(vs.vector_store)
    bm25 = vs.get_bm25_retriever()

    dense_only = HybridRetriever(dense=dense, bm25=None, reranker=None, final_k=10)
    hybrid = HybridRetriever(dense=dense, bm25=bm25, reranker=None, final_k=10)
    strategies = {"dense_only": dense_only, "hybrid": hybrid}

    if enable_reranker:
        from rag.rerankers.bge_reranker import BGEReranker

        strategies["hybrid_rerank"] = HybridRetriever(
            dense=dense, bm25=bm25, reranker=BGEReranker(), final_k=10
        )
    return strategies


def _run_strategy(strategy, cases: List[Dict], k: int) -> Dict:
    per_case = []
    latencies_ms: List[float] = []
    for case in cases:
        started = time.perf_counter()
        candidates = strategy.retrieve(case["query"], top_k=k)
        latencies_ms.append((time.perf_counter() - started) * 1000)
        retrieved_ids = [c.doc_id for c in candidates]
        relevant = case.get("relevant_doc_ids", [])
        per_case.append({
            "id": case.get("id") or case.get("query"),
            "recall_at_k": recall_at_k(retrieved_ids, relevant, k),
            "mrr": mrr(retrieved_ids, relevant),
            "ndcg_at_k": ndcg_at_k(retrieved_ids, relevant, k),
        })

    def _avg(key: str) -> float:
        if not per_case:
            return 0.0
        return sum(item[key] for item in per_case) / len(per_case)

    def _p95(values: List[float]) -> float:
        if not values:
            return 0.0
        sorted_values = sorted(values)
        idx = min(len(sorted_values) - 1, int(math.ceil(0.95 * len(sorted_values)) - 1))
        return sorted_values[idx]

    return {
        "case_count": len(per_case),
        "recall_at_k": round(_avg("recall_at_k"), 4),
        "mrr": round(_avg("mrr"), 4),
        "ndcg_at_k": round(_avg("ndcg_at_k"), 4),
        "latency_ms_avg": round(sum(latencies_ms) / len(latencies_ms), 2) if latencies_ms else 0.0,
        "latency_ms_p95": round(_p95(latencies_ms), 2),
        "per_case": per_case,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="evals/retrieval_golden.jsonl")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument("--enable-reranker", action="store_true")
    parser.add_argument("--report", help="写机读 JSON 报告到该路径")
    parser.add_argument("--gate", action="store_true", help="启用阈值门禁")
    parser.add_argument("--min-recall", type=float, default=0.6)
    parser.add_argument("--min-mrr", type=float, default=0.5)
    parser.add_argument("--dry-run", action="store_true",
                        help="只校验 golden 格式（含 relevant_doc_ids），不实际检索。CI 默认")
    args = parser.parse_args()

    golden_path = Path(args.golden)
    if not golden_path.exists():
        # golden 未生成时，CI 只允许 dry-run 通过而不阻塞
        print(json.dumps({"golden": str(golden_path), "status": "missing", "dry_run": True},
                         ensure_ascii=False))
        return

    cases = load_golden(golden_path)

    if args.dry_run:
        invalid = [c for c in cases if not c.get("query") or not isinstance(c.get("relevant_doc_ids"), list)]
        report = {
            "dry_run": True,
            "case_count": len(cases),
            "invalid_count": len(invalid),
            "invalid_ids": [c.get("id") for c in invalid[:5]],
        }
        print(json.dumps(report, ensure_ascii=False))
        if invalid:
            raise SystemExit(1)
        return

    strategies = _build_strategies(args.enable_reranker)

    report: Dict[str, Dict] = {}
    for name, strategy in strategies.items():
        report[name] = _run_strategy(strategy, cases, args.k)
        print(json.dumps({
            "strategy": name,
            "recall_at_k": report[name]["recall_at_k"],
            "mrr": report[name]["mrr"],
            "ndcg_at_k": report[name]["ndcg_at_k"],
            "latency_ms_avg": report[name]["latency_ms_avg"],
            "latency_ms_p95": report[name]["latency_ms_p95"],
        }, ensure_ascii=False))

    if args.report:
        Path(args.report).parent.mkdir(parents=True, exist_ok=True)
        with open(args.report, "w", encoding="utf-8") as f:
            json.dump({"k": args.k, "strategies": report}, f, ensure_ascii=False, indent=2)

    if args.gate:
        baseline = report.get("hybrid") or report.get("dense_only") or {}
        recall = baseline.get("recall_at_k", 0.0)
        mrr_v = baseline.get("mrr", 0.0)
        if recall < args.min_recall or mrr_v < args.min_mrr:
            print(json.dumps({
                "gate": "failed",
                "recall_at_k": recall,
                "mrr": mrr_v,
                "min_recall": args.min_recall,
                "min_mrr": args.min_mrr,
            }, ensure_ascii=False))
            raise SystemExit(1)


if __name__ == "__main__":
    main()
