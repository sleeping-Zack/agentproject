"""真实检索评测与可复现的离线检索回归门禁。

在线模式分别调用 Dense、Dense+BM25+RRF、可选 Cross-Encoder Rerank。
CI 使用 ``--fixture`` 读取冻结排名，不初始化 Chroma、Embedding 或外部模型；
它验证指标公式和相对基线，不能替代定期在线实测。
"""
from __future__ import annotations

import argparse
import json
import math
import subprocess
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Mapping, Optional


METRIC_NAMES = ("recall_at_k", "precision_at_k", "mrr", "ndcg_at_k", "hit_rate")


def load_golden(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"retrieval golden not found: {path}")
    cases: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                row = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"invalid JSON at {path}:{line_number}: {exc}") from exc
            cases.append(row)
    validate_golden(cases)
    return cases


def validate_golden(cases: List[Dict[str, Any]]) -> None:
    if not cases:
        raise ValueError("retrieval golden must contain at least one case")
    seen_ids = set()
    for index, case in enumerate(cases, start=1):
        case_id = case.get("id")
        relevant = case.get("relevant_doc_ids")
        if not isinstance(case_id, str) or not case_id.strip():
            raise ValueError(f"case {index}: id is required")
        if case_id in seen_ids:
            raise ValueError(f"duplicate retrieval case id: {case_id}")
        seen_ids.add(case_id)
        if not isinstance(case.get("query"), str) or not case["query"].strip():
            raise ValueError(f"case {case_id}: query is required")
        if not isinstance(relevant, list) or not relevant:
            raise ValueError(f"case {case_id}: relevant_doc_ids must be a non-empty list")
        if len(relevant) != len(set(relevant)) or not all(isinstance(item, str) for item in relevant):
            raise ValueError(f"case {case_id}: relevant_doc_ids must contain unique strings")


def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    relevant_set = set(relevant)
    return len(set(retrieved[:k]) & relevant_set) / len(relevant_set) if relevant_set else 0.0


def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
    if k <= 0:
        return 0.0
    relevant_set = set(relevant)
    return len(set(retrieved[:k]) & relevant_set) / k


def mrr(retrieved: List[str], relevant: List[str], k: Optional[int] = None) -> float:
    relevant_set = set(relevant)
    candidates = retrieved if k is None else retrieved[:k]
    for rank, doc_id in enumerate(candidates, start=1):
        if doc_id in relevant_set:
            return 1.0 / rank
    return 0.0


def hit_rate(retrieved: List[str], relevant: List[str], k: int) -> float:
    return float(bool(set(retrieved[:k]) & set(relevant)))


def ndcg_at_k(
    retrieved: List[str],
    relevant: List[str],
    k: int,
    relevance: Optional[Mapping[str, float]] = None,
) -> float:
    grades = dict(relevance or {doc_id: 1.0 for doc_id in relevant})
    if not grades:
        return 0.0

    def gain(grade: float, rank: int) -> float:
        return (2**grade - 1) / math.log2(rank + 1)

    dcg = sum(gain(float(grades.get(doc_id, 0.0)), rank)
              for rank, doc_id in enumerate(retrieved[:k], start=1))
    ideal = sorted((float(value) for value in grades.values()), reverse=True)[:k]
    idcg = sum(gain(grade, rank) for rank, grade in enumerate(ideal, start=1))
    return dcg / idcg if idcg else 0.0


def _candidate_payload(candidate: Any, rank: int) -> Dict[str, Any]:
    payload = {
        "doc_id": candidate.doc_id,
        "rank": rank,
        "dense_score": candidate.dense_score,
        "sparse_score": candidate.sparse_score,
        "fusion_score": candidate.fusion_score,
        "rerank_score": candidate.rerank_score,
        "ranking_score": candidate.ranking_score,
    }
    payload.update({key: value for key, value in (candidate.meta or {}).items()
                    if key in {
                        "retrieved_by",
                        "dense_rank",
                        "bm25_rank",
                        "hybrid_rank",
                        "rerank_rank",
                        "final_rank",
                        "ranking_strategy",
                        "rerank_applied",
                        "rerank_evaluated",
                        "rerank_reason",
                    }})
    return payload


def _run_strategy(
    strategy: Any,
    cases: List[Dict[str, Any]],
    k: int,
    *,
    candidate_k: Optional[int] = None,
) -> Dict[str, Any]:
    candidate_k = max(k, candidate_k or k)
    per_case = []
    latencies_ms: List[float] = []
    for case in cases:
        started = time.perf_counter()
        candidates = strategy.retrieve(case["query"], top_k=candidate_k)
        latencies_ms.append((time.perf_counter() - started) * 1000)
        retrieved_ids = [candidate.doc_id for candidate in candidates]
        relevant = case["relevant_doc_ids"]
        row = {
            "id": case["id"],
            "recall_at_k": recall_at_k(retrieved_ids, relevant, k),
            "precision_at_k": precision_at_k(retrieved_ids, relevant, k),
            "mrr": mrr(retrieved_ids, relevant, k=10),
            "ndcg_at_k": ndcg_at_k(
                retrieved_ids, relevant, k, relevance=case.get("relevance")
            ),
            "hit_rate": hit_rate(retrieved_ids, relevant, k),
            "recall_at_candidate_k": recall_at_k(retrieved_ids, relevant, candidate_k),
            "retrieved": [
                _candidate_payload(candidate, rank)
                for rank, candidate in enumerate(candidates, start=1)
            ],
        }
        per_case.append(row)

    def average(key: str) -> float:
        return sum(float(item[key]) for item in per_case) / len(per_case) if per_case else 0.0

    ordered_latency = sorted(latencies_ms)
    p95_index = max(0, math.ceil(0.95 * len(ordered_latency)) - 1)
    return {
        "case_count": len(per_case),
        **{name: round(average(name), 4) for name in METRIC_NAMES},
        "candidate_k": candidate_k,
        "recall_at_candidate_k": round(average("recall_at_candidate_k"), 4),
        "latency_ms_avg": round(sum(latencies_ms) / len(latencies_ms), 3)
        if latencies_ms else 0.0,
        "latency_ms_p95": round(ordered_latency[p95_index], 3) if ordered_latency else 0.0,
        "per_case": per_case,
    }


class _FixtureRetriever:
    def __init__(self, rankings: Mapping[str, List[str]], strategy_name: str):
        self.rankings = rankings
        self.strategy_name = strategy_name

    def retrieve(self, query: str, top_k: int):
        from langchain_core.documents import Document

        from rag.schemas import RetrievalCandidate

        doc_ids = self.rankings.get(query, [])[:top_k]
        candidates = []
        for rank, doc_id in enumerate(doc_ids, start=1):
            kwargs: Dict[str, Any] = {
                "doc_id": doc_id,
                "document": Document(page_content=doc_id, metadata={"doc_id": doc_id}),
                "meta": {"retrieved_by": [self.strategy_name]},
            }
            if self.strategy_name == "dense_only":
                kwargs["dense_score"] = 1.0 / rank
            elif self.strategy_name == "hybrid":
                kwargs["fusion_score"] = 1.0 / rank
            else:
                kwargs["rerank_score"] = 1.0 / rank
            candidates.append(RetrievalCandidate(**kwargs))
        return candidates


def _build_fixture_strategies(
    fixture_path: Path,
    cases: List[Dict[str, Any]],
) -> tuple[Dict[str, Any], set[str]]:
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    rankings = fixture.get("rankings") or {}
    required = {"dense_only", "hybrid"}
    supported = required | {"hybrid_rerank"}
    if not required.issubset(rankings) or not set(rankings).issubset(supported):
        raise ValueError(
            "fixture must contain dense_only and hybrid, with optional hybrid_rerank"
        )
    case_ids = {case["id"] for case in cases}
    query_by_id = {case["id"]: case["query"] for case in cases}
    known_ids: set[str] = set()
    strategies = {}
    for strategy_name, by_case in rankings.items():
        if set(by_case) != case_ids:
            raise ValueError(f"fixture {strategy_name} case ids do not match golden")
        by_query = {}
        for case_id, doc_ids in by_case.items():
            if len(doc_ids) != len(set(doc_ids)):
                raise ValueError(f"fixture {strategy_name}/{case_id} contains duplicate doc ids")
            by_query[query_by_id[case_id]] = doc_ids
            known_ids.update(doc_ids)
        strategies[strategy_name] = _FixtureRetriever(by_query, strategy_name)
    return strategies, known_ids


def _build_online_strategies(
    enable_reranker: bool,
    k: int,
    *,
    candidate_k: Optional[int] = None,
):
    from rag.retrievers.dense_retriever import DenseRetriever
    from rag.retrievers.hybrid_retriever import HybridRetriever
    from rag.schemas import stable_doc_id
    from rag.vector_store import VectorStoreService
    from utils.config_handler import chroma_conf

    vector_service = VectorStoreService()
    dense = DenseRetriever(vector_service.vector_store)
    bm25 = vector_service.get_bm25_retriever()
    retrieval_depth = max(20, k, candidate_k or k)
    common = {
        "dense_k": retrieval_depth,
        "bm25_k": retrieval_depth,
        "final_k": k,
        "fusion_anchor_k": 20,
    }
    strategies = {
        "dense_only": HybridRetriever(dense=dense, bm25=None, **common),
        "hybrid": HybridRetriever(dense=dense, bm25=bm25, **common),
    }
    if enable_reranker:
        from rag.rerankers.factory import build_reranker

        config = chroma_conf.get("retrieval") or {}
        reranker = build_reranker({**config, "enable_reranker": True})
        if reranker is None:
            raise RuntimeError("reranker is enabled but no backend was built")
        strategies["hybrid_rerank"] = HybridRetriever(
            dense=dense,
            bm25=bm25,
            reranker=reranker,
            rerank_top_n=retrieval_depth,
            rerank_strategy=str(config.get("rerank_strategy", "shadow")),
            rerank_hybrid_weight=float(config.get("rerank_hybrid_weight", 0.7)),
            rerank_model_weight=float(config.get("rerank_model_weight", 0.3)),
            rerank_fusion_k=int(config.get("rerank_fusion_k", 10)),
            rerank_bypass_exact_queries=bool(
                config.get("rerank_bypass_exact_queries", True)
            ),
            **common,
        )
    documents = vector_service._all_documents_from_chroma()
    return strategies, {stable_doc_id(document, index) for index, document in enumerate(documents)}


def _validate_relevant_doc_ids(cases: Iterable[Dict[str, Any]], known_ids: set[str]) -> None:
    failures = {}
    for case in cases:
        unknown = sorted(set(case["relevant_doc_ids"]) - known_ids)
        if unknown:
            failures[case["id"]] = unknown
    if failures:
        raise ValueError(f"golden contains doc ids absent from corpus/fixture: {failures}")


def compare_baseline(
    current: Mapping[str, Mapping[str, float]],
    baseline_payload: Mapping[str, Any],
) -> Dict[str, Any]:
    baseline = baseline_payload.get("strategies") or {}
    allowed = baseline_payload.get("allowed_regression") or {}
    deltas: Dict[str, Dict[str, float]] = {}
    failures = []
    for strategy_name, baseline_metrics in baseline.items():
        if strategy_name not in current:
            failures.append(f"missing_strategy:{strategy_name}")
            continue
        deltas[strategy_name] = {}
        for metric in METRIC_NAMES:
            delta = round(float(current[strategy_name][metric]) - float(baseline_metrics[metric]), 4)
            deltas[strategy_name][metric] = delta
            tolerance = float(allowed.get(metric, 0.0))
            if delta < -tolerance:
                failures.append(f"{strategy_name}:{metric}_regressed:{delta}")
    return {"passed": not failures, "deltas": deltas, "failures": failures}


def compare_strategy_cases(
    reference: Mapping[str, Any],
    candidate: Mapping[str, Any],
) -> Dict[str, Any]:
    """定位候选策略相对参考策略的逐查询退化，而不只看平均数。"""
    reference_cases = {row["id"]: row for row in reference.get("per_case") or []}
    candidate_cases = {row["id"]: row for row in candidate.get("per_case") or []}
    if set(reference_cases) != set(candidate_cases):
        raise ValueError("strategy comparison case ids do not match")

    recall_regressed = []
    recall_improved = []
    ranking_regressed = []
    details = []
    for case_id in sorted(reference_cases):
        before = reference_cases[case_id]
        after = candidate_cases[case_id]
        deltas = {
            metric: round(float(after[metric]) - float(before[metric]), 4)
            for metric in ("recall_at_k", "mrr", "ndcg_at_k")
        }
        if deltas["recall_at_k"] < 0:
            recall_regressed.append(case_id)
        elif deltas["recall_at_k"] > 0:
            recall_improved.append(case_id)
        if deltas["mrr"] < 0 or deltas["ndcg_at_k"] < 0:
            ranking_regressed.append(case_id)
        if any(value != 0 for value in deltas.values()):
            details.append({"id": case_id, "deltas": deltas})

    return {
        "case_count": len(reference_cases),
        "recall_regressed_count": len(recall_regressed),
        "recall_regressed_case_ids": recall_regressed,
        "recall_improved_count": len(recall_improved),
        "recall_improved_case_ids": recall_improved,
        "ranking_regressed_count": len(ranking_regressed),
        "ranking_regressed_case_ids": ranking_regressed,
        "aggregate_deltas": {
            metric: round(float(candidate[metric]) - float(reference[metric]), 4)
            for metric in METRIC_NAMES
        },
        "changed_cases": details,
    }


def _current_commit() -> Optional[str]:
    try:
        completed = subprocess.run(
            ["git", "rev-parse", "HEAD"],
            check=True,
            capture_output=True,
            text=True,
            timeout=5,
        )
    except (OSError, subprocess.SubprocessError):
        return None
    return completed.stdout.strip() or None


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--golden", default="evals/retrieval_golden.jsonl")
    parser.add_argument("--fixture", help="冻结排名 JSON；提供时完全离线运行")
    parser.add_argument("--k", type=int, default=5)
    parser.add_argument(
        "--candidate-k",
        type=int,
        help="同时测量更深候选池的 Recall；在线评测建议设为 20",
    )
    parser.add_argument("--enable-reranker", action="store_true")
    parser.add_argument("--allow-reranker-fallback", action="store_true")
    parser.add_argument("--report", help="写机读 JSON 报告")
    parser.add_argument("--baseline", help="批准的基线 JSON")
    parser.add_argument("--gate", action="store_true")
    parser.add_argument(
        "--gate-strategy",
        choices=["dense_only", "hybrid", "hybrid_rerank"],
        default="hybrid",
        help="strategy that must satisfy absolute thresholds; baseline deltas still check all",
    )
    parser.add_argument("--min-recall", type=float, default=0.6)
    parser.add_argument("--min-precision", type=float, default=0.1)
    parser.add_argument("--min-mrr", type=float, default=0.5)
    parser.add_argument("--min-ndcg", type=float, default=0.5)
    parser.add_argument("--min-hit-rate", type=float, default=0.0)
    parser.add_argument("--min-candidate-recall", type=float)
    parser.add_argument("--max-p95-latency-ms", type=float)
    parser.add_argument("--max-recall-regressed-cases", type=int)
    parser.add_argument("--dry-run", "--schema-check", dest="dry_run", action="store_true")
    parser.add_argument("--split", help="只评测指定 split 的样本，如 'test' 或 'dev'")
    args = parser.parse_args()

    if args.k <= 0:
        parser.error("--k must be greater than zero")
    if args.candidate_k is not None and args.candidate_k < args.k:
        parser.error("--candidate-k must be greater than or equal to --k")
    if args.max_recall_regressed_cases is not None and args.max_recall_regressed_cases < 0:
        parser.error("--max-recall-regressed-cases must be >= 0")

    try:
        cases = load_golden(Path(args.golden))
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    if args.split:
        cases = [c for c in cases if c.get("split") == args.split]
        if not cases:
            parser.error(f"--split {args.split!r} matched no cases in golden set")

    if args.dry_run:
        print(json.dumps({"dry_run": True, "case_count": len(cases)}, ensure_ascii=False))
        return

    try:
        if args.fixture:
            strategies, known_ids = _build_fixture_strategies(Path(args.fixture), cases)
            evaluation_mode = "offline_fixture"
        else:
            strategies, known_ids = _build_online_strategies(
                args.enable_reranker,
                args.k,
                candidate_k=args.candidate_k,
            )
            evaluation_mode = "online_retrievers"
        _validate_relevant_doc_ids(cases, known_ids)
    except (FileNotFoundError, ValueError) as exc:
        parser.error(str(exc))

    report: Dict[str, Dict[str, Any]] = {}
    for name, strategy in strategies.items():
        report[name] = _run_strategy(
            strategy,
            cases,
            args.k,
            candidate_k=args.candidate_k,
        )
        print(json.dumps({"strategy": name, **{key: report[name][key] for key in METRIC_NAMES},
                          "recall_at_candidate_k": report[name]["recall_at_candidate_k"],
                          "latency_ms_avg": report[name]["latency_ms_avg"]}, ensure_ascii=False))

    comparisons = {}
    if "hybrid" in report and "hybrid_rerank" in report:
        comparisons["hybrid_to_hybrid_rerank"] = compare_strategy_cases(
            report["hybrid"], report["hybrid_rerank"]
        )
        print(json.dumps({"comparison": comparisons["hybrid_to_hybrid_rerank"]},
                         ensure_ascii=False))

    reranker_status = None
    if "hybrid_rerank" in strategies and not args.fixture:
        reranker = strategies["hybrid_rerank"].reranker
        reranker_status = {
            "active": bool(getattr(reranker, "is_active", False)),
            "operational": bool(getattr(reranker, "is_operational", False)),
            "successful_calls": int(getattr(reranker, "successful_calls", 0)),
            "failed_calls": int(getattr(reranker, "failed_calls", 0)),
            "last_latency_ms": getattr(reranker, "last_latency_ms", None),
            "error": getattr(reranker, "last_error", None),
        }
        if not reranker_status["operational"] and not args.allow_reranker_fallback:
            print(json.dumps({"reranker": reranker_status, "status": "failed"}, ensure_ascii=False))
            raise SystemExit(2)

    baseline_result = None
    baseline_commit = None
    if args.baseline:
        baseline_payload = json.loads(Path(args.baseline).read_text(encoding="utf-8"))
        baseline_commit = baseline_payload.get("baseline_commit")
        baseline_result = compare_baseline(report, baseline_payload)

    output = {
        "schema_version": 2,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "baseline_commit": baseline_commit,
        "current_commit": _current_commit(),
        "mode": evaluation_mode,
        "k": args.k,
        "strategies": report,
        "comparisons": comparisons,
        "reranker": reranker_status,
        "baseline": baseline_result,
    }
    if args.report:
        report_path = Path(args.report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        report_path.write_text(json.dumps(output, ensure_ascii=False, indent=2), encoding="utf-8")

    if args.gate:
        selected = report.get(args.gate_strategy)
        if selected is None:
            parser.error(f"--gate-strategy {args.gate_strategy!r} was not evaluated")
        failures = []
        if selected["recall_at_k"] < args.min_recall:
            failures.append("recall_below_threshold")
        if selected["precision_at_k"] < args.min_precision:
            failures.append("precision_below_threshold")
        if selected["mrr"] < args.min_mrr:
            failures.append("mrr_below_threshold")
        if selected["ndcg_at_k"] < args.min_ndcg:
            failures.append("ndcg_below_threshold")
        if selected["hit_rate"] < args.min_hit_rate:
            failures.append("hit_rate_below_threshold")
        if (
            args.min_candidate_recall is not None
            and selected["recall_at_candidate_k"] < args.min_candidate_recall
        ):
            failures.append("candidate_recall_below_threshold")
        if (
            args.max_p95_latency_ms is not None
            and selected["latency_ms_p95"] > args.max_p95_latency_ms
        ):
            failures.append("p95_latency_above_threshold")
        comparison = comparisons.get("hybrid_to_hybrid_rerank")
        if (
            args.max_recall_regressed_cases is not None
            and args.gate_strategy == "hybrid_rerank"
            and comparison
            and comparison["recall_regressed_count"] > args.max_recall_regressed_cases
        ):
            failures.append("too_many_recall_regressed_cases")
        if baseline_result and not baseline_result["passed"]:
            failures.extend(baseline_result["failures"])
        gate = {
            "passed": not failures,
            "strategy": args.gate_strategy,
            "failures": failures,
        }
        print(json.dumps({"gate": gate}, ensure_ascii=False))
        if failures:
            raise SystemExit(1)


if __name__ == "__main__":
    main()
