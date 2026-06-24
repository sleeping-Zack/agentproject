from typing import Dict, Iterable, List, Tuple


def _contains_any(text: str, values: Iterable[str]) -> bool:
    return any(value in text for value in values)


def evaluate_case(case: Dict, retrieved: List[Dict], answer: str, k: int = 3) -> Dict[str, float]:
    expected_keywords = case.get("expected_keywords", [])
    expected_sources = case.get("expected_sources", [])
    top_docs = retrieved[:k]
    joined_content = "\n".join(doc.get("content", "") for doc in top_docs)

    keyword_hits = sum(1 for keyword in expected_keywords if keyword in joined_content)
    recall_at_k = keyword_hits / len(expected_keywords) if expected_keywords else 1.0

    mrr = 0.0
    for index, doc in enumerate(top_docs, start=1):
        if _contains_any(doc.get("content", ""), expected_keywords) or doc.get("source") in expected_sources:
            mrr = 1 / index
            break

    citation_hits = sum(1 for source in expected_sources if source in answer)
    citation_hit_rate = citation_hits / len(expected_sources) if expected_sources else 1.0

    answer_keyword_hits = sum(1 for keyword in expected_keywords if keyword in answer)
    hallucination_rate = 0.0 if answer_keyword_hits or citation_hit_rate > 0 else 1.0

    return {
        "recall_at_k": round(recall_at_k, 4),
        "mrr": round(mrr, 4),
        "citation_hit_rate": round(citation_hit_rate, 4),
        "hallucination_rate": round(hallucination_rate, 4),
    }


def evaluate_cases(cases: List[Dict], results: List[Tuple[List[Dict], str]], k: int = 3) -> Dict:
    metrics = [evaluate_case(case, docs, answer, k=k) for case, (docs, answer) in zip(cases, results)]
    if not metrics:
        return {"case_count": 0, "recall_at_k": 0, "mrr": 0, "citation_hit_rate": 0, "hallucination_rate": 0}
    return {
        "case_count": len(metrics),
        "recall_at_k": round(sum(item["recall_at_k"] for item in metrics) / len(metrics), 4),
        "mrr": round(sum(item["mrr"] for item in metrics) / len(metrics), 4),
        "citation_hit_rate": round(
            sum(item["citation_hit_rate"] for item in metrics) / len(metrics), 4
        ),
        "hallucination_rate": round(
            sum(item["hallucination_rate"] for item in metrics) / len(metrics), 4
        ),
    }


def summarize_strategy_metrics(cases: List[Dict], strategy_results: Dict, k: int = 3) -> Dict:
    return {
        strategy: evaluate_cases(cases, results, k=k)
        for strategy, results in strategy_results.items()
    }
