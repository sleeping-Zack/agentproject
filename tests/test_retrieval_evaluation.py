from langchain_core.documents import Document

from rag.schemas import RetrievalCandidate
from scripts.evaluate_retrieval import (
    _build_online_strategies,
    _run_strategy,
    compare_strategy_cases,
)


class StaticStrategy:
    def __init__(self, doc_ids):
        self.doc_ids = doc_ids

    def retrieve(self, query, top_k):
        return [
            RetrievalCandidate(
                doc_id=doc_id,
                document=Document(page_content=doc_id, metadata={"doc_id": doc_id}),
            )
            for doc_id in self.doc_ids[:top_k]
        ]


def _strategy_report(case_id, *, recall, mrr, ndcg):
    return {
        "recall_at_k": recall,
        "precision_at_k": recall,
        "mrr": mrr,
        "ndcg_at_k": ndcg,
        "hit_rate": float(recall > 0),
        "per_case": [
            {
                "id": case_id,
                "recall_at_k": recall,
                "mrr": mrr,
                "ndcg_at_k": ndcg,
            }
        ],
    }


def test_run_strategy_reports_candidate_pool_recall_separately():
    report = _run_strategy(
        StaticStrategy(["relevant-a", "relevant-b", "other"]),
        [
            {
                "id": "case",
                "query": "query",
                "relevant_doc_ids": ["relevant-a", "relevant-b"],
            }
        ],
        1,
        candidate_k=3,
    )

    assert report["recall_at_k"] == 0.5
    assert report["candidate_k"] == 3
    assert report["recall_at_candidate_k"] == 1.0


def test_compare_strategy_cases_lists_recall_and_ranking_regressions():
    reference = _strategy_report("case", recall=1.0, mrr=1.0, ndcg=1.0)
    candidate = _strategy_report("case", recall=0.5, mrr=0.5, ndcg=0.6)

    comparison = compare_strategy_cases(reference, candidate)

    assert comparison["recall_regressed_count"] == 1
    assert comparison["recall_regressed_case_ids"] == ["case"]
    assert comparison["ranking_regressed_case_ids"] == ["case"]
    assert comparison["aggregate_deltas"]["recall_at_k"] == -0.5


def test_online_strategy_builder_expands_underlying_candidate_depth(monkeypatch):
    class VectorService:
        vector_store = object()

        def get_bm25_retriever(self):
            return object()

        def _all_documents_from_chroma(self):
            return []

    monkeypatch.setattr("rag.vector_store.VectorStoreService", VectorService)
    monkeypatch.setattr(
        "rag.retrievers.dense_retriever.DenseRetriever",
        lambda _store: object(),
    )

    strategies, _ = _build_online_strategies(False, 5, candidate_k=50)

    assert strategies["dense_only"].dense_k == 50
    assert strategies["hybrid"].dense_k == 50
    assert strategies["hybrid"].bm25_k == 50
    assert strategies["hybrid"].fusion_anchor_k == 20
