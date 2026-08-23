import pytest
from langchain_core.documents import Document
from langchain_core.messages import AIMessage

from agent.budget import BudgetManager, bind_budget_manager
from agent.planner import SubTask
from agent.react_agent import ReactAgent
from observability.context import bind_request_context
from observability.tracing import trace_recorder
from rag.rag_service import RagResult, RagSummarizeService
from rag.rag_utils import knowledge_gap_answer
from rag.schemas import RetrievalCandidate
from services.cache import SemanticCache


class FakeChain:
    def invoke(self, payload):
        assert payload["input"] == "怎么保养滤网"
        assert "滤网每周清理" in payload["context"]
        assert "证据ID:manual.pdf#c1" in payload["context"]
        return "建议每周清理滤网。"


class FakeHybrid:
    def __init__(self, candidates):
        self._candidates = candidates
        self.calls = 0

    def retrieve(self, query):
        self.calls += 1
        return self._candidates


class CountingChain:
    def __init__(self):
        self.calls = 0

    def invoke(self, payload):
        self.calls += 1
        assert "滤网每周清理" in payload["context"]
        return f"建议每周清理滤网。回答版本 {self.calls}"


def _cached_service():
    service = RagSummarizeService.__new__(RagSummarizeService)
    service._semantic_cache = SemanticCache(
        embedder=lambda _: [1.0, 0.0],
        threshold=0.95,
        ttl=10,
    )
    service._retrieval_cfg = {"version": "hybrid-rrf-v1"}
    service._prompt_version = "rag_summarize:v1"
    service._chain = CountingChain()

    doc = Document(
        page_content="滤网每周清理",
        metadata={"source": "manual.pdf", "chunk_id": "c1"},
    )
    candidate = RetrievalCandidate(
        doc_id="manual.pdf#c1",
        document=doc,
        dense_score=0.82,
        fusion_score=0.5,
    )
    service._hybrid = FakeHybrid([candidate])
    return service


def test_rag_summarize_result_returns_structured_evidence():
    service = RagSummarizeService.__new__(RagSummarizeService)
    service._semantic_cache = None
    service._chain = FakeChain()
    service._retrieval_cfg = {}

    doc = Document(
        page_content="滤网每周清理",
        metadata={"source": "manual.pdf", "chunk_id": "c1"},
    )
    candidate = RetrievalCandidate(
        doc_id="manual.pdf#c1",
        document=doc,
        dense_score=0.82,
        fusion_score=0.5,
    )
    service._hybrid = FakeHybrid([candidate])

    result = service.rag_summarize_result("怎么保养滤网")

    assert result.answer.startswith("建议每周清理")
    assert result.evidence[0].id == "manual.pdf#c1"
    assert result.evidence[0].content == "滤网每周清理"
    assert result.evidence[0].score == 0.5
    assert "[manual.pdf#c1] manual.pdf" in result.answer
    assert "[1] manual.pdf" not in result.answer


def test_semantic_cache_hit_restores_complete_rag_result():
    service = _cached_service()

    first = service.rag_summarize_result("怎么保养滤网", tenant_id="tenant-a")
    second = service.rag_summarize_result("怎么保养滤网", tenant_id="tenant-a")

    assert service._hybrid.calls == 1
    assert service._chain.calls == 1
    assert second is not first
    assert second.answer == first.answer
    assert second.evidence == first.evidence
    assert second.citations == first.citations
    assert second.evidence[0].id == "manual.pdf#c1"
    assert second.citations[0].evidence_id == "manual.pdf#c1"


def test_semantic_cache_misses_after_corpus_version_change():
    service = _cached_service()

    first = service.rag_summarize_result(
        "怎么保养滤网",
        tenant_id="tenant-a",
        corpus_version="corpus-v1",
    )
    second = service.rag_summarize_result(
        "怎么保养滤网",
        tenant_id="tenant-a",
        corpus_version="corpus-v2",
    )

    assert service._hybrid.calls == 2
    assert service._chain.calls == 2
    assert first.answer != second.answer


def test_semantic_cache_isolates_tenants():
    service = _cached_service()

    with bind_request_context(tenant_id="tenant-a"):
        tenant_a = service.rag_summarize_result("怎么保养滤网")
    with bind_request_context(tenant_id="tenant-b"):
        tenant_b = service.rag_summarize_result("怎么保养滤网")
    with bind_request_context(tenant_id="tenant-a"):
        tenant_a_cached = service.rag_summarize_result("怎么保养滤网")

    assert service._hybrid.calls == 2
    assert service._chain.calls == 2
    assert tenant_a.answer != tenant_b.answer
    assert tenant_a_cached.answer == tenant_a.answer


def test_rag_explains_knowledge_gap_before_model_call_when_no_safe_evidence():
    service = RagSummarizeService.__new__(RagSummarizeService)
    service._semantic_cache = None
    service._retrieval_cfg = {}
    service._hybrid = FakeHybrid([])

    class ExplodingChain:
        def invoke(self, _payload):
            raise AssertionError("model must not run without evidence")

    service._chain = ExplodingChain()

    result = service.rag_summarize_result("量子计算股票明天会涨吗")

    assert "没有检索到可用于回答这个问题的内容" in result.answer
    assert "请求未执行" not in result.answer
    assert result.evidence == []
    assert result.verification["reasons"] == ["evidence_required"]


def test_rag_explains_low_relevance_before_model_call():
    service = RagSummarizeService.__new__(RagSummarizeService)
    service._semantic_cache = None
    service._retrieval_cfg = {
        "min_dense_relevance": 0.15,
        "min_sparse_relevance": 1.0,
    }
    candidate = RetrievalCandidate(
        doc_id="irrelevant#1",
        document=Document(
            page_content="扫地机器人未来可能支持自动倒垃圾。",
            metadata={"source": "manual.txt", "chunk_id": "1"},
        ),
        dense_score=0.04,
        fusion_score=0.02,
    )
    service._hybrid = FakeHybrid([candidate])

    class ExplodingChain:
        def invoke(self, _payload):
            raise AssertionError("model must not run for out-of-domain retrieval")

    service._chain = ExplodingChain()

    result = service.rag_summarize_result("量子计算股票明天会涨吗")

    assert "相关性较低" in result.answer
    assert "请求未执行" not in result.answer
    assert result.verification["reasons"] == ["retrieval_relevance_below_threshold"]


def test_knowledge_gap_answer_explains_price_comparison_limit_precisely():
    answer = knowledge_gap_answer("讲一下最贵的机器人是哪一个", "knowledge_no_results")

    assert answer == (
        "当前知识库没有收录各型号的具体售价或可比较价格表，"
        "因此无法判断哪款机器人最贵。"
        "你可以提供候选型号及最新报价，我再帮你排序比较。"
    )


def test_knowledge_gap_answer_distinguishes_no_results_from_low_relevance():
    no_results = knowledge_gap_answer("量子计算是什么", "knowledge_no_results")
    low_relevance = knowledge_gap_answer("量子计算是什么", "knowledge_irrelevant")

    assert "没有检索到" in no_results
    assert "相关性较低" in low_relevance
    assert no_results != low_relevance


def test_planner_treats_empty_rag_result_as_honest_success(monkeypatch):
    answer = knowledge_gap_answer(
        "讲一下最贵的机器人是哪一个",
        "retrieval_relevance_below_threshold",
    )

    monkeypatch.setattr(
        "agent.react_agent.rag.rag_summarize_result",
        lambda query, **kwargs: RagResult(
            answer=answer,
            business_status="empty",
            verification={
                "passed": False,
                "action": "refuse",
                "reasons": ["retrieval_relevance_below_threshold"],
            },
        ),
    )
    agent = ReactAgent.__new__(ReactAgent)
    planner = agent._build_planner_agent()
    task = SubTask(
        id="t1",
        kind="rag_qa",
        description="检索知识库回答价格比较问题",
        args={"query": "讲一下最贵的机器人是哪一个"},
    )

    result = planner.executor.execute([task])[0]

    assert result.success is True
    assert result.error is None
    assert result.content == answer


def test_planner_tool_trace_records_safe_knowledge_gap_reason():
    request_id = "req-rag-empty-trace"
    trace_recorder.start_trace(request_id, "session-rag-empty")
    rag_result = RagResult(
        answer="当前知识库检索到的内容与这个问题相关性较低，无法据此给出可靠答案。",
        business_status="empty",
        verification={
            "passed": False,
            "action": "refuse",
            "reasons": ["retrieval_relevance_below_threshold"],
            "dense_relevance": 0.04,
            "sparse_relevance": 0.0,
            "internal_detail": "must-not-leak",
        },
    )

    result = ReactAgent._run_planner_tool(
        request_id,
        "rag_summarize",
        {"query": "量子计算是什么"},
        lambda: rag_result,
        result_text=lambda value: value.answer,
    )

    assert result is rag_result
    event = next(
        item
        for item in trace_recorder.export_trace(request_id)["events"]
        if item["category"] == "tool" and item["name"] == "rag_summarize"
    )
    assert event["metadata"]["business_status"] == "empty"
    assert event["metadata"]["verification"]["reasons"] == [
        "retrieval_relevance_below_threshold"
    ]
    assert "internal_detail" not in event["metadata"]["verification"]


def test_rag_refuses_generated_claim_that_is_not_grounded_in_evidence():
    service = RagSummarizeService.__new__(RagSummarizeService)
    service._semantic_cache = None
    service._retrieval_cfg = {}
    service.verify_generation = True
    service.verifier = None
    service._chain = type(
        "UnsupportedChain",
        (),
        {"invoke": lambda self, payload: "可以直接用水冲洗电机。"},
    )()
    candidate = RetrievalCandidate(
        doc_id="manual.pdf#c1",
        document=Document(
            page_content="滤网应每周拆下并使用干布清理。",
            metadata={"source": "manual.pdf", "chunk_id": "c1"},
        ),
        dense_score=0.8,
        fusion_score=0.5,
    )
    service._hybrid = FakeHybrid([candidate])

    result = service.rag_summarize_result("滤网如何维护")

    assert result.answer.startswith("请求未执行")
    assert result.business_status == "verification_failed"
    assert result.verification["passed"] is False
    assert "unsupported_claim_rate_exceeded" in result.verification["reasons"]


def test_rag_retries_failed_generation_once_before_returning_answer():
    service = RagSummarizeService.__new__(RagSummarizeService)
    service._semantic_cache = None
    service._retrieval_cfg = {}
    service.verify_generation = True
    service.verifier = None

    class RecoveringChain:
        def __init__(self):
            self.calls = 0
            self.payloads = []

        def invoke(self, payload):
            self.calls += 1
            self.payloads.append(payload)
            if self.calls == 1:
                return "可以直接用水冲洗电机。"
            return "滤网应每周拆下并使用干布清理。"

    service._chain = RecoveringChain()
    candidate = RetrievalCandidate(
        doc_id="manual.pdf#c1",
        document=Document(
            page_content="滤网应每周拆下并使用干布清理。",
            metadata={"source": "manual.pdf", "chunk_id": "c1"},
        ),
        dense_score=0.8,
        fusion_score=0.5,
    )
    service._hybrid = FakeHybrid([candidate])

    result = service.rag_summarize_result("滤网如何维护")

    assert service._chain.calls == 2
    assert service._chain.payloads[1]["input"] != service._chain.payloads[0]["input"]
    assert "可以直接用水冲洗电机" in service._chain.payloads[1]["input"]
    assert "unsupported_claim_rate_exceeded" in service._chain.payloads[1]["input"]
    assert result.verification["passed"] is True
    assert result.answer.startswith("滤网应每周")


def test_rag_generation_uses_current_budget_and_records_actual_model_usage():
    service = RagSummarizeService.__new__(RagSummarizeService)
    service._semantic_cache = None
    service._retrieval_cfg = {}
    service.verify_generation = False
    service.budget_manager = None
    service.max_output_tokens = 1600
    service._chain = type(
        "UsageChain",
        (),
        {
            "invoke": lambda self, payload: AIMessage(
                content="滤网应每周清理。",
                usage_metadata={
                    "input_tokens": 23,
                    "output_tokens": 7,
                    "total_tokens": 30,
                },
            )
        },
    )()
    service._hybrid = FakeHybrid(
        [
            RetrievalCandidate(
                doc_id="manual.pdf#c1",
                document=Document(
                    page_content="滤网应每周清理。",
                    metadata={"source": "manual.pdf", "chunk_id": "c1"},
                ),
                dense_score=0.8,
                fusion_score=0.5,
            )
        ]
    )
    manager = BudgetManager(max_tokens=4000, max_cost=1.0)
    request_id = "req-rag-budget-usage"
    trace_recorder.start_trace(request_id, "session-rag-budget")

    with bind_budget_manager(manager), bind_request_context(request_id=request_id):
        result = service.rag_summarize_result("滤网如何维护")

    assert result.answer.startswith("滤网应每周清理")
    assert manager.snapshot()["used_model_calls"] == 1
    assert manager.used_tokens == 30
    usage_events = [
        event
        for event in trace_recorder.export_trace(request_id)["events"]
        if event["category"] == "diagnostic"
        and event["metadata"].get("type") == "model_usage"
    ]
    assert usage_events[-1]["metadata"]["tokens_in"] == 23
    assert usage_events[-1]["metadata"]["tokens_out"] == 7
    assert usage_events[-1]["metadata"]["cost_mode"] == "actual"


def test_rag_generation_passes_budget_bounded_output_limit_to_model_chain():
    service = RagSummarizeService.__new__(RagSummarizeService)
    service._semantic_cache = None
    service._retrieval_cfg = {}
    service.verify_generation = False
    service.budget_manager = BudgetManager(max_tokens=2000, max_cost=1.0)
    service.max_output_tokens = 321
    service._chain = None
    observed_limits = []

    class LimitedChain:
        def invoke(self, payload):
            return "滤网应每周清理。"

    def build_chain(max_output_tokens=None):
        observed_limits.append(max_output_tokens)
        return LimitedChain()

    service._init_chain = build_chain
    service._hybrid = FakeHybrid(
        [
            RetrievalCandidate(
                doc_id="manual.pdf#c1",
                document=Document(
                    page_content="滤网应每周清理。",
                    metadata={"source": "manual.pdf", "chunk_id": "c1"},
                ),
                dense_score=0.8,
                fusion_score=0.5,
            )
        ]
    )

    service.rag_summarize_result("滤网如何维护")

    assert observed_limits == [321]
    assert service.budget_manager.snapshot()["used_model_calls"] == 1


def test_rag_model_error_does_not_charge_the_full_output_reservation():
    service = RagSummarizeService.__new__(RagSummarizeService)
    service._semantic_cache = None
    service._retrieval_cfg = {}
    service.verify_generation = False
    service.budget_manager = BudgetManager(max_tokens=4000, max_cost=1.0)
    service.max_output_tokens = 1600
    service._chain = type(
        "ExplodingChain",
        (),
        {"invoke": lambda self, payload: (_ for _ in ()).throw(RuntimeError("boom"))},
    )()
    service._hybrid = FakeHybrid(
        [
            RetrievalCandidate(
                doc_id="manual.pdf#c1",
                document=Document(
                    page_content="滤网应每周清理。",
                    metadata={"source": "manual.pdf", "chunk_id": "c1"},
                ),
                dense_score=0.8,
                fusion_score=0.5,
            )
        ]
    )

    with pytest.raises(RuntimeError, match="boom"):
        service.rag_summarize_result("滤网如何维护")

    snapshot = service.budget_manager.snapshot()
    assert snapshot["used_model_calls"] == 1
    assert 0 < snapshot["used_tokens"] < service.max_output_tokens
    assert snapshot["reserved_model_calls"] == 0
