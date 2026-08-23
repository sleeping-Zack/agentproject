from agent.runner import AgentBackendResult, AgentRunner, AgentTask, ReactAgentBackend
from observability.tracing import trace_recorder
from services.approval_store import SQLiteApprovalStore
from services.artifact_store import SQLiteArtifactStore


class TraceAgent:
    def execute_stream(self, query, session_id, request_id, tenant_id, **kwargs):
        trace_recorder.start_trace(request_id, session_id)
        with trace_recorder.span(
            request_id,
            category="rag",
            name="evidence",
            metadata={
                "evidence": [
                    {
                        "id": "manual-1",
                        "source": "manual.pdf",
                        "content": "滤网每周清理",
                        "metadata": {"chunk_id": "c1"},
                        "score": 0.82,
                    }
                ]
            },
        ):
            pass
        with trace_recorder.span(
            request_id,
            category="tool",
            name="rag_summarize",
            metadata={
                "args_hash": "abc",
                "redacted_args": {"query": query},
                "result": "滤网每周清理。引用来源：manual-1",
                "result_truncated": False,
            },
        ):
            yield "引用来源：manual-1"


def test_react_backend_returns_tool_results_and_model_name():
    backend = ReactAgentBackend(agent=TraceAgent())

    result = backend(
        AgentTask(query="怎么保养滤网", session_id="s", request_id="req-backend"),
        state=None,
    )

    assert result.answer == "引用来源：manual-1"
    assert result.model_name
    assert result.tool_results[0]["tool"] == "rag_summarize"
    assert "args_hash" in result.tool_results[0]["metadata"]
    assert result.tool_results[0]["content"] == "滤网每周清理。引用来源：manual-1"
    assert result.evidence[0]["id"] == "manual-1"
    assert result.evidence[0]["content"] == "滤网每周清理"


class RejectedRagTraceAgent:
    def execute_stream(self, query, session_id, request_id, tenant_id, **kwargs):
        trace_recorder.start_trace(request_id, session_id)
        with trace_recorder.span(
            request_id,
            category="rag",
            name="evidence",
            metadata={
                "business_status": "verification_failed",
                "evidence": [
                    {
                        "id": "manual-1",
                        "source": "manual.pdf",
                        "content": "主刷卡住时先关机并清理异物。",
                    }
                ],
                "verification": {
                    "passed": False,
                    "action": "refuse",
                    "reasons": ["unsupported_claim_rate_exceeded"],
                },
            },
        ):
            pass
        with trace_recorder.span(
            request_id,
            category="tool",
            name="rag_summarize",
            metadata={
                "redacted_args": {"query": query},
                "result": "请求未执行：生成结果未通过证据一致性校验。",
            },
        ):
            yield "请求未执行：生成结果未通过证据一致性校验。"


def test_runner_does_not_report_rejected_rag_result_as_success(tmp_path):
    runner = AgentRunner(
        backend=ReactAgentBackend(agent=RejectedRagTraceAgent()),
        approval_store=SQLiteApprovalStore(str(tmp_path / "approvals.db")),
        artifact_store=SQLiteArtifactStore(str(tmp_path / "artifacts.db")),
        max_verification_retries=0,
    )

    result = runner.run(
        AgentTask(
            query="主刷卡住了怎么办",
            session_id="s-rag-rejected",
            request_id="req-rag-rejected",
        )
    )

    assert result.state.tool_calls[0].status == "verification_failed"


def test_runner_explains_missing_price_evidence_as_a_normal_knowledge_gap(tmp_path):
    class EmptyPriceBackend:
        def __init__(self):
            self.calls = 0

        def __call__(self, _task, _state):
            self.calls += 1
            return AgentBackendResult(
                answer="X9 Pro 是最贵的机器人。",
                tool_results=[
                    {
                        "tool": "rag_summarize",
                        "status": "empty",
                        "content": "知识库中没有足够证据支持回答该问题。",
                        "metadata": {
                            "business_status": "empty",
                            "verification": {
                                "passed": False,
                                "reasons": ["retrieval_relevance_below_threshold"],
                            },
                        },
                    }
                ],
            )

    backend = EmptyPriceBackend()
    runner = AgentRunner(
        backend=backend,
        approval_store=SQLiteApprovalStore(str(tmp_path / "approvals.db")),
        artifact_store=SQLiteArtifactStore(str(tmp_path / "artifacts.db")),
        max_verification_retries=1,
    )

    result = runner.run(
        AgentTask(
            query="讲一下最贵的机器人是哪一个",
            request_id="req-price-knowledge-gap",
        )
    )

    assert backend.calls == 1
    assert result.state.status == "completed"
    assert result.verifier is not None and result.verifier.passed is True
    assert result.verifier.reasons == ["knowledge_irrelevant"]
    assert "各型号" in result.answer
    assert "价格" in result.answer
    assert "无法判断哪款" in result.answer
    assert "回答未通过证据校验" not in result.answer
    assert "X9 Pro" not in result.answer
    assert result.state.tool_calls[0].status == "empty"


def test_runner_keeps_harmful_contradiction_as_a_safety_rejection(tmp_path):
    class HarmfulBackend:
        def __call__(self, _task, _state):
            return AgentBackendResult(
                answer="可以直接用水冲洗电机。[safety]",
                evidence=[
                    {
                        "id": "safety",
                        "source": "安全手册",
                        "content": "严禁用水冲洗电机。",
                    }
                ],
                tool_results=[
                    {
                        "tool": "rag_summarize",
                        "status": "success",
                        "content": "严禁用水冲洗电机。[safety]",
                    }
                ],
            )

    runner = AgentRunner(
        backend=HarmfulBackend(),
        approval_store=SQLiteApprovalStore(str(tmp_path / "approvals.db")),
        artifact_store=SQLiteArtifactStore(str(tmp_path / "artifacts.db")),
        max_verification_retries=0,
    )

    result = runner.run(
        AgentTask(query="电机如何清洗", request_id="req-harmful-grounding")
    )

    assert result.state.status == "rejected"
    assert result.verifier is not None
    assert result.verifier.action == "refuse"
    assert "evidence_contradiction" in result.verifier.reasons
    assert "harmful_instruction" in result.verifier.reasons
    assert "用水冲洗电机" not in result.answer
    assert "知识库没有收录各型号" not in result.answer


class RetryTraceAgent:
    def __init__(self):
        self.calls = 0

    def execute_stream(self, query, session_id, request_id, tenant_id, **kwargs):
        self.calls += 1
        if self.calls == 1:
            with trace_recorder.span(
                request_id,
                category="rag",
                name="evidence",
                metadata={
                    "business_status": "success",
                    "evidence": [
                        {
                            "id": "manual-1",
                            "source": "manual.pdf",
                            "content": "主刷卡住时先关机清理异物。",
                        }
                    ],
                },
            ):
                pass
            with trace_recorder.span(
                request_id,
                category="tool",
                name="rag_summarize",
                metadata={
                    "redacted_args": {"query": query},
                    "result": "主刷卡住时先关机清理异物。",
                },
            ):
                yield "主刷卡住时先关机清理异物。"
            return
        yield "这是第二轮无工具回答。"


class VerifiedRagSummaryTraceAgent:
    def __init__(self):
        self.calls = 0

    def execute_stream(self, query, session_id, request_id, tenant_id, **kwargs):
        self.calls += 1
        trace_recorder.start_trace(request_id, session_id)
        evidence = [
            {
                "id": "manual-cleaning",
                "source": "扫地机器人手册.pdf",
                "content": "清理滚刷缠绕物，并检查主吸口和风道是否堵塞。",
            },
            {
                "id": "manual-unrelated",
                "source": "故障排除.txt",
                "content": "拖布支架生锈时应清理锈迹。",
            },
        ]
        verified_summary = (
            "先清理滚刷缠绕物，再检查主吸口和风道是否堵塞。"
            "[manual-cleaning]"
        )
        with trace_recorder.span(
            request_id,
            category="rag",
            name="evidence",
            metadata={
                "business_status": "success",
                "evidence": evidence,
                "verification": {"passed": True, "action": "accept"},
            },
        ):
            pass
        with trace_recorder.span(
            request_id,
            category="tool",
            name="rag_summarize",
            metadata={
                "redacted_args": {"query": query},
                "result": verified_summary,
                "result_truncated": False,
            },
        ):
            pass
        yield "先清理滚刷和风道，再观察清洁效果。"


def test_runner_reuses_verified_rag_summary_when_outer_answer_drops_citations(tmp_path):
    agent = VerifiedRagSummaryTraceAgent()
    backend = ReactAgentBackend(agent=agent)
    runner = AgentRunner(
        backend=backend,
        approval_store=SQLiteApprovalStore(str(tmp_path / "approvals.db")),
        artifact_store=SQLiteArtifactStore(str(tmp_path / "artifacts.db")),
        max_verification_retries=0,
    )

    result = runner.run(
        AgentTask(
            query="扫地机器人最近清洁效果下降，应该如何系统排查？",
            session_id="s-verified-rag",
            request_id="req-verified-rag",
        )
    )

    assert agent.calls == 1
    assert result.state.status == "completed"
    assert result.answer == (
        "先清理滚刷缠绕物，再检查主吸口和风道是否堵塞。[manual-cleaning]"
    )
    assert "拖布支架生锈" not in result.answer
    assert "生成式回答未通过校验" not in result.answer
    assert result.verifier is not None and result.verifier.passed is True


def test_rag_summary_fallback_requires_successful_inner_verification():
    answer, _structured, strategy = ReactAgentBackend._evidence_fallback(
        [
            {
                "id": "manual-1",
                "source": "manual.pdf",
                "content": "滤网应使用干布清理。",
            }
        ],
        [
            {
                "tool": "rag_summarize",
                "status": "verification_failed",
                "content": "可以直接用水冲洗电机。[manual-1]",
                "metadata": {"verification": {"passed": False}},
            }
        ],
    )

    assert strategy == "verified_evidence_excerpt"
    assert "可以直接用水冲洗电机" not in answer
    assert "滤网应使用干布清理" in answer


def test_numeric_rag_citation_is_not_reused_across_multiple_retrievals():
    answer, _structured, strategy = ReactAgentBackend._evidence_fallback(
        [
            {"id": "first-doc", "source": "a.txt", "content": "清理滚刷。"},
            {"id": "second-doc", "source": "b.txt", "content": "清理风道。"},
        ],
        [
            {
                "tool": "rag_summarize",
                "status": "success",
                "content": "清理滚刷。[1]",
                "metadata": {"verification": {"passed": True}},
            },
            {
                "tool": "rag_summarize",
                "status": "success",
                "content": "清理风道。[1]",
                "metadata": {"verification": {"passed": True}},
            },
        ],
    )

    assert strategy == "verified_evidence_excerpt"
    assert answer.startswith("生成式回答未通过校验")


def test_runner_stops_after_first_rag_call_when_evidence_fallback_is_verified(tmp_path):
    agent = RetryTraceAgent()
    runner = AgentRunner(
        backend=ReactAgentBackend(agent=agent),
        approval_store=SQLiteApprovalStore(str(tmp_path / "approvals.db")),
        artifact_store=SQLiteArtifactStore(str(tmp_path / "artifacts.db")),
        max_verification_retries=1,
    )

    result = runner.run(
        AgentTask(
            query="主刷卡住了怎么办",
            session_id="s-tool-retry",
            request_id="req-tool-retry",
        )
    )

    assert agent.calls == 1
    assert len(result.state.tool_calls) == 1
    assert result.state.tool_calls[0].tool_name == "rag_summarize"
    assert result.state.status == "completed"
    assert "manual-1" in result.answer
    verifier_events = [
        event
        for event in trace_recorder.export_trace("req-tool-retry")["events"]
        if event["category"] == "diagnostic"
        and event["metadata"].get("type") == "verifier"
    ]
    assert verifier_events[-1]["metadata"]["status"] == "ok"


class UsageTraceAgent:
    def execute_stream(self, query, session_id, request_id, tenant_id, **kwargs):
        trace_recorder.start_trace(request_id, session_id)
        trace_recorder.record_diagnostic_event(
            request_id=request_id,
            step_id="model-usage",
            event_type="model_usage",
            status="ok",
            latency_ms=0.0,
            tokens_in=11,
            tokens_out=13,
            cost=0.024,
            model_name="mock-model",
            cost_mode="actual",
        )
        yield "answer"


def test_react_backend_returns_actual_usage_from_trace():
    backend = ReactAgentBackend(agent=UsageTraceAgent())

    result = backend(
        AgentTask(query="hello", session_id="s", request_id="req-usage"),
        state=None,
    )

    assert result.tokens_in == 11
    assert result.tokens_out == 13
    assert result.cost == 0.024
    assert result.cost_mode == "actual"


class ToolResultBackend:
    def __call__(self, task, state):
        return AgentBackendResult(
            answer="ok",
            tool_results=[
                {"tool": "rag_summarize", "status": "success"},
                {"tool": "get_weather", "status": "success"},
            ],
        )


def test_runner_blocks_when_backend_tool_results_exceed_budget(tmp_path):
    from services.approval_store import SQLiteApprovalStore
    from services.artifact_store import SQLiteArtifactStore

    runner = AgentRunner(
        backend=ToolResultBackend(),
        approval_store=SQLiteApprovalStore(str(tmp_path / "approvals.db")),
        artifact_store=SQLiteArtifactStore(str(tmp_path / "artifacts.db")),
        max_tool_calls=1,
    )

    result = runner.run(AgentTask(query="hello", request_id="req-tool-budget"))

    assert result.state.status == "blocked"
    assert result.state.error == "max_tool_calls_exceeded"


class ActualUsageBackend:
    def __call__(self, task, state):
        return AgentBackendResult(
            answer="ok",
            tokens_in=10,
            tokens_out=20,
            cost=0.123,
            cost_mode="actual",
        )


class OverBudgetToolBackend:
    def __call__(self, task, state):
        return AgentBackendResult(
            answer="天气查询完成。",
            tool_results=[
                {
                    "tool": "get_weather",
                    "status": "success",
                    "args": {"city": "深圳"},
                    "content": "深圳多云，30℃。",
                }
            ],
            tokens_in=100,
            tokens_out=50,
            cost=0.01,
            cost_mode="actual",
        )


def test_runner_preserves_executed_tool_call_when_model_usage_exceeds_budget(tmp_path):
    runner = AgentRunner(
        backend=OverBudgetToolBackend(),
        approval_store=SQLiteApprovalStore(str(tmp_path / "approvals.db")),
        artifact_store=SQLiteArtifactStore(str(tmp_path / "artifacts.db")),
        max_tokens=100,
        max_model_output_tokens=10,
    )

    result = runner.run(
        AgentTask(query="深圳天气", request_id="req-over-budget-tool")
    )

    assert result.state.status == "blocked"
    assert result.state.error == "max_tokens_exceeded"
    assert len(result.state.tool_calls) == 1
    assert result.state.tool_calls[0].tool_name == "get_weather"
    assert result.state.tool_calls[0].result == "深圳多云，30℃。"


def test_runner_prefers_actual_backend_usage_over_estimate(tmp_path):
    from services.approval_store import SQLiteApprovalStore
    from services.artifact_store import SQLiteArtifactStore

    runner = AgentRunner(
        backend=ActualUsageBackend(),
        approval_store=SQLiteApprovalStore(str(tmp_path / "approvals.db")),
        artifact_store=SQLiteArtifactStore(str(tmp_path / "artifacts.db")),
    )

    result = runner.run(AgentTask(query="hello", request_id="req-actual-usage"))

    assert result.state.budget.used_tokens == 30
    assert result.state.budget.used_cost == 0.123
