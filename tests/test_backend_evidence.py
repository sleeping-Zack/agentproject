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
