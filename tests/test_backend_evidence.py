from agent.runner import AgentBackendResult, AgentRunner, AgentTask, ReactAgentBackend
from observability.tracing import trace_recorder


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
            metadata={"args_hash": "abc", "redacted_args": {"query": query}},
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
    assert result.evidence[0]["id"] == "manual-1"
    assert result.evidence[0]["content"] == "滤网每周清理"


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
