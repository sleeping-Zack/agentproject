from agent.runner import AgentBackendResult, AgentRunner, AgentTask
from services.approval_store import SQLiteApprovalStore
from services.artifact_store import SQLiteArtifactStore


class FakeBackend:
    def __call__(self, task: AgentTask, state):
        return AgentBackendResult(
            answer="建议每周清理尘盒。\n\n引用来源：manual-1",
            evidence=[{"id": "manual-1", "content": "每周清理尘盒"}],
            tool_results=[{"tool": "rag_summarize", "status": "ok"}],
        )


def _runner(tmp_path, max_steps=8):
    return AgentRunner(
        backend=FakeBackend(),
        approval_store=SQLiteApprovalStore(str(tmp_path / "approvals.db")),
        artifact_store=SQLiteArtifactStore(str(tmp_path / "artifacts.db")),
        max_steps=max_steps,
    )


def test_runner_completes_and_persists_final_answer(tmp_path):
    runner = _runner(tmp_path)

    result = runner.run(
        AgentTask(
            query="怎么保养尘盒",
            session_id="s-1",
            tenant_id="tenant-a",
            user_role="user",
            scene="qa",
            request_id="req-run-1",
        )
    )

    assert result.state.status == "completed"
    assert result.answer.startswith("建议每周清理")
    assert result.state.artifacts
    artifacts = runner.artifact_store.list_artifacts("req-run-1", tenant_id="tenant-a")
    assert artifacts[0].payload["answer"] == result.answer


def test_runner_pauses_for_sensitive_tool_approval(tmp_path):
    runner = _runner(tmp_path)

    result = runner.run(
        AgentTask(
            query="生成本月使用记录报告",
            session_id="s-1",
            tenant_id="tenant-a",
            user_role="user",
            scene="report",
            request_id="req-approval",
        )
    )

    assert result.state.status == "pending_approval"
    assert result.approval_id
    approval = runner.approval_store.get(result.approval_id)
    assert approval.status == "pending"
    assert approval.tool_name == "fetch_external_data"


def test_runner_blocks_when_budget_is_exhausted(tmp_path):
    runner = _runner(tmp_path, max_steps=0)

    result = runner.run(
        AgentTask(
            query="怎么保养尘盒",
            session_id="s-1",
            tenant_id="tenant-a",
            request_id="req-budget",
        )
    )

    assert result.state.status == "blocked"
    assert result.state.error == "max_steps_exceeded"
