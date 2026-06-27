from agent.runner import AgentBackendResult, AgentRunner, AgentTask
from services.approval_store import SQLiteApprovalStore
from services.artifact_store import SQLiteArtifactStore


class LongAnswerBackend:
    def __call__(self, task, state):
        return AgentBackendResult(answer="x" * 200)


def test_runner_blocks_when_estimated_tokens_exceed_budget(tmp_path):
    runner = AgentRunner(
        backend=LongAnswerBackend(),
        approval_store=SQLiteApprovalStore(str(tmp_path / "approvals.db")),
        artifact_store=SQLiteArtifactStore(str(tmp_path / "artifacts.db")),
        max_tokens=10,
    )

    result = runner.run(AgentTask(query="hello", request_id="req-token-budget"))

    assert result.state.status == "blocked"
    assert result.state.error == "max_tokens_exceeded"
