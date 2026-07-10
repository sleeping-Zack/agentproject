from agent.runner import AgentBackendResult, AgentRunner, AgentTask
from agent.memory import ConversationMemory
from services.approval_store import SQLiteApprovalStore
from services.artifact_store import SQLiteArtifactStore
from services.persistence import SQLiteStore


class FakeBackend:
    def __call__(self, task: AgentTask, state):
        return AgentBackendResult(
            answer="建议每周清理尘盒。\n\n引用来源：manual-1",
            evidence=[{"id": "manual-1", "content": "每周清理尘盒"}],
            tool_results=[{"tool": "rag_summarize", "status": "ok"}],
        )


def _runner(tmp_path, max_steps=8, conversation_memory=None):
    return AgentRunner(
        backend=FakeBackend(),
        approval_store=SQLiteApprovalStore(str(tmp_path / "approvals.db")),
        artifact_store=SQLiteArtifactStore(str(tmp_path / "artifacts.db")),
        conversation_memory=conversation_memory,
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


def test_runner_retry_commits_each_message_once(tmp_path):
    store = SQLiteStore(str(tmp_path / "messages.db"))
    memory = ConversationMemory(store=store)
    runner = _runner(tmp_path, conversation_memory=memory)
    task = AgentTask(
        query="怎么保养尘盒",
        session_id="s-retry",
        tenant_id="tenant-a",
        scene="qa",
        request_id="req-retry",
    )

    assert runner.run(task).state.status == "completed"
    assert runner.run(task).state.status == "completed"

    assert store.get_session_messages("s-retry", tenant_id="tenant-a") == [
        {"role": "user", "content": "怎么保养尘盒"},
        {"role": "assistant", "content": "建议每周清理尘盒。\n\n引用来源：manual-1"},
    ]
    assert memory.get_messages("s-retry", tenant_id="tenant-a") == [
        {"role": "user", "content": "怎么保养尘盒"},
        {"role": "assistant", "content": "建议每周清理尘盒。\n\n引用来源：manual-1"},
    ]


def test_runner_does_not_commit_pending_or_failed_final_answers(tmp_path):
    store = SQLiteStore(str(tmp_path / "messages.db"))
    memory = ConversationMemory(store=store)
    pending_runner = _runner(tmp_path, conversation_memory=memory)

    pending = pending_runner.run(
        AgentTask(
            query="生成本月使用记录报告",
            session_id="s-pending",
            tenant_id="tenant-a",
            scene="report",
            request_id="req-pending",
        )
    )

    class FailingBackend:
        def __call__(self, task, state):
            raise RuntimeError("backend failed")

    failed_runner = AgentRunner(
        backend=FailingBackend(),
        approval_store=SQLiteApprovalStore(str(tmp_path / "failed-approvals.db")),
        artifact_store=SQLiteArtifactStore(str(tmp_path / "failed-artifacts.db")),
        conversation_memory=memory,
    )
    failed = failed_runner.run(
        AgentTask(
            query="怎么保养尘盒",
            session_id="s-failed",
            tenant_id="tenant-a",
            request_id="req-failed",
        )
    )

    assert pending.state.status == "pending_approval"
    assert failed.state.status == "failed"
    assert store.get_session_messages("s-pending", tenant_id="tenant-a") == []
    assert store.get_session_messages("s-failed", tenant_id="tenant-a") == []


def test_runner_commits_rejected_answer(tmp_path):
    class UnsupportedBackend:
        def __call__(self, task, state):
            return AgentBackendResult(answer="没有依据的答案")

    store = SQLiteStore(str(tmp_path / "messages.db"))
    memory = ConversationMemory(store=store)
    runner = AgentRunner(
        backend=UnsupportedBackend(),
        approval_store=SQLiteApprovalStore(str(tmp_path / "approvals.db")),
        artifact_store=SQLiteArtifactStore(str(tmp_path / "artifacts.db")),
        conversation_memory=memory,
    )

    result = runner.run(
        AgentTask(
            query="怎么保养尘盒",
            session_id="s-rejected",
            tenant_id="tenant-a",
            scene="qa",
            request_id="req-rejected",
        )
    )

    assert result.state.status == "rejected"
    assert store.get_session_messages("s-rejected", tenant_id="tenant-a") == [
        {"role": "user", "content": "怎么保养尘盒"},
        {"role": "assistant", "content": result.answer},
    ]
