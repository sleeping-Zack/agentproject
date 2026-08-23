from agent.memory import ConversationMemory
from agent.long_term_memory import LongTermMemoryService
from agent.planner import (
    RoutingGoal,
    SemanticRouteProposal,
    TaskRouter,
)
from agent.policies import ToolPolicy
from agent.runner import (
    AgentBackendResult,
    AgentRunner,
    AgentTask,
    AutoRoutingBackend,
)
from agent.answer_schema import AnswerClaim, StructuredAnswer
from agent.budget import current_budget_manager
from agent.verifier import AnswerVerifier, VerifyResult
from agent.tools.registry import build_default_tool_registry
from services.approval_store import SQLiteApprovalStore
from services.artifact_store import SQLiteArtifactStore
from services.memory_store import SQLiteMemoryStore
from services.persistence import SQLiteStore
from observability.context import request_context


class FakeBackend:
    def __call__(self, task: AgentTask, state):
        return AgentBackendResult(
            answer="建议每周清理尘盒。\n\n引用来源：manual-1",
            evidence=[{"id": "manual-1", "content": "每周清理尘盒"}],
            tool_results=[{"tool": "rag_summarize", "status": "ok"}],
        )


class NoToolBackend:
    def __init__(self):
        self.calls = 0

    def __call__(self, task: AgentTask, state):
        self.calls += 1
        return AgentBackendResult(answer="防滑砖建议使用低水量拖地。")


class ReportDataBackend:
    def __call__(self, task: AgentTask, state):
        return AgentBackendResult(
            answer="已读取本人最近设备数据。",
            tool_results=[
                {
                    "tool": "fetch_external_data",
                    "status": "success",
                    "args": {"user_id": "1005", "month": "2025-09"},
                    "content": '{"status":"ok"}',
                }
            ],
        )


def _runner(tmp_path, max_steps=8, conversation_memory=None):
    return AgentRunner(
        backend=FakeBackend(),
        approval_store=SQLiteApprovalStore(str(tmp_path / "approvals.db")),
        artifact_store=SQLiteArtifactStore(str(tmp_path / "artifacts.db")),
        conversation_memory=conversation_memory,
        max_steps=max_steps,
    )


def test_runner_rejects_answer_when_declared_required_tool_was_not_executed(tmp_path):
    backend = NoToolBackend()
    runner = AgentRunner(
        backend=backend,
        approval_store=SQLiteApprovalStore(str(tmp_path / "approvals.db")),
        artifact_store=SQLiteArtifactStore(str(tmp_path / "artifacts.db")),
        max_verification_retries=1,
    )

    result = runner.run(
        AgentTask(
            query="防滑砖地面使用注意事项",
            request_id="req-required-rag",
            required_tools=("rag_summarize",),
        )
    )

    assert result.state.status == "rejected"
    assert result.verifier is not None
    assert result.verifier.missing_required_tools == ["rag_summarize"]
    assert "required_tool_missing" in result.verifier.reasons
    assert result.verifier.citation_coverage == 0.0
    assert backend.calls == 1
    assert "未能完成必要的知识库检索" in result.answer
    assert "回答未通过证据校验" not in result.answer


def test_report_access_arguments_preserve_chinese_month_and_explicit_user():
    args = AgentRunner._report_access_args(
        AgentTask(query="生成用户1008在2026年7月的使用报告")
    )

    assert args == {"user_id": "1008", "month": "2026-07"}


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


def test_runner_binds_shared_budget_while_committing_memory(tmp_path):
    class MemoryProbe:
        manager = None
        request_id = None

        @staticmethod
        def apply_response_policies(answer, **_kwargs):
            return answer

        def commit_turn(self, **_kwargs):
            self.manager = current_budget_manager()
            self.request_id = request_context().request_id

    memory = MemoryProbe()
    runner = AgentRunner(
        backend=FakeBackend(),
        approval_store=SQLiteApprovalStore(str(tmp_path / "approvals.db")),
        artifact_store=SQLiteArtifactStore(str(tmp_path / "artifacts.db")),
        conversation_memory=memory,
    )

    result = runner.run(AgentTask(query="怎么保养尘盒", request_id="req-memory-budget"))

    assert memory.manager is result.state.budget.manager
    assert memory.request_id == "req-memory-budget"


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


def test_runner_allows_own_report_and_uses_real_policy_arguments(tmp_path):
    runner = _runner(tmp_path)

    result = runner.run(
        AgentTask(
            query="生成我的本月使用报告",
            session_id="s-own-report",
            tenant_id="tenant-a",
            user_id="user-1005",
            data_user_id="1005",
            user_role="user",
            scene="report",
            request_id="req-own-report",
        )
    )

    assert result.state.status == "completed"
    assert result.approval_id is None
    policy_call = next(
        call for call in result.state.tool_calls if call.tool_name == "fetch_external_data"
    )
    assert policy_call.args == {"user_id": "1005", "month": "2025-09"}


def test_semantic_route_resolves_report_scene_before_governance(tmp_path):
    registry = build_default_tool_registry(
        ["fetch_external_data", "rag_summarize"]
    )
    policy = ToolPolicy(tool_registry=registry)
    router = TaskRouter(
        semantic_enabled=True,
        semantic_classifier=lambda _query, _context: SemanticRouteProposal(
            execution_mode="react",
            goals=(
                RoutingGoal(
                    id="g1",
                    description="读取本人最近设备数据",
                    required_tools=("fetch_external_data",),
                ),
            ),
            risk="medium",
            confidence=0.94,
            reasons=("semantic_tool_required",),
        ),
    )
    backend = AutoRoutingBackend(
        router=router,
        react_backend=ReportDataBackend(),
        planner_backend=ReportDataBackend(),
        tool_policy=policy,
    )
    runner = AgentRunner(
        backend=backend,
        policy=policy,
        approval_store=SQLiteApprovalStore(str(tmp_path / "approvals.db")),
        artifact_store=SQLiteArtifactStore(str(tmp_path / "artifacts.db")),
    )
    task = AgentTask(
        query="看看我最近的设备数据有什么异常",
        session_id="s-semantic-report",
        tenant_id="tenant-a",
        user_id="user-1005",
        data_user_id="1005",
        user_role="user",
        request_id="req-semantic-report",
    )

    result = runner.run(task)

    assert result.state.status == "completed"
    assert task.scene == "report"
    assert task.routing_decision.required_tools == ("fetch_external_data",)
    assert any(
        call.tool_name == "fetch_external_data" and call.status == "approved"
        for call in result.state.tool_calls
    )


def test_runner_binds_cross_user_approval_to_requester_and_real_arguments(tmp_path):
    runner = _runner(tmp_path)

    result = runner.run(
        AgentTask(
            query="生成用户1001在2025-09的使用报告",
            session_id="s-cross-report",
            tenant_id="tenant-a",
            user_id="user-1005",
            data_user_id="1005",
            user_role="user",
            scene="report",
            request_id="req-cross-report",
        )
    )

    approval = runner.approval_store.get(result.approval_id)
    assert result.state.status == "pending_approval"
    assert approval.principal_id == "user-1005"
    assert approval.args == {"user_id": "1001", "month": "2025-09"}


def test_runner_resumes_only_the_approved_report_scope(tmp_path):
    runner = _runner(tmp_path)
    original = AgentTask(
        query="生成用户1001在2025-09的使用报告",
        session_id="s-approved-report",
        tenant_id="tenant-a",
        user_id="user-1005",
        data_user_id="1005",
        user_role="user",
        scene="report",
        request_id="req-pending-report",
    )
    pending = runner.run(original)
    runner.approval_store.approve(pending.approval_id, decided_by="operator-1")

    resumed = runner.run(
        AgentTask(
            **{
                **original.__dict__,
                "request_id": "req-resumed-report",
                "approval_id": pending.approval_id,
            }
        )
    )
    wrong_scope = runner.run(
        AgentTask(
            **{
                **original.__dict__,
                "query": "生成用户1002在2025-09的使用报告",
                "request_id": "req-wrong-scope",
                "approval_id": pending.approval_id,
            }
        )
    )

    assert resumed.state.status == "completed"
    assert wrong_scope.state.status == "rejected"
    assert wrong_scope.state.error == "approval_arguments_mismatch"


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


class _BudgetHungryPlannerBackend:
    manages_budget = True
    defers_answer_tokens = True

    def __init__(self):
        self.calls = 0

    def __call__(self, task, state):
        self.calls += 1
        state.budget.manager.record_tokens(1100)
        partial = "已核验资料：每次清扫后应清理滚刷和轮组。"
        return AgentBackendResult(
            answer="未经证据支持的综合结论",
            evidence=[
                {
                    "id": "plan-step-t1",
                    "source": "planner",
                    "content": partial,
                }
            ],
            tool_results=[
                {
                    "tool": "plan:rag_qa",
                    "status": "success",
                    "args": {"step_id": "t1"},
                    "content": partial,
                }
            ],
            budget_accounted=True,
            safe_fallback_answer=f"## 已完成部分\n{partial}",
            safe_fallback_structured_answer=StructuredAnswer(
                summary=partial,
                claims=[
                    AnswerClaim(
                        text=partial,
                        evidence_ids=["plan-step-t1"],
                    )
                ],
                citations=["plan-step-t1"],
            ),
        )


class _RetryThenAcceptFallbackVerifier:
    def verify(self, *, answer, **kwargs):
        if answer == "未经证据支持的综合结论":
            return VerifyResult(
                passed=False,
                action="retry",
                score=2.0,
                reasons=["unsupported_claims"],
            )
        return AnswerVerifier().verify(answer=answer, **kwargs)


class _CountingRouter:
    def __init__(self):
        self.calls = 0

    def route(self, query, context=None):
        self.calls += 1
        from agent.planner import TaskRoutingDecision

        return TaskRoutingDecision(
            execution_mode="react",
            complexity_score=1,
            reasons=("test_route",),
        )


class _RepairableBackend:
    def __init__(self):
        self.execution_modes = []
        self.queries = []

    def __call__(self, task, state):
        self.execution_modes.append(task.execution_mode)
        self.queries.append(task.query)
        if task.execution_mode == "direct":
            return AgentBackendResult(answer="清理滚刷。[manual-1]")
        return AgentBackendResult(
            answer="滚刷无需清理。",
            evidence=[{"id": "manual-1", "content": "每次使用后清理滚刷。"}],
            tool_results=[
                {
                    "tool": "rag_summarize",
                    "status": "success",
                    "content": "每次使用后清理滚刷。",
                }
            ],
        )


class _RepairVerifier:
    def verify(self, *, answer, **kwargs):
        if answer == "滚刷无需清理。":
            return VerifyResult(
                passed=False,
                action="retry",
                score=1.0,
                reasons=["unsupported_claims"],
            )
        return VerifyResult(passed=True, action="pass", score=10.0)


def test_verification_retry_repairs_from_evidence_without_rerouting_or_tools(tmp_path):
    router = _CountingRouter()
    react_backend = _RepairableBackend()
    backend = AutoRoutingBackend(
        router=router,
        react_backend=react_backend,
        planner_backend=react_backend,
    )
    runner = AgentRunner(
        backend=backend,
        verifier=_RepairVerifier(),
        approval_store=SQLiteApprovalStore(str(tmp_path / "approvals.db")),
        artifact_store=SQLiteArtifactStore(str(tmp_path / "artifacts.db")),
        max_verification_retries=1,
    )

    result = runner.run(
        AgentTask(query="滚刷需要清理吗", request_id="req-targeted-repair")
    )

    assert result.state.status == "completed"
    assert result.answer == "清理滚刷。[manual-1]"
    assert router.calls == 1
    assert react_backend.execution_modes == ["react", "direct"]
    assert "每次使用后清理滚刷" in react_backend.queries[1]
    assert len(result.state.observations) == 1
    assert len(result.state.tool_calls) == 1


def test_runner_uses_verified_partial_instead_of_starting_impossible_retry(tmp_path):
    backend = _BudgetHungryPlannerBackend()
    runner = AgentRunner(
        backend=backend,
        verifier=_RetryThenAcceptFallbackVerifier(),
        approval_store=SQLiteApprovalStore(str(tmp_path / "approvals.db")),
        artifact_store=SQLiteArtifactStore(str(tmp_path / "artifacts.db")),
        max_tokens=1200,
        max_verification_retries=1,
    )

    result = runner.run(
        AgentTask(
            query="结合资料分析问题并给出步骤",
            request_id="req-budget-limited-retry",
        )
    )

    assert backend.calls == 1
    assert result.state.status == "completed"
    assert result.state.error is None
    assert result.answer.startswith("## 已完成部分")
    assert result.verifier is not None
    assert result.verifier.passed is True


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


def test_runner_persists_explicit_policy_before_answer_and_applies_it_next_turn(tmp_path):
    store = SQLiteStore(str(tmp_path / "messages.db"))
    long_term = LongTermMemoryService(SQLiteMemoryStore(str(tmp_path / "memory.db")))
    memory = ConversationMemory(store=store, long_term_memory=long_term)
    runner = _runner(tmp_path, conversation_memory=memory)

    remembered = runner.run(
        AgentTask(
            query="请记住,以后回答的前两个字必须先说你好",
            session_id="s-policy",
            tenant_id="tenant-a",
            user_id="user-1",
            request_id="req-policy",
        )
    )
    next_answer = runner.run(
        AgentTask(
            query="怎么保养尘盒",
            session_id="s-next",
            tenant_id="tenant-a",
            user_id="user-1",
            request_id="req-next",
        )
    )

    assert remembered.answer.startswith("你好")
    assert remembered.state.steps == []
    assert next_answer.answer.startswith("你好")
    assert long_term.list_memories("tenant-a", "user-1")[0].key == "policy.response_prefix"


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
