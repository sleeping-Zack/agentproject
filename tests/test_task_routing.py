from types import SimpleNamespace
from uuid import uuid4

import pytest

from agent.budget import BudgetManager
from agent.planner import (
    PlanExecutor,
    PlanRunResult,
    PlannerAgent,
    RoutingContext,
    RoutingGoal,
    SemanticRouteProposal,
    SemanticTaskClassifier,
    SubTask,
    SubTaskResult,
    TaskPlanner,
    TaskRouter,
    TaskRoutingDecision,
)
from agent.react_agent import ReactAgent
from agent.runner import (
    AgentBackendResult,
    AgentTask,
    AutoRoutingBackend,
    PlannerAgentBackend,
)
from agent.verifier import AnswerVerifier
from observability.event_bus import event_bus
from observability.tracing import trace_recorder


SIMPLE_QUERY = "扫地机器人有什么优点"
COMPLEX_QUERY = "结合本月使用报告和知识库保养建议，分析问题并制定分步骤维护计划"


class _FixedRouter:
    def __init__(self, execution_mode: str):
        self.execution_mode = execution_mode
        self.queries = []

    def route(self, query: str) -> TaskRoutingDecision:
        self.queries.append(query)
        return TaskRoutingDecision(
            execution_mode=self.execution_mode,
            complexity_score=7 if self.execution_mode == "plan_execute" else 1,
            reasons=["test decision"],
        )


class _SpyBackend:
    def __init__(self, answer: str):
        self.answer = answer
        self.calls = []

    def __call__(self, task, state):
        self.calls.append((task, state))
        return AgentBackendResult(answer=self.answer)


@pytest.mark.parametrize(
    ("query", "expected_mode"),
    [
        (SIMPLE_QUERY, "direct"),
        (COMPLEX_QUERY, "plan_execute"),
    ],
)
def test_task_router_selects_execution_mode_from_the_chat_message(
    query,
    expected_mode,
):
    decision = TaskRouter().route(query)

    assert decision.execution_mode == expected_mode
    assert decision.complexity_score >= 0
    assert decision.reasons


def test_complex_query_scores_higher_than_an_ordinary_question():
    router = TaskRouter(semantic_enabled=False)

    ordinary = router.route(SIMPLE_QUERY)
    complex_task = router.route(COMPLEX_QUERY)

    assert complex_task.complexity_score > ordinary.complexity_score


def test_semantic_router_understands_an_implicit_cross_capability_task():
    captured = {}

    def classify(query, context):
        captured["query"] = query
        captured["history"] = context.recent_messages
        return SemanticRouteProposal(
            execution_mode="plan_execute",
            goals=(
                RoutingGoal(
                    id="g1",
                    description="读取最近的设备使用记录并识别异常",
                    required_tools=("fetch_external_data",),
                ),
                RoutingGoal(
                    id="g2",
                    description="检索官方维护资料并形成处理顺序",
                    required_tools=("rag_summarize",),
                    depends_on=("g1",),
                ),
            ),
            risk="medium",
            confidence=0.91,
            reasons=("semantic_multiple_goals", "semantic_dependencies"),
        )

    router = TaskRouter(semantic_classifier=classify, semantic_enabled=True)
    context = RoutingContext(
        available_tools=("fetch_external_data", "rag_summarize"),
        recent_messages=("用户：最近清洁效果下降",),
        remaining_steps=8,
        remaining_tool_calls=5,
        remaining_tokens=12000,
    )

    decision = router.route(
        "把这件事彻底搞明白，给我一个能落地的处理顺序",
        context=context,
    )

    assert decision.execution_mode == "plan_execute"
    assert decision.decision_source == "semantic_model"
    assert decision.required_tools == ("fetch_external_data", "rag_summarize")
    assert decision.confidence == pytest.approx(0.91)
    assert captured["history"] == context.recent_messages


def test_semantic_classifier_parses_strict_json_and_accounts_for_its_model_call():
    manager = BudgetManager(max_tokens=4000, max_cost=1.0)
    response = SimpleNamespace(
        content="""```json
        {
          "execution_mode": "plan_execute",
          "goals": [
            {
              "id": "g1",
              "description": "读取设备数据",
              "required_tools": ["fetch_external_data"],
              "depends_on": []
            },
            {
              "id": "g2",
              "description": "检索维护资料",
              "required_tools": ["rag_summarize"],
              "depends_on": ["g1"]
            }
          ],
          "risk": "medium",
          "confidence": 0.92,
          "reasons": ["semantic_multiple_goals", "semantic_dependencies"]
        }
        ```""",
        usage_metadata={"input_tokens": 80, "output_tokens": 90},
        response_metadata={},
    )
    classifier = SemanticTaskClassifier(
        model_invoker=lambda _prompt, _context, _max_tokens: response,
    )

    proposal = classifier(
        "处理隐式复杂任务",
        RoutingContext(
            available_tools=("fetch_external_data", "rag_summarize"),
            budget_manager=manager,
        ),
    )

    assert proposal.execution_mode == "plan_execute"
    assert proposal.goals[1].depends_on == ("g1",)
    assert manager.snapshot()["used_model_calls"] == 1
    assert manager.used_tokens == 170


def test_invalid_semantic_output_uses_deterministic_fallback():
    classifier = SemanticTaskClassifier(
        model_invoker=lambda _prompt, _context, _max_tokens: SimpleNamespace(
            content="这不是 JSON"
        ),
    )
    router = TaskRouter(
        semantic_classifier=classifier,
        semantic_enabled=True,
    )

    decision = router.route(
        "扫地机器人怎么保养",
        context=RoutingContext(available_tools=("rag_summarize",)),
    )

    assert decision.execution_mode == "react"
    assert decision.decision_source == "deterministic_fallback"
    assert "semantic_router_fallback" in decision.reasons


def test_semantic_failure_never_downgrades_an_uncertain_request_to_no_tool_mode():
    router = TaskRouter(
        semantic_classifier=lambda _query, _context: (_ for _ in ()).throw(
            ValueError("invalid_semantic_route")
        ),
        semantic_enabled=True,
    )

    decision = router.route(
        "把前面的现象和资料放在一起判断后续怎么处理",
        context=RoutingContext(),
    )

    assert decision.execution_mode == "react"
    assert decision.transition == "direct_to_react"
    assert decision.confidence == pytest.approx(0.35)


def test_semantic_router_respects_negation_instead_of_matching_plan_words():
    router = TaskRouter(
        semantic_classifier=lambda _query, _context: SemanticRouteProposal(
            execution_mode="direct",
            goals=(RoutingGoal(id="g1", description="给出一句简短结论"),),
            confidence=0.96,
            reasons=("semantic_single_goal",),
        ),
        semantic_enabled=True,
    )

    decision = router.route(
        "不要分步骤，也不要制定计划，只用一句话回答",
        context=RoutingContext(),
    )

    assert decision.execution_mode == "direct"
    assert decision.decision_source == "semantic_model"


def test_low_confidence_semantic_decision_safely_falls_back_to_react():
    router = TaskRouter(
        semantic_classifier=lambda _query, _context: SemanticRouteProposal(
            execution_mode="plan_execute",
            goals=(
                RoutingGoal(id="g1", description="目标一"),
                RoutingGoal(id="g2", description="目标二"),
            ),
            confidence=0.42,
        ),
        semantic_enabled=True,
    )

    decision = router.route("含义不清楚的请求", context=RoutingContext())

    assert decision.execution_mode == "react"
    assert "low_semantic_confidence" in decision.reasons


def test_capability_validator_downgrades_an_unexecutable_plan():
    router = TaskRouter(
        semantic_classifier=lambda _query, _context: SemanticRouteProposal(
            execution_mode="plan_execute",
            goals=(
                RoutingGoal(
                    id="g1",
                    description="读取外部工单",
                    required_tools=("read_ticket_system",),
                ),
                RoutingGoal(id="g2", description="提出修复方案", depends_on=("g1",)),
            ),
            confidence=0.95,
        ),
        semantic_enabled=True,
    )

    decision = router.route(
        "读取工单并修复",
        context=RoutingContext(
            available_tools=("rag_summarize",),
            remaining_steps=8,
            remaining_tool_calls=5,
            remaining_tokens=12000,
        ),
    )

    assert decision.execution_mode == "react"
    assert decision.unavailable_tools == ("read_ticket_system",)
    assert decision.transition == "plan_execute_to_react"
    assert "required_capability_unavailable" in decision.reasons


def test_verification_feedback_can_escalate_and_downgrade_execution():
    goals = (
        RoutingGoal(id="g1", description="读取报告"),
        RoutingGoal(id="g2", description="检索资料", depends_on=("g1",)),
    )
    previous_react = TaskRoutingDecision(
        execution_mode="react",
        complexity_score=6,
        goals=goals,
        confidence=0.61,
    )
    router = TaskRouter(semantic_enabled=False)
    retry_context = RoutingContext(
        remaining_steps=6,
        remaining_tool_calls=4,
        remaining_tokens=9000,
        prior_decision=previous_react,
        verification_feedback={
            "action": "retry",
            "reasons": ["unsupported_claim_rate_exceeded"],
        },
    )

    escalated = router.route("继续完成任务", context=retry_context)

    assert escalated.execution_mode == "plan_execute"
    assert escalated.transition == "react_to_plan_execute"
    assert escalated.decision_source == "runtime_feedback"

    previous_plan = TaskRoutingDecision(
        execution_mode="plan_execute",
        complexity_score=8,
        goals=goals,
        confidence=0.9,
    )
    downgraded = router.route(
        "继续完成任务",
        context=RoutingContext(
            prior_decision=previous_plan,
            verification_feedback={"action": "retry", "reasons": ["citation_invalid"]},
        ),
    )

    assert downgraded.execution_mode == "react"
    assert downgraded.transition == "plan_execute_to_react"
    assert downgraded.decision_source == "runtime_feedback"


def test_plan_retry_is_not_downgraded_when_execution_budget_is_insufficient():
    previous = TaskRoutingDecision(
        execution_mode="plan_execute",
        complexity_score=8,
        goals=(
            RoutingGoal(id="g1", description="读取资料"),
            RoutingGoal(id="g2", description="形成结论", depends_on=("g1",)),
        ),
        confidence=0.9,
    )

    decision = TaskRouter(semantic_enabled=False).route(
        "继续完成任务",
        context=RoutingContext(
            remaining_steps=2,
            remaining_tool_calls=0,
            remaining_tokens=200,
            prior_decision=previous,
            verification_feedback={"action": "retry"},
        ),
    )

    assert decision.execution_mode == "plan_execute"
    assert decision.transition is None
    assert "verification_retry_budget_insufficient" in decision.reasons


def test_planner_uses_semantic_goals_and_dependencies_instead_of_keywords():
    decision = TaskRoutingDecision(
        execution_mode="plan_execute",
        complexity_score=8,
        goals=(
            RoutingGoal(
                id="goal-a",
                description="读取设备数据",
                required_tools=("fetch_external_data",),
            ),
            RoutingGoal(
                id="goal-b",
                description="查询厂家资料",
                required_tools=("rag_summarize",),
                depends_on=("goal-a",),
                tool_input="主刷旋转速度变慢的检测与修复方法",
            ),
        ),
        required_tools=("fetch_external_data", "rag_summarize"),
        confidence=0.93,
        decision_source="semantic_model",
    )

    plan = TaskPlanner().plan("没有任何领域关键词的原始请求", routing_decision=decision)

    assert [(item.kind, item.description) for item in plan] == [
        ("report", "读取设备数据"),
        ("rag_qa", "查询厂家资料"),
    ]
    assert plan[1].depends_on == ["t1"]
    assert plan[1].args["query"] == "主刷旋转速度变慢的检测与修复方法"


def test_planner_synthesis_must_be_supported_by_dependency_results():
    dependencies = {
        "t1": "设备报告显示手动操作占比为90%。",
        "t2": "知识库建议检查主刷是否缠绕杂物。",
    }

    grounded = ReactAgent._verify_dependency_synthesis(
        "诊断清扫变慢",
        "设备报告显示手动操作占比为90%。[t1]\n"
        "知识库建议检查主刷是否缠绕杂物。[t2]",
        dependencies,
    )
    fabricated = ReactAgent._verify_dependency_synthesis(
        "诊断清扫变慢",
        "请在应用中开启不存在于证据中的防滑砖专用模式。[t1]",
        dependencies,
    )

    assert grounded is None
    assert fabricated is not None
    assert fabricated.startswith("subtask_verification_failed:")


@pytest.mark.parametrize(
    ("execution_mode", "expected_answer", "react_calls", "planner_calls"),
    [
        ("direct", "react answer", 1, 0),
        ("react", "react answer", 1, 0),
        ("plan_execute", "planned answer", 0, 1),
    ],
)
def test_auto_routing_backend_invokes_only_the_selected_execution_path(
    execution_mode,
    expected_answer,
    react_calls,
    planner_calls,
):
    router = _FixedRouter(execution_mode)
    react_backend = _SpyBackend("react answer")
    planner_backend = _SpyBackend("planned answer")
    backend = AutoRoutingBackend(
        router=router,
        react_backend=react_backend,
        planner_backend=planner_backend,
    )
    task = AgentTask(query="same chat entry point")
    state = SimpleNamespace()
    trace_recorder.start_trace(task.request_id, task.session_id)

    result = backend(task, state)

    assert result.answer == expected_answer
    assert router.queries == [task.query]
    assert len(react_backend.calls) == react_calls
    assert len(planner_backend.calls) == planner_calls


def test_auto_routing_backend_publishes_a_structured_routing_event():
    request_id = str(uuid4())
    router = _FixedRouter("plan_execute")
    backend = AutoRoutingBackend(
        router=router,
        react_backend=_SpyBackend("react answer"),
        planner_backend=_SpyBackend("planned answer"),
    )
    task = AgentTask(
        query=COMPLEX_QUERY,
        request_id=request_id,
        emit_events=True,
    )
    event_bus.open(request_id)
    trace_recorder.start_trace(request_id, task.session_id)

    try:
        backend(task, SimpleNamespace())
        routing_events = [
            item
            for item in event_bus.replay(request_id)
            if item.event_type == "routing_completed"
        ]

        assert len(routing_events) == 1
        payload = routing_events[0].payload
        assert payload["execution_mode"] == "plan_execute"
        assert payload["complexity_score"] == 7
        assert payload["reasons"] == ["test decision"]
    finally:
        event_bus.discard(request_id)


def test_planner_agent_publishes_plan_lifecycle_in_execution_order():
    planner = TaskPlanner(
        llm_planner=lambda _query: [
            SubTask(id="t1", kind="report", description="读取使用报告"),
            SubTask(id="t2", kind="rag_qa", description="检索保养建议"),
        ]
    )
    executor = PlanExecutor(max_workers=1)
    executor.register_handler(
        "report",
        lambda task: SubTaskResult(
            id=task.id,
            kind=task.kind,
            success=True,
            content="报告结果",
        ),
    )
    executor.register_handler(
        "rag_qa",
        lambda task: SubTaskResult(
            id=task.id,
            kind=task.kind,
            success=True,
            content="知识库结果",
        ),
    )
    agent = PlannerAgent(planner=planner, executor=executor)
    request_id = str(uuid4())
    events = []
    trace_recorder.start_trace(request_id, "session-a")

    result = agent.run(
        COMPLEX_QUERY,
        request_id=request_id,
        event_callback=lambda event_type, payload: events.append(
            (event_type, payload)
        ),
    )

    event_types = [event_type for event_type, _payload in events]
    assert result.answer
    assert event_types == [
        "plan_created",
        "plan_step_started",
        "plan_step_completed",
        "plan_step_started",
        "plan_step_completed",
        "plan_completed",
    ]
    assert [payload["id"] for _event_type, payload in events[1:5]] == [
        "t1",
        "t1",
        "t2",
        "t2",
    ]


def test_planner_backend_propagates_chat_and_governance_context():
    class _PlanningAgent:
        def __init__(self):
            self.call = None

        def run_plan(self, query, **kwargs):
            self.call = (query, kwargs)
            return PlanRunResult(
                plan=[SubTask(id="t1", kind="report", description="生成使用报告")],
                results=[
                    SubTaskResult(
                        id="t1",
                        kind="report",
                        success=True,
                        content="已完成使用报告。\n已生成维护计划。",
                    )
                ],
                answer="分步骤维护计划",
            )

    request_id = str(uuid4())
    planning_agent = _PlanningAgent()
    backend = PlannerAgentBackend(agent=planning_agent)
    task = AgentTask(
        query=COMPLEX_QUERY,
        session_id="session-a",
        tenant_id="tenant-a",
        user_id="user-a",
        data_user_id="data-user-a",
        user_role="operator",
        scene="report",
        request_id=request_id,
        approval_id="approval-a",
        emit_events=True,
    )
    manager = object()
    state = SimpleNamespace(
        budget=SimpleNamespace(manager=manager),
    )
    trace_recorder.start_trace(request_id, task.session_id)

    result = backend(task, state)

    assert result.answer == "分步骤维护计划"
    assert result.structured_answer is not None
    assert result.structured_answer.citations == ["plan-step-t1"]
    assert len(result.structured_answer.claims) == 2
    assert result.evidence[-1]["metadata"]["lineage"] == (
        "verified_source_subtask_output"
    )
    verification = AnswerVerifier().verify(
        query=task.query,
        answer=result.answer,
        evidence=result.evidence,
        scene="report",
        tool_results=result.tool_results,
        structured_answer=result.structured_answer,
    )
    assert verification.passed is True
    assert planning_agent.call == (
        COMPLEX_QUERY,
        {
            "request_id": request_id,
            "tenant_id": "tenant-a",
            "budget_manager": manager,
            "session_id": "session-a",
            "user_id": "user-a",
            "data_user_id": "data-user-a",
            "user_role": "operator",
            "scene": "report",
            "approval_id": "approval-a",
            "emit_events": True,
        },
    )
