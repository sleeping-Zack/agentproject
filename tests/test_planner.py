from threading import Barrier

from agent.budget import BudgetManager
from agent.planner import (
    PlanExecutor,
    PlannerAgent,
    ResultAggregator,
    SubTask,
    SubTaskResult,
    TaskPlanner,
)
from agent.policies import PlanValidator, Replanner


def test_planner_decomposes_compound_query():
    planner = TaskPlanner()

    plan = planner.plan("帮我查一下杭州的天气，并给我一份本月使用报告")
    kinds = [task.kind for task in plan]

    assert "weather" in kinds
    assert "report" in kinds


def test_planner_falls_back_to_generic_for_unknown_query():
    planner = TaskPlanner()

    plan = planner.plan("随便聊两句")

    assert len(plan) == 1
    assert plan[0].kind == "generic"


def test_executor_runs_handlers_and_collects_results():
    executor = PlanExecutor(max_workers=2)

    def handler_a(task):
        return SubTaskResult(id=task.id, kind=task.kind, success=True, content="A 完成")

    def handler_b(task):
        return SubTaskResult(id=task.id, kind=task.kind, success=True, content="B 完成")

    executor.register_handler("a", handler_a)
    executor.register_handler("b", handler_b)

    plan = [SubTask(id="t1", kind="a", description=""), SubTask(id="t2", kind="b", description="")]
    results = executor.execute(plan)

    assert [r.content for r in results] == ["A 完成", "B 完成"]


def test_executor_returns_error_when_handler_missing():
    executor = PlanExecutor(max_workers=1)
    plan = [SubTask(id="t1", kind="unknown", description="")]

    results = executor.execute(plan)

    assert not results[0].success
    assert "no handler" in (results[0].error or "")


def test_parallel_executor_tasks_share_one_budget_manager():
    manager = BudgetManager(max_tool_calls=2)
    executor = PlanExecutor(max_workers=4, budget_manager=manager)
    admitted = Barrier(2)

    def handler(task):
        admitted.wait(timeout=2)
        return SubTaskResult(
            id=task.id,
            kind=task.kind,
            success=True,
            content=task.id,
        )

    executor.register_handler("work", handler)
    plan = [
        SubTask(id=f"t{index}", kind="work", description="")
        for index in range(6)
    ]

    results = executor.execute(plan)

    assert sum(result.success for result in results) == 2
    assert manager.snapshot()["used_tool_calls"] == 2
    assert all(
        result.success or result.error == "max_tool_calls_exceeded"
        for result in results
    )


def test_aggregator_merges_multiple_successes():
    aggregator = ResultAggregator()
    plan = [
        SubTask(id="t1", kind="weather", description=""),
        SubTask(id="t2", kind="rag_qa", description=""),
    ]
    results = [
        SubTaskResult(id="t1", kind="weather", success=True, content="今天合肥晴天26度"),
        SubTaskResult(id="t2", kind="rag_qa", success=True, content="主刷可以拆下清洗"),
    ]

    answer = aggregator.aggregate("查询天气并问问主刷", plan, results)

    assert "环境与天气" in answer
    assert "知识库参考" in answer
    assert "晴天" in answer and "主刷" in answer


def test_planner_agent_end_to_end_with_mock_handlers():
    planner = TaskPlanner()
    executor = PlanExecutor(max_workers=1)
    executor.register_handler(
        "rag_qa",
        lambda task: SubTaskResult(id=task.id, kind=task.kind, success=True,
                                   content="知识库回答"),
    )
    executor.register_handler(
        "weather",
        lambda task: SubTaskResult(id=task.id, kind=task.kind, success=True,
                                   content="多云"),
    )
    agent = PlannerAgent(planner=planner, executor=executor)

    result = agent.run("今天天气怎么样，扫地机器人怎么保养")

    assert result.answer
    assert any(r.kind == "weather" for r in result.results)
    assert any(r.kind == "rag_qa" for r in result.results)


def test_planner_agent_blocks_invalid_plan_before_execution():
    planner = TaskPlanner(
        llm_planner=lambda query: [
            SubTask(id="t1", kind="unknown", description="bad task"),
        ]
    )
    executor = PlanExecutor(max_workers=1)
    agent = PlannerAgent(
        planner=planner,
        executor=executor,
        validator=PlanValidator(),
        max_steps=8,
    )

    result = agent.run("执行非法计划")

    assert result.results == []
    assert "计划被阻止" in result.answer
    assert "invalid_task_kind" in result.answer


def test_planner_agent_replans_failed_subtask_once():
    planner = TaskPlanner(
        llm_planner=lambda query: [
            SubTask(id="t1", kind="rag_qa", description="will fail", args={"query": query}),
        ]
    )
    executor = PlanExecutor(max_workers=1)
    executor.register_handler(
        "rag_qa",
        lambda task: SubTaskResult(
            id=task.id,
            kind=task.kind,
            success=False,
            content="",
            error="retriever unavailable",
        ),
    )
    executor.register_handler(
        "generic",
        lambda task: SubTaskResult(
            id=task.id,
            kind=task.kind,
            success=True,
            content="fallback answer",
        ),
    )
    agent = PlannerAgent(
        planner=planner,
        executor=executor,
        validator=PlanValidator(),
        replanner=Replanner(),
        max_replans=1,
    )

    result = agent.run("怎么保养滤网")

    assert any(task.id == "fallback-1" for task in result.plan)
    assert any(item.kind == "generic" and item.success for item in result.results)
    assert "fallback answer" in result.answer
