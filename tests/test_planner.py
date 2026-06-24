from agent.planner import (
    PlanExecutor,
    PlannerAgent,
    ResultAggregator,
    SubTask,
    SubTaskResult,
    TaskPlanner,
)


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
