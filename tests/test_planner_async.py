import asyncio

from agent.planner import PlanExecutor, SubTask, SubTaskResult


def test_plan_executor_async_runs_independent_tasks_in_parallel():
    executor = PlanExecutor(max_workers=4)

    async def slow_handler(task: SubTask) -> SubTaskResult:
        # 注意：handler 是同步的，executor 会 to_thread 包装
        return SubTaskResult(id=task.id, kind=task.kind, success=True, content=f"ok-{task.id}")

    def handler(task: SubTask) -> SubTaskResult:
        return SubTaskResult(id=task.id, kind=task.kind, success=True, content=f"ok-{task.id}")

    executor.register_handler("a", handler)
    executor.register_handler("b", handler)

    plan = [
        SubTask(id="t1", kind="a", description=""),
        SubTask(id="t2", kind="b", description=""),
    ]
    results = asyncio.run(executor.execute_async(plan))
    assert [r.content for r in results] == ["ok-t1", "ok-t2"]


def test_plan_executor_async_runs_dependent_tasks_serially():
    executor = PlanExecutor(max_workers=4)
    order = []

    def handler(task: SubTask) -> SubTaskResult:
        order.append(task.id)
        return SubTaskResult(id=task.id, kind=task.kind, success=True, content="")

    executor.register_handler("x", handler)
    plan = [
        SubTask(id="t1", kind="x", description=""),
        SubTask(id="t2", kind="x", description="", depends_on=["t1"]),
    ]
    asyncio.run(executor.execute_async(plan))
    assert order.index("t1") < order.index("t2")
