"""Multi-agent planner / executor / aggregator framework.

This is a deliberately small but explicit implementation of the
plan → execute → aggregate pattern used in production Agent systems.

Why this exists in the project:

    * A single ReAct loop is fine for short Q&A but breaks down for
      multi-step tasks ("帮我分析一下这台机器人本月使用情况、找出耗材问题、
      并给一份保养清单"). We need an explicit task graph.

    * Planner decomposes a user request into typed sub-tasks
      (rag_qa / report / weather / generic). Each sub-task is independent.
    * Executor dispatches them to specialised handlers and can run
      independent tasks concurrently via a thread pool.
    * Aggregator merges the per-step results into a final answer.

The whole pipeline is observable via TraceRecorder and reported into
metrics_registry so the existing trace / metrics endpoints work uniformly.
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional

from observability.metrics import metrics_registry
from observability.tracing import trace_recorder


@dataclass
class SubTask:
    id: str
    kind: str
    description: str
    args: Dict = field(default_factory=dict)
    depends_on: List[str] = field(default_factory=list)


@dataclass
class SubTaskResult:
    id: str
    kind: str
    success: bool
    content: str
    error: Optional[str] = None


@dataclass
class PlanRunResult:
    plan: List[SubTask]
    results: List[SubTaskResult]
    answer: str


class TaskPlanner:
    """Decompose a user query into typed sub-tasks.

    The default rule-based planner avoids requiring an LLM call for the
    plan step, which keeps unit tests deterministic. A real deployment can
    inject an LLM-backed planner by passing `llm_planner=...`.
    """

    REPORT_KEYWORDS = ("报告", "使用记录", "月报", "总结")
    WEATHER_KEYWORDS = ("天气", "气温", "湿度", "下雨", "雨水")
    KB_KEYWORDS = ("怎么办", "如何", "为什么", "推荐", "选购", "故障", "保养", "维护",
                   "清洁", "拖布", "电池", "WiFi", "wifi", "充电", "吸力")

    def __init__(self, llm_planner: Optional[Callable[[str], List[SubTask]]] = None) -> None:
        self._llm_planner = llm_planner

    def plan(self, query: str) -> List[SubTask]:
        if self._llm_planner is not None:
            try:
                plan = self._llm_planner(query)
                if plan:
                    return plan
            except Exception:
                pass
        return self._rule_based_plan(query)

    def _rule_based_plan(self, query: str) -> List[SubTask]:
        tasks: List[SubTask] = []
        wants_report = any(kw in query for kw in self.REPORT_KEYWORDS)
        wants_weather = any(kw in query for kw in self.WEATHER_KEYWORDS)
        wants_kb = any(kw in query for kw in self.KB_KEYWORDS)

        if wants_weather:
            tasks.append(SubTask(
                id=f"t{len(tasks)+1}",
                kind="weather",
                description="获取当前用户所在城市的天气",
                args={"query": query},
            ))
        if wants_kb:
            tasks.append(SubTask(
                id=f"t{len(tasks)+1}",
                kind="rag_qa",
                description="检索知识库回答问题",
                args={"query": query},
            ))
        if wants_report:
            tasks.append(SubTask(
                id=f"t{len(tasks)+1}",
                kind="report",
                description="生成本月使用报告",
                args={"query": query},
            ))
        if not tasks:
            tasks.append(SubTask(
                id="t1",
                kind="generic",
                description="走默认 ReAct Agent 回答",
                args={"query": query},
            ))
        return tasks


class PlanExecutor:
    """Run sub-tasks with controlled concurrency.

    Each `kind` maps to a handler callable registered via `register_handler`.
    Tasks whose `depends_on` is empty are eligible to run in parallel.
    """

    def __init__(self, max_workers: int = 4) -> None:
        self.max_workers = max_workers
        self._handlers: Dict[str, Callable[[SubTask], SubTaskResult]] = {}

    def register_handler(self, kind: str, handler: Callable[[SubTask], SubTaskResult]) -> None:
        self._handlers[kind] = handler

    def _run_single(self, task: SubTask) -> SubTaskResult:
        handler = self._handlers.get(task.kind)
        if handler is None:
            return SubTaskResult(
                id=task.id, kind=task.kind, success=False,
                content="", error=f"no handler for kind={task.kind}",
            )
        start = metrics_registry.now()
        try:
            result = handler(task)
            metrics_registry.observe_histogram(
                "agent_subtask_latency_ms",
                metrics_registry.elapsed_ms(start),
                {"kind": task.kind},
            )
            metrics_registry.inc_counter(
                "agent_subtask_total",
                {"kind": task.kind, "status": "success" if result.success else "failure"},
            )
            return result
        except Exception as exc:
            metrics_registry.inc_counter(
                "agent_subtask_total",
                {"kind": task.kind, "status": "failure"},
            )
            return SubTaskResult(
                id=task.id, kind=task.kind, success=False,
                content="", error=str(exc),
            )

    def execute(self, plan: List[SubTask]) -> List[SubTaskResult]:
        results: Dict[str, SubTaskResult] = {}
        pending = list(plan)
        if self.max_workers <= 1 or len(pending) <= 1:
            for task in pending:
                results[task.id] = self._run_single(task)
            return [results[t.id] for t in plan]

        # Independent tasks can run in parallel; tasks with deps run after.
        ready = [t for t in pending if not t.depends_on]
        remaining = [t for t in pending if t.depends_on]
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            futures = {pool.submit(self._run_single, t): t for t in ready}
            for future in as_completed(futures):
                task = futures[future]
                results[task.id] = future.result()
        for task in remaining:
            results[task.id] = self._run_single(task)
        return [results[t.id] for t in plan]

    async def execute_async(self, plan: List[SubTask]) -> List[SubTaskResult]:
        """asyncio 版本：独立子任务用 asyncio.gather 并发，比 ThreadPoolExecutor
        更省内存，且能与 FastAPI 的事件循环统一。"""
        import asyncio
        results: Dict[str, SubTaskResult] = {}
        ready = [t for t in plan if not t.depends_on]
        remaining = [t for t in plan if t.depends_on]
        if ready:
            ready_results = await asyncio.gather(
                *(asyncio.to_thread(self._run_single, t) for t in ready)
            )
            for task, result in zip(ready, ready_results):
                results[task.id] = result
        for task in remaining:
            results[task.id] = await asyncio.to_thread(self._run_single, task)
        return [results[t.id] for t in plan]


class ResultAggregator:
    """Combine sub-task results into a final answer."""

    def aggregate(self, query: str, plan: List[SubTask], results: List[SubTaskResult]) -> str:
        successful = [r for r in results if r.success and r.content.strip()]
        if not successful:
            errors = "; ".join(r.error or "未知错误" for r in results if not r.success)
            return f"很抱歉，未能成功处理你的请求：{errors or '所有子任务都未返回内容'}"
        if len(successful) == 1:
            return successful[0].content
        sections: List[str] = [f"## 综合回答\n针对「{query}」按以下子任务整理："]
        kind_label = {
            "weather": "环境与天气",
            "rag_qa": "知识库参考",
            "report": "使用报告",
            "generic": "通用回答",
        }
        for result in successful:
            sections.append(
                f"\n### {kind_label.get(result.kind, result.kind)}\n{result.content.strip()}"
            )
        return "\n".join(sections)


class PlannerAgent:
    """High-level orchestrator: Planner → Executor → Aggregator with tracing."""

    def __init__(
        self,
        planner: Optional[TaskPlanner] = None,
        executor: Optional[PlanExecutor] = None,
        aggregator: Optional[ResultAggregator] = None,
        validator: Optional[Any] = None,
        replanner: Optional[Any] = None,
        max_steps: int = 8,
        max_replans: int = 1,
    ) -> None:
        self.planner = planner or TaskPlanner()
        self.executor = executor or PlanExecutor()
        self.aggregator = aggregator or ResultAggregator()
        self.validator = validator
        self.replanner = replanner
        self.max_steps = max_steps
        self.max_replans = max_replans

    def run(self, query: str, request_id: Optional[str] = None) -> PlanRunResult:
        if request_id:
            with trace_recorder.span(request_id, category="planner", name="plan"):
                plan = self.planner.plan(query)
                validation_error = self._validation_error(plan)
                if validation_error:
                    return PlanRunResult(plan=plan, results=[], answer=validation_error)
            with trace_recorder.span(
                request_id, category="planner", name="execute",
                metadata={"task_count": len(plan)},
            ):
                results = self.executor.execute(plan)
                plan, results = self._replan_failed(query, plan, results)
            with trace_recorder.span(request_id, category="planner", name="aggregate"):
                answer = self.aggregator.aggregate(query, plan, results)
        else:
            plan = self.planner.plan(query)
            validation_error = self._validation_error(plan)
            if validation_error:
                metrics_registry.inc_counter("agent_planner_runs_total")
                return PlanRunResult(plan=plan, results=[], answer=validation_error)
            results = self.executor.execute(plan)
            plan, results = self._replan_failed(query, plan, results)
            answer = self.aggregator.aggregate(query, plan, results)

        metrics_registry.inc_counter("agent_planner_runs_total")
        return PlanRunResult(plan=plan, results=results, answer=answer)

    def _validation_error(self, plan: List[SubTask]) -> str:
        if self.validator is None:
            return ""
        validation = self.validator.validate(plan, max_steps=self.max_steps)
        if validation.valid:
            return ""
        return "计划被阻止：" + ",".join(validation.errors)

    def _replan_failed(
        self,
        query: str,
        plan: List[SubTask],
        results: List[SubTaskResult],
    ) -> tuple[List[SubTask], List[SubTaskResult]]:
        if self.replanner is None or self.max_replans <= 0:
            return plan, results
        failed = [
            (task, result)
            for task, result in zip(plan, results)
            if not result.success
        ]
        if not failed:
            return plan, results

        fallback_plan: List[SubTask] = []
        for task, result in failed:
            fallback_plan.extend(
                self.replanner.replan(
                    query=query,
                    failed_task=task,
                    failure_reason=result.error or "subtask_failed",
                )
            )
        if not fallback_plan:
            return plan, results
        if self.validator is not None:
            validation = self.validator.validate(
                [*plan, *fallback_plan],
                max_steps=self.max_steps,
            )
            if not validation.valid:
                blocked = SubTaskResult(
                    id="replan-blocked",
                    kind="generic",
                    success=False,
                    content="",
                    error="replan_blocked:" + ",".join(validation.errors),
                )
                return plan, [*results, blocked]
        fallback_results = self.executor.execute(fallback_plan)
        return [*plan, *fallback_plan], [*results, *fallback_results]
