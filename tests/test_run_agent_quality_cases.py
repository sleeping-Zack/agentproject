from __future__ import annotations

from dataclasses import dataclass, field
from types import SimpleNamespace

import pytest

from observability.tracing import trace_recorder
from scripts.run_agent_quality_cases import _usage, run_case, run_cases, select_cases


@dataclass
class FakeState:
    status: str = "completed"
    error: str | None = None
    scene: str = "security_boundary"
    tool_calls: list = field(default_factory=list)
    observations: list = field(default_factory=list)
    plan: list = field(default_factory=list)
    approval_id: str | None = None


@dataclass
class FakeToolCall:
    tool_name: str
    args: dict
    status: str
    result: str = ""
    error: str | None = None
    approval_id: str | None = None
    risk_level: str = "low"


class FakeRunner:
    def __init__(self, *, failure: Exception | None = None):
        self.failure = failure
        self.tasks = []

    def run(self, task):
        self.tasks.append(task)
        if self.failure:
            raise self.failure
        trace_recorder.start_trace(task.request_id, task.session_id)
        trace_recorder.record_diagnostic_event(
            request_id=task.request_id,
            step_id="model-1",
            event_type="model_usage",
            status="ok",
            latency_ms=0,
            tokens_in=11,
            tokens_out=7,
            cost=0.018,
            cost_mode="actual",
            model_name="fake-model",
        )
        trace_recorder.record_diagnostic_event(
            request_id=task.request_id,
            step_id="step-1",
            event_type="verifier",
            status="ok",
            latency_ms=0,
            tokens_in=11,
            tokens_out=7,
            cost=0.018,
            cost_mode="actual",
            model_name="fake-model",
        )
        state = FakeState(
            tool_calls=[
                FakeToolCall(
                    tool_name="rag_summarize",
                    args={"query": "q"},
                    status="success",
                    result="ok",
                    error=None,
                    approval_id=None,
                    risk_level="low",
                )
            ],
            observations=[SimpleNamespace(metadata={"id": "evidence-1"})],
            plan=[
                {
                    "id": "t1",
                    "kind": "rag_qa",
                    "description": "检索知识库",
                    "arguments": {"query": "q"},
                }
            ],
        )
        return SimpleNamespace(
            answer="answer",
            state=state,
            artifacts=[],
            verifier=None,
        )


def _case(case_id="case-1", split="dev", **extra):
    return {
        "case_id": case_id,
        "query": "question",
        "split": split,
        **extra,
    }


def test_select_cases_filters_validates_and_limits():
    dataset = {
        "dev-b": _case("dev-b"),
        "test-a": _case("test-a", "test"),
        "dev-a": _case("dev-a"),
    }

    assert [case["case_id"] for case in select_cases(
        dataset, split="dev", case_ids=[], limit=1
    )] == ["dev-a"]
    assert [case["case_id"] for case in select_cases(
        dataset, split="dev", case_ids=["dev-b"], limit=None
    )] == ["dev-b"]
    with pytest.raises(ValueError, match="unknown case IDs"):
        select_cases(dataset, split="dev", case_ids=["missing"], limit=None)
    with pytest.raises(ValueError, match="do not belong"):
        select_cases(dataset, split="dev", case_ids=["test-a"], limit=None)
    with pytest.raises(ValueError, match="limit must be positive"):
        select_cases(dataset, split="dev", case_ids=[], limit=0)


def test_select_cases_fails_closed_or_filters_unsupported_runtime_features():
    dataset = {
        "dev-fault": _case(
            "dev-fault", context={"fault_injection": "timeout"}
        ),
        "dev-plain": _case("dev-plain"),
        "dev-turns": _case("dev-turns", turns=[{"role": "user", "content": "hi"}]),
    }

    with pytest.raises(ValueError, match="require runtime fixtures"):
        select_cases(dataset, split="dev", case_ids=[], limit=None)
    assert [
        case["case_id"]
        for case in select_cases(
            dataset,
            split="dev",
            case_ids=[],
            limit=None,
            runnable_only=True,
        )
    ] == ["dev-plain"]
    with pytest.raises(ValueError, match="explicitly requested"):
        select_cases(
            dataset,
            split="dev",
            case_ids=["dev-fault"],
            limit=None,
            runnable_only=True,
        )


def test_run_case_extracts_contract_and_honors_dataset_identity_context():
    runner = FakeRunner()
    record = run_case(
        runner,
        _case(
            context={
                "tenant_id": "tenant-b",
                "principal_id": "admin-1",
                "data_user_id": "1008",
                "user_role": "admin",
            },
            scene="security_boundary",
            expected={
                "tools": [
                    {"name": "rag_summarize"},
                    {"name": "get_weather"},
                ]
            },
        ),
        variant="candidate",
        model_snapshot="snapshot-1",
        tenant_id="fallback-tenant",
        user_id="fallback-user",
    )

    task = runner.tasks[0]
    assert (task.tenant_id, task.user_id, task.data_user_id, task.user_role) == (
        "tenant-b",
        "admin-1",
        "1008",
        "admin",
    )
    assert task.required_tools == ("rag_summarize", "get_weather")
    assert record["case_id"] == "case-1"
    assert record["agent_answer"] == "answer"
    assert record["status"] == "completed"
    assert record["tool_calls"][0]["tool_name"] == "rag_summarize"
    assert record["planner_steps"][0]["id"] == "t1"
    assert record["approval_records"] == []
    assert record["evidence"] == [{"id": "evidence-1"}]
    assert record["model_metadata"] == {
        "model_snapshot": "snapshot-1",
        "observed_model": "fake-model",
        "variant": "candidate",
    }
    assert record["policy_context"]["principal_id"] == "admin-1"
    assert record["estimated_cost"] == 0.018
    assert record["cost_mode"] == "actual"
    assert (record["tokens_in"], record["tokens_out"]) == (11, 7)
    assert isinstance(record["latency_ms"], float)


def test_usage_does_not_double_count_runner_verifier_diagnostic():
    events = [
        {
            "category": "diagnostic",
            "metadata": {
                "type": event_type,
                "tokens_in": 10,
                "tokens_out": 5,
                "cost": 0.01,
                "cost_mode": "actual",
                "model_name": "model-a",
            },
        }
        for event_type in ("model_usage", "verifier")
    ]

    assert _usage(events) == (10, 5, 0.01, "actual", "model-a")
    assert _usage([]) == (None, None, None, "not_available", None)


def test_run_case_writes_failure_record_and_run_cases_resumes():
    failed = run_case(
        FakeRunner(failure=RuntimeError("offline")),
        _case("case-failed", scene="failure_recovery"),
        variant="candidate",
        model_snapshot="snapshot-1",
        tenant_id="tenant-a",
        user_id="user-a",
    )
    assert failed["status"] == "failed"
    assert failed["agent_answer"] == ""
    assert failed["estimated_cost"] is None
    assert failed["tokens_in"] is None
    assert failed["error"] == "RuntimeError: offline"

    written = []
    generated = run_cases(
        [_case("already"), _case("new")],
        runner=FakeRunner(),
        variant="candidate",
        model_snapshot="snapshot-1",
        tenant_id="tenant-a",
        user_id="user-a",
        existing_case_ids={"already"},
        on_result=written.append,
    )
    assert [record["case_id"] for record in generated] == ["new"]
    assert written == generated
