from agent.state import AgentState, ArtifactRef, Budget, ToolCallRecord


def test_budget_stops_when_step_limit_is_reached():
    budget = Budget(max_steps=1)
    assert budget.can_continue()

    budget.record_step()

    assert not budget.can_continue()
    assert budget.stop_reason() == "max_steps_exceeded"


def test_agent_state_records_tool_calls_and_artifacts():
    state = AgentState(
        request_id="req-1",
        session_id="session-1",
        tenant_id="tenant-a",
        user_goal="生成清扫报告",
    )

    state.record_step(step_type="plan", name="detect_intent", status="ok")
    state.add_tool_call(
        ToolCallRecord(
            tool_name="fetch_external_data",
            args={"user_id": "u-1", "month": "2026-06"},
            status="pending_approval",
            approval_id="appr-1",
            risk_level="medium",
        )
    )
    state.add_artifact(ArtifactRef(artifact_id="art-1", type="answer", name="final"))
    state.mark_pending_approval("appr-1")

    assert state.status == "pending_approval"
    assert state.approval_id == "appr-1"
    assert state.steps[0].step_id == "step-1"
    assert state.tool_calls[0].tool_name == "fetch_external_data"
    assert state.artifacts[0].artifact_id == "art-1"
