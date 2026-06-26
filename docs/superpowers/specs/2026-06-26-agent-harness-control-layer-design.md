# Agent Harness Control Layer Design

## Goal

Fill the remaining gap between the current Agent application and a real harness control layer: a unified runner, shared run state, human approval, answer verification, dynamic tool policy, artifact management, richer diagnostic events, model-router integration, and evaluation gates.

## Design

The project already has LangChain ReAct, planner, workflow, judge, model router, metrics, trace, SQLite, and MCP. This phase adds an outer harness layer rather than rewriting LangChain internals. The new `AgentRunner` owns `AgentState`, budgets, policy decisions, approval pauses, verifier checks, artifact persistence, and final run status. LangChain remains one execution backend.

## Components

- `agent/state.py`: unified `AgentState`, `Budget`, `StepRecord`, `ToolCallRecord`, `ArtifactRef`, and status literals.
- `agent/runner.py`: controller that creates state, applies safety/policy, chooses plan/workflow/ReAct path, verifies output, stores artifacts, updates trace, and returns `AgentRunResult`.
- `agent/policies.py`: dynamic tenant/user/scene tool policy with `ALLOW`, `DENY`, `NEED_APPROVAL`, `NEED_REDACTION`.
- `agent/verifier.py`: answer verification wrapper over `rag.judge` plus citation/evidence checks and retry/refuse decision.
- `safety/approval.py` and `services/approval_store.py`: real pending approval state, approve/deny actions, and auditable decision data.
- `services/artifact_store.py`: SQLite-backed artifacts tied to `request_id`, `tenant_id`, type, name, payload, and metadata.
- `rag/eval_gate.py`: scenario-bucketed evaluation gate and failure breakdown for CI-style quality checks.

## API Changes

- `POST /harness/run`: run through `AgentRunner`.
- `POST /approvals/{approval_id}/approve`
- `POST /approvals/{approval_id}/deny`
- `GET /approvals/{approval_id}`
- `GET /artifacts/{request_id}`

## Constraints

The runner controls the outer lifecycle and final verification. It does not reimplement the token-by-token LangChain ReAct loop. That keeps the implementation small while demonstrating the harness responsibilities expected in interviews.

## Testing

Tests cover state creation, budget stop conditions, policy decisions, pending approval lifecycle, verifier pass/refuse decisions, artifact persistence, runner status transitions, diagnostic event shape, model-router usage through factory, and evaluation gate thresholds.
