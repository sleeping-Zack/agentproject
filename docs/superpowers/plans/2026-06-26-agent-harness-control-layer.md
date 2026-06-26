# Agent Harness Control Layer Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a real outer harness layer around the existing Agent system.

**Architecture:** The new `AgentRunner` owns unified state, budget, policy, approval, verifier, artifact, and diagnostic trace logic. Existing LangChain Agent, planner, workflow, judge, model router, and persistence modules are reused as execution backends.

**Tech Stack:** Python 3.10, FastAPI, SQLite, pytest, existing LangChain/LangGraph stack.

---

### Task 1: Unified State and Runner

**Files:**
- Create: `agent/state.py`, `agent/runner.py`
- Test: `tests/test_agent_runner.py`, `tests/test_agent_state.py`

- [ ] Define `AgentState`, budgets, observations, tool records, artifact refs, status.
- [ ] Implement budget stop conditions and run status transitions.
- [ ] Return `pending_approval`, `completed`, `blocked`, `failed`, or `rejected`.

### Task 2: Policy and HITL Approval

**Files:**
- Create: `agent/policies.py`, `safety/approval.py`, `services/approval_store.py`
- Modify: `api/server.py`
- Test: `tests/test_policy_engine.py`, `tests/test_approval_flow.py`

- [ ] Add tenant/user/scene decisions: allow, deny, approval, redaction.
- [ ] Store pending approvals and expose approve/deny APIs.
- [ ] Stop sensitive tools instead of passing `confirmed=True` inside the tool.

### Task 3: Verification and Artifacts

**Files:**
- Create: `agent/verifier.py`, `services/artifact_store.py`
- Modify: `api/server.py`, `observability/tracing.py`
- Test: `tests/test_answer_verifier.py`, `tests/test_artifact_store.py`

- [ ] Verify final answers against evidence and judge scores.
- [ ] Refuse or retry when verification fails.
- [ ] Persist final answers, reports, tool results, evidence, and evaluation reports as artifacts.

### Task 4: Model Router and Evaluation Gate

**Files:**
- Modify: `model/factory.py`, `rag/eval_gate.py`, `scripts/evaluate_agent.py`
- Test: `tests/test_model_factory_router.py`, `tests/test_eval_gate.py`

- [ ] Ensure factory-created chat model comes from `ModelRouter`.
- [ ] Add quality gate thresholds and scenario/failure breakdown.
- [ ] Keep offline mock routing available for tests and demos.

### Task 5: Diagnostics and Documentation

**Files:**
- Modify: `observability/tracing.py`, `README.md`, `docs/interview_playbook.md`
- Test: `tests/test_trace_diagnostics.py`, `tests/test_docs.py`

- [ ] Add structured diagnostic event fields.
- [ ] Explain harness design and interview answers.
- [ ] Verify all tests and static checks.
