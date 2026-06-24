# Agent Production Readiness Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add production-readiness features that map directly to ByteDance Agent application development interview expectations.

**Architecture:** Add thin, testable modules around the current app: MCP adapters, workflow orchestration, SQLite persistence, auth/rate-limit middleware, provider abstraction, OpenTelemetry export, stronger safety, benchmark/evaluation scripts, and interview docs. Keep external dependencies minimal and use deterministic tests.

**Tech Stack:** Python 3.10, FastAPI, LangChain/LangGraph, SQLite, pytest, optional MCP-compatible JSON-RPC stdio, Docker, GitHub Actions.

---

### Task 1: MCP Tool Server

**Files:**
- Create: `mcp_server.py`, `mcp_adapter/__init__.py`, `mcp_adapter/server.py`
- Modify: `api/server.py`, `README.md`
- Test: `tests/test_mcp_server.py`

- [x] Add JSON-RPC handlers for `initialize`, `tools/list`, and `tools/call`.
- [x] Expose `rag_summarize`, `get_weather`, and `fetch_external_data`.
- [x] Add HTTP endpoint that reuses the same adapter.

### Task 2: RAG Evaluation

**Files:**
- Modify: `evals/rag_golden.jsonl`, `scripts/evaluate_rag.py`
- Create: `rag/evaluation.py`
- Test: `tests/test_rag_evaluation.py`

- [x] Expand the golden set to at least 50 cases.
- [x] Calculate recall@k, MRR, citation hit rate, hallucination proxy.
- [x] Compare `top_k`, `hybrid`, and `rerank` strategies.

### Task 3: Explicit Workflow

**Files:**
- Create: `agent/workflows/__init__.py`, `agent/workflows/report_workflow.py`
- Modify: `api/server.py`
- Test: `tests/test_report_workflow.py`

- [x] Add report workflow nodes for intent, user info, record fetch, RAG supplement, report generation, fallback.
- [x] Use deterministic direct runner with explicit step boundaries for tests.
- [x] Route report intents through the workflow from API.

### Task 4: Service Controls

**Files:**
- Create: `services/persistence.py`, `services/rate_limit.py`, `services/cache.py`, `services/task_queue.py`
- Modify: `api/server.py`, `.env.example`
- Test: `tests/test_service_controls.py`

- [x] Add API key authentication.
- [x] Add in-memory rate limiting.
- [x] Persist sessions and traces in SQLite.
- [x] Add cache and task queue abstractions.

### Task 5: Model Provider Fallback

**Files:**
- Create: `model/providers.py`
- Modify: `model/factory.py`, `config/rag.yml`, `.env.example`
- Test: `tests/test_model_providers.py`

- [x] Add `MockProvider` and `TongyiProvider`.
- [x] Choose provider by config/env.
- [x] Fall back to mock when requested for offline demo/tests.

### Task 6: Observability and Security

**Files:**
- Modify: `observability/tracing.py`, `safety/security.py`, `agent/tools/registry.py`, `api/server.py`
- Test: `tests/test_observability.py`, `tests/test_security.py`

- [x] Export OpenTelemetry-style spans.
- [x] Detect RAG prompt injection in retrieved content.
- [x] Validate tool arguments and role scopes.
- [x] Require confirmation for sensitive tools.

### Task 7: Benchmarking and Docs

**Files:**
- Create: `scripts/benchmark_api.py`, `docs/demo.md`, `docs/interview_playbook.md`
- Modify: `README.md`
- Test: `tests/test_benchmark.py`, `tests/test_docs.py`

- [x] Add P50/P95/QPS/failure-rate metric calculation.
- [x] Add demo examples and trace sample.
- [x] Add interview explanation covering all three rounds of changes.

### Verification

- [x] Run `.\\.venv\\Scripts\\python.exe -m pytest tests -q`.
- [x] Run `.\\.venv\\Scripts\\python.exe -m ruff check .`.
- [x] Run `.\\.venv\\Scripts\\python.exe -c "import app; import api.server; import mcp_server; print('imports ok')"`.
- [ ] Commit and push to `origin/main`.
