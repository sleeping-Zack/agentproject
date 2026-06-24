# Agent Engineering Upgrade Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Upgrade the sweeping-robot Agent demo into a reproducible, testable, service-ready Agent application project.

**Architecture:** Keep the Streamlit UI, add a FastAPI service surface, and isolate core production concerns into small modules: deterministic service adapters, tool registry, safety, memory, tracing, and RAG utilities. Existing LangChain code is modified only where needed to consume those modules.

**Tech Stack:** Python, Streamlit, FastAPI, LangChain/LangGraph, Chroma, pytest, Docker, GitHub Actions.

---

### Task 1: Reproducible Project Foundation

**Files:**
- Create: `.gitignore`, `.env.example`, `README.md`, `pyproject.toml`, `Dockerfile`, `.dockerignore`, `.github/workflows/ci.yml`
- Modify: `config/agent.yml`, `config/chroma.yml`

- [x] Add dependency and runtime metadata.
- [x] Document setup, environment variables, startup commands, tests, evaluation, and deployment.
- [x] Ignore editor files, caches, logs, local vector stores, and generated MD5 manifests.

### Task 2: Deterministic Tools and Tool Registry

**Files:**
- Create: `services/tool_data_service.py`, `agent/tools/registry.py`
- Modify: `agent/tools/agent_tools.py`, `agent/tools/middleware.py`
- Test: `tests/test_tool_data_service.py`, `tests/test_tool_registry.py`, `tests/test_report_flow.py`

- [x] Replace random user/location/month data with config-backed deterministic data.
- [x] Parse usage records with `csv.DictReader`.
- [x] Add tool allowlist and MCP-style manifest.
- [x] Wire allowlist checks and trace spans into runtime middleware.

### Task 3: Safety and Observability

**Files:**
- Create: `safety/security.py`, `observability/tracing.py`
- Modify: `app.py`, `agent/react_agent.py`, `agent/tools/middleware.py`, `utils/logger_handler.py`
- Test: `tests/test_security.py`, `tests/test_observability.py`

- [x] Detect obvious prompt-injection requests.
- [x] Redact sensitive values in trace metadata.
- [x] Apply user-input safety at Streamlit and API entry points.
- [x] Export traces through the API.

### Task 4: RAG Metadata, Citations, and Evaluation

**Files:**
- Create: `rag/rag_utils.py`, `evals/rag_golden.jsonl`, `scripts/evaluate_rag.py`
- Modify: `rag/vector_store.py`, `rag/rag_service.py`, `config/chroma.yml`
- Test: `tests/test_rag_utils.py`

- [x] Add metadata, citation, and hybrid ranking helpers.
- [x] Store chunk version/source metadata when loading documents.
- [x] Return citations with RAG answers.
- [x] Add small golden-set evaluation script.

### Task 5: Session Memory, API, and Reliability

**Files:**
- Create: `agent/memory.py`, `api/server.py`, `utils/streaming.py`
- Modify: `agent/react_agent.py`, `app.py`
- Test: `tests/test_memory.py`, `tests/test_streaming.py`, `tests/test_prompt_regression.py`

- [x] Add bounded in-process memory.
- [x] Fix empty stream fallback helper.
- [x] Pass session history into the Agent.
- [x] Add FastAPI health/chat/tool-manifest/trace endpoints.
- [x] Use stream fallback in Streamlit.

### Verification

- [x] `python -m pytest tests -q` passes for current unit modules.
- [x] Import check for lightweight modules passes in the Python 3.10 virtual environment.
- [x] Document runtime dependency requirements for LangChain/Streamlit/FastAPI.
