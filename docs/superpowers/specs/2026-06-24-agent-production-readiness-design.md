# Agent Production Readiness Design

## Goal

Upgrade the project from a reproducible Agent application into a stronger ByteDance-style portfolio project with real MCP exposure, explicit workflow orchestration, larger RAG evaluation, service-side controls, model fallback, OpenTelemetry-friendly observability, security hardening, benchmarking, and interview-ready documentation.

## Scope

This phase implements ten additions:

1. MCP stdio and HTTP adapters for selected tools.
2. A larger RAG golden set and metrics: recall@k, MRR, citation hit rate, hallucination proxy, and strategy comparison.
3. Explicit report workflow orchestration with LangGraph when available and a deterministic fallback runner for tests.
4. Demo documentation with curl examples, expected responses, and trace output.
5. Server-side API key authentication, in-memory rate limiting, SQLite persistence for sessions/traces, and cache/task abstractions.
6. Model provider abstraction with Tongyi and Mock providers.
7. OpenTelemetry-friendly span export from the existing trace recorder.
8. Stronger security: RAG injection detection, tool argument validation, role scopes, and sensitive-tool confirmation.
9. API benchmark script with P50/P95/QPS/failure-rate output.
10. README and interview playbook documentation that explain all three rounds of changes.

## Design Choices

The project remains local-first. SQLite is used instead of a full database service, a simple in-memory limiter is used instead of Redis, and the MCP implementation includes both a real JSON-RPC stdio surface and an HTTP endpoint compatible with the project API. This keeps the code reviewable while demonstrating the production design points expected in interviews.

## Data Flow

User/API/MCP input passes through safety validation, role/scope checks, and optional confirmation checks. Agent requests create traces, use configured model providers, can run through the explicit report workflow for report intents, and persist session/trace data through SQLite. RAG evaluation loads golden cases and compares strategy metrics without requiring a live model call.

## Testing

Tests cover MCP JSON-RPC behavior, RAG metric calculation, workflow steps, API auth/rate limit/persistence surfaces, model fallback, OpenTelemetry span export, security validation, benchmark metric calculation, and documentation existence.
