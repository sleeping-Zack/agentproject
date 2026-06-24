# Agent Engineering Upgrade Design

## Goal

Turn the current sweeping-robot customer-service Agent demo into a reproducible engineering portfolio project aligned with Agent application development roles.

## Scope

The upgrade covers ten concrete areas: reproducible setup, tests and evaluation, deterministic tool data, MCP-style tool registry, stronger RAG metadata and citations, conversation memory, security controls, observability, service/deployment surfaces, and reliability fixes.

## Architecture

The Streamlit app remains as a demo UI. A FastAPI app exposes production-style chat, health, tool manifest, and trace endpoints. The LangChain Agent uses deterministic service adapters behind its tools, a tool registry for allowlist enforcement and MCP-style metadata, conversation memory for session state, safety guards for prompt-injection checks, and a trace recorder for request/tool/RAG spans.

## Components

- `services/tool_data_service.py`: deterministic service adapter for user context, weather, current month, and usage records.
- `agent/tools/registry.py`: tool metadata, allowlist enforcement, and MCP-style manifest export.
- `agent/memory.py`: bounded in-process conversation history, user profile, and last tool result storage.
- `safety/security.py`: prompt-injection checks and sensitive-value redaction.
- `observability/tracing.py`: request trace and span recorder.
- `rag/rag_utils.py`: source metadata, hybrid ranking, and citation formatting helpers.
- `api/server.py`: FastAPI service endpoints.
- `evals/` and `scripts/`: golden-set based RAG evaluation entry point.

## Testing

Unit tests cover tool data determinism, CSV parsing, MCP registry behavior, security, trace export, RAG utilities, conversation memory, stream fallback, report flow, and prompt regression. Evaluation data provides a small golden set for RAG recall checks.

## Constraints

The implementation stays small and local. It does not introduce distributed tracing, a real identity system, or a production database. Those are documented as extension points rather than overbuilt into this portfolio project.
