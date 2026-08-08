import asyncio
import json
import logging
import os
import time
from contextlib import asynccontextmanager
from contextvars import ContextVar
from typing import Dict, List, Literal, Optional
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import JSONResponse, PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent.long_term_memory import MemoryCategory, calculate_time_decay
from agent.react_agent import ReactAgent
from agent.policies import ToolPolicy
from agent.runner import (
    AgentRunner,
    AgentTask,
    AutoRoutingBackend,
    PlannerAgentBackend,
    ReactAgentBackend,
)
from agent.tools.agent_tools import (
    fetch_external_data,
    get_weather,
    rag_summarize,
    tool_registry,
)
from mcp_adapter.server import MCPToolServer
from observability.event_bus import AgentEvent, EventStreamConflictError, event_bus
from observability.metrics import metrics_registry
from observability.otel import configure_telemetry
from observability.tracing import otel_spans_from_trace_payload, trace_recorder
from rag.judge import LLMJudge, evaluate_batch
from safety.auth import ADMIN_ROLES, AuthContext, resolve_auth_context
from safety.security import UnsafeInputError, assert_safe_user_input
from services.factories import (
    create_approval_store,
    create_artifact_store,
    create_session_store,
)
from services.rate_limit import create_rate_limiter

LOGGER = logging.getLogger(__name__)


@asynccontextmanager
async def _app_lifespan(_app: FastAPI):
    await start_memory_retention()
    try:
        yield
    finally:
        await stop_memory_retention()


app = FastAPI(
    title="Sweeper Agent API",
    version="0.4.0",
    lifespan=_app_lifespan,
)
_request_auth: ContextVar[Optional[AuthContext]] = ContextVar(
    "request_auth", default=None
)


@app.middleware("http")
async def authenticate_request(request: Request, call_next):
    if request.url.path in {"/health", "/docs", "/openapi.json", "/redoc"}:
        return await call_next(request)
    try:
        context = resolve_auth_context(
            api_key=request.headers.get("X-API-Key", ""),
            authorization=request.headers.get("Authorization"),
            header_tenant_id=request.headers.get("X-Tenant-ID"),
            header_user_role=request.headers.get("X-User-Role"),
            header_principal_id=request.headers.get("X-Principal-ID"),
        )
    except ValueError as exc:
        return JSONResponse(status_code=401, content={"detail": str(exc)})
    token = _request_auth.set(context)
    request.state.auth_context = context
    try:
        return await call_next(request)
    finally:
        _request_auth.reset(token)


configure_telemetry(app)
store = create_session_store()
agent = ReactAgent(session_store=store)
approval_store = create_approval_store()
artifact_store = create_artifact_store()
runtime_tool_policy = ToolPolicy(tool_registry=tool_registry)
harness_runner = AgentRunner(
    backend=AutoRoutingBackend(
        react_backend=ReactAgentBackend(agent=agent),
        planner_backend=PlannerAgentBackend(agent=agent),
        tool_policy=runtime_tool_policy,
    ),
    policy=runtime_tool_policy,
    approval_store=approval_store,
    artifact_store=artifact_store,
    conversation_memory=agent.memory,
)
_retention_task: Optional[asyncio.Task] = None
rate_limiter = create_rate_limiter(
    max_requests=int(os.getenv("AGENT_RATE_LIMIT_REQUESTS", "60")),
    window_seconds=int(os.getenv("AGENT_RATE_LIMIT_WINDOW_SECONDS", "60")),
)
mcp_server = MCPToolServer(
    tool_handlers={
        "rag_summarize": lambda args: rag_summarize.invoke(
            {
                "query": args["query"],
                "information_gap": args.get(
                    "information_gap",
                    "外部调用需要知识库证据",
                ),
            }
        ),
        "get_weather": lambda args: get_weather.invoke({"city": args["city"]}),
        "fetch_external_data": lambda args: fetch_external_data.invoke(
            {"user_id": args["user_id"], "month": args["month"]}
        ),
    },
    policy=harness_runner.policy,
    approval_store=approval_store,
)


async def _retention_loop() -> None:
    interval = max(
        60,
        int(os.getenv("AGENT_MEMORY_RETENTION_INTERVAL_SECONDS", "86400")),
    )
    while True:
        try:
            result = await asyncio.to_thread(agent.long_term_memory.run_retention)
            metrics_registry.inc_counter(
                "agent_memory_retention_runs_total", {"status": "success"}
            )
            for name, count in result.items():
                metrics_registry.inc_counter(
                    "agent_memory_retention_deleted_total",
                    {"kind": str(name)},
                    value=float(count or 0),
                )
        except asyncio.CancelledError:
            raise
        except Exception:
            metrics_registry.inc_counter(
                "agent_memory_retention_runs_total", {"status": "failure"}
            )
            LOGGER.exception("memory retention job failed")
        await asyncio.sleep(interval)


async def start_memory_retention() -> None:
    global _retention_task
    enabled = os.getenv(
        "AGENT_MEMORY_RETENTION_ENABLED", "false"
    ).strip().lower() == "true"
    if enabled and (_retention_task is None or _retention_task.done()):
        _retention_task = asyncio.create_task(_retention_loop())


async def stop_memory_retention() -> None:
    global _retention_task
    if _retention_task is None:
        return
    _retention_task.cancel()
    try:
        await _retention_task
    except asyncio.CancelledError:
        pass
    _retention_task = None


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = "default"
    stream: bool = False
    tenant_id: Optional[str] = None
    request_id: Optional[str] = Field(default=None, min_length=1, max_length=128)
    approval_id: Optional[str] = Field(default=None, min_length=1, max_length=128)


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    answer: str
    trace_url: str
    status: str = "completed"
    approval_id: Optional[str] = None


class PlanRequest(BaseModel):
    message: str = Field(..., min_length=1)
    tenant_id: Optional[str] = None


class PlanResponse(BaseModel):
    request_id: str
    plan: List[Dict]
    results: List[Dict]
    answer: str
    trace_url: str


class HarnessRunRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = "default"
    tenant_id: Optional[str] = None
    user_role: str = "user"
    scene: str = "default"
    approval_id: Optional[str] = None


class HarnessRunResponse(BaseModel):
    request_id: str
    session_id: str
    status: str
    answer: str
    approval_id: Optional[str] = None
    artifacts: List[Dict]
    verifier: Optional[Dict] = None
    trace_url: str


class ApprovalDecisionRequest(BaseModel):
    decided_by: str = "operator"


class MemoryWriteRequest(BaseModel):
    key: str = Field(..., min_length=1, max_length=200)
    value: str = Field(..., min_length=1, max_length=4000)
    category: MemoryCategory
    importance: float = Field(default=0.5, ge=0.0, le=1.0)


class MemoryForgetRequest(BaseModel):
    key: Optional[str] = Field(default=None, min_length=1, max_length=200)


class MemoryReviewRequest(BaseModel):
    decision: Literal["accept", "reject"]


class JudgeCase(BaseModel):
    query: str
    context: str = ""
    answer: str


class JudgeRequest(BaseModel):
    cases: List[JudgeCase]


def _expected_api_key() -> str:
    return os.getenv("AGENT_API_KEY", "dev-api-key")


def _authorize(api_key: Optional[str]) -> None:
    if _request_auth.get() is not None:
        return
    if os.getenv("AGENT_AUTH_MODE", "principal_api_key").strip().lower() == "jwt":
        raise HTTPException(status_code=401, detail="Bearer token is required")
    if api_key != _expected_api_key():
        raise HTTPException(status_code=401, detail="invalid api key")


def _auth_context(
    api_key: Optional[str],
    header_tenant_id: Optional[str] = None,
    header_user_role: Optional[str] = None,
    header_principal_id: Optional[str] = None,
    body_tenant_id: Optional[str] = None,
) -> AuthContext:
    authenticated = _request_auth.get()
    if authenticated is not None:
        supplied = {
            "tenant_id": header_tenant_id or body_tenant_id,
            "user_role": header_user_role,
            "principal_id": header_principal_id,
        }
        for name, value in supplied.items():
            if value is not None and value.strip() and value.strip() != getattr(
                authenticated, name
            ):
                raise HTTPException(
                    status_code=401,
                    detail=f"{name} does not match authenticated identity",
                )
        return authenticated
    try:
        return resolve_auth_context(
            api_key=api_key or "",
            header_tenant_id=header_tenant_id,
            header_user_role=header_user_role,
            header_principal_id=header_principal_id,
            body_tenant_id=body_tenant_id,
        )
    except ValueError as exc:
        raise HTTPException(status_code=401, detail=str(exc)) from exc


def _rate_limit(request: Request, tenant_id: str) -> None:
    """优先按 tenant 维度限流，匿名请求回退到 IP，避免单租户挤占其他租户配额。"""
    if tenant_id and tenant_id != "default":
        key = f"tenant:{tenant_id}"
    else:
        client_host = request.client.host if request.client else "unknown"
        key = f"ip:{client_host}"
    if not rate_limiter.allow(key):
        raise HTTPException(status_code=429, detail="rate limit exceeded")


def _resolve_tenant(body_tenant: Optional[str], header_tenant: Optional[str]) -> str:
    authenticated = _request_auth.get()
    if authenticated is not None:
        supplied = header_tenant or body_tenant
        if supplied and supplied != authenticated.tenant_id:
            raise HTTPException(
                status_code=401,
                detail="tenant_id does not match authenticated identity",
            )
        return authenticated.tenant_id
    return header_tenant or body_tenant or "default"


def _require_approval_operator(context: AuthContext) -> None:
    if context.user_role not in ADMIN_ROLES:
        raise HTTPException(status_code=403, detail="approval requires operator/admin role")


def _require_memory_user_id(principal_id: Optional[str]) -> str:
    authenticated = _request_auth.get()
    if (
        authenticated is not None
        and os.getenv("AGENT_AUTH_MODE", "principal_api_key").strip().lower()
        != "legacy_headers"
    ):
        return authenticated.principal_id
    if principal_id is None or not principal_id.strip():
        raise HTTPException(
            status_code=400,
            detail="X-Principal-ID is required for cross-session memory",
        )
    return principal_id.strip()


def _load_tenant_approval(approval_id: str, tenant_id: str):
    try:
        approval = approval_store.get(approval_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="approval not found") from exc
    if approval.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="approval not found")
    return approval


@app.get("/health")
async def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/auth/me")
async def authenticated_identity(
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_principal_id: Optional[str] = Header(default=None, alias="X-Principal-ID"),
) -> Dict[str, str]:
    auth = _auth_context(x_api_key, x_tenant_id, x_user_role, x_principal_id)
    return {
        "tenant_id": auth.tenant_id,
        "user_id": auth.principal_id,
        "role": auth.user_role,
        "data_user_id": auth.data_user_id or "",
    }


@app.get("/memory")
async def list_memory(
    include_inactive: bool = False,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_principal_id: Optional[str] = Header(default=None, alias="X-Principal-ID"),
) -> List[Dict]:
    auth = _auth_context(x_api_key, x_tenant_id, x_user_role, x_principal_id)
    user_id = _require_memory_user_id(x_principal_id)
    memories = agent.long_term_memory.list_memories(
        auth.tenant_id, user_id, include_inactive=include_inactive
    )
    return [_memory_payload(memory) for memory in memories]


@app.post("/memory")
async def remember(
    request: MemoryWriteRequest,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_principal_id: Optional[str] = Header(default=None, alias="X-Principal-ID"),
) -> Dict:
    auth = _auth_context(x_api_key, x_tenant_id, x_user_role, x_principal_id)
    user_id = _require_memory_user_id(x_principal_id)
    try:
        memory = agent.long_term_memory.remember(
            auth.tenant_id,
            user_id,
            request.key,
            request.value,
            request.category,
            importance=request.importance,
            explicit=True,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _memory_payload(memory)


@app.delete("/memory")
async def forget_memory(
    request: Optional[MemoryForgetRequest] = None,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_principal_id: Optional[str] = Header(default=None, alias="X-Principal-ID"),
) -> Dict[str, int]:
    auth = _auth_context(x_api_key, x_tenant_id, x_user_role, x_principal_id)
    user_id = _require_memory_user_id(x_principal_id)
    deleted = agent.long_term_memory.forget(
        auth.tenant_id, user_id, key=request.key if request else None
    )
    return {"deleted": deleted}


@app.post("/memory/{memory_id}/review")
async def review_pending_memory(
    memory_id: str,
    request: MemoryReviewRequest,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_principal_id: Optional[str] = Header(default=None, alias="X-Principal-ID"),
) -> Dict:
    auth = _auth_context(x_api_key, x_tenant_id, x_user_role, x_principal_id)
    user_id = _require_memory_user_id(x_principal_id)
    try:
        memory = agent.long_term_memory.review_pending(
            auth.tenant_id,
            user_id,
            memory_id,
            request.decision,
        )
    except KeyError as exc:
        raise HTTPException(
            status_code=404, detail="pending memory not found"
        ) from exc
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    return _memory_payload(memory)


def _memory_payload(memory) -> Dict:
    return {
        "memory_id": memory.memory_id,
        "key": memory.key,
        "value": memory.value,
        "category": memory.category.value,
        "status": memory.status,
        "version": memory.version,
        "confidence": memory.confidence,
        "importance": memory.importance,
        "explicit": memory.explicit,
        "metadata": memory.metadata,
        "source_event_id": memory.source_event_id,
        "supersedes_id": memory.supersedes_id,
        "last_confirmed_at": memory.last_confirmed_at.isoformat(),
        "recency_score": calculate_time_decay(memory.category, memory.last_confirmed_at),
    }


def _serialise_datetime(value):
    return value.isoformat() if hasattr(value, "isoformat") else value


def _limit(value: int) -> int:
    if value < 1 or value > 200:
        raise HTTPException(status_code=400, detail="limit must be between 1 and 200")
    return value


@app.get("/sessions")
async def list_sessions(
    limit: int = 100,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_principal_id: Optional[str] = Header(default=None, alias="X-Principal-ID"),
) -> List[Dict]:
    auth = _auth_context(x_api_key, x_tenant_id, x_user_role, x_principal_id)
    user_id = _require_memory_user_id(x_principal_id)
    loader = getattr(store, "list_sessions", None)
    if not callable(loader):
        raise HTTPException(status_code=503, detail="session history backend is unavailable")
    sessions = loader(auth.tenant_id, user_id, _limit(limit))
    return [
        {**item, "created_at": _serialise_datetime(item.get("created_at")),
         "updated_at": _serialise_datetime(item.get("updated_at"))}
        for item in sessions
    ]


@app.get("/sessions/{session_id}/messages")
async def get_session_messages(
    session_id: str,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_principal_id: Optional[str] = Header(default=None, alias="X-Principal-ID"),
) -> Dict:
    auth = _auth_context(x_api_key, x_tenant_id, x_user_role, x_principal_id)
    user_id = _require_memory_user_id(x_principal_id)
    loader = getattr(store, "get_session_messages", None)
    if not callable(loader):
        raise HTTPException(status_code=503, detail="session history backend is unavailable")
    messages = loader(session_id, tenant_id=auth.tenant_id, user_id=user_id)
    if not messages:
        raise HTTPException(status_code=404, detail="session not found")
    return {"session_id": session_id, "messages": messages}


@app.get("/memory/events")
async def list_memory_events(
    limit: int = 100,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_principal_id: Optional[str] = Header(default=None, alias="X-Principal-ID"),
) -> List[Dict]:
    auth = _auth_context(x_api_key, x_tenant_id, x_user_role, x_principal_id)
    user_id = _require_memory_user_id(x_principal_id)
    events = agent.long_term_memory.store.list_event_details(
        auth.tenant_id, user_id, _limit(limit)
    )
    return [
        {**event, "created_at": _serialise_datetime(event.get("created_at"))}
        for event in events
    ]


@app.get("/memory/summaries")
async def list_memory_summaries(
    limit: int = 100,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_principal_id: Optional[str] = Header(default=None, alias="X-Principal-ID"),
) -> List[Dict]:
    auth = _auth_context(x_api_key, x_tenant_id, x_user_role, x_principal_id)
    user_id = _require_memory_user_id(x_principal_id)
    summaries = agent.long_term_memory.store.list_summaries(
        auth.tenant_id, user_id, _limit(limit)
    )
    return [
        {**summary, "updated_at": _serialise_datetime(summary.get("updated_at"))}
        for summary in summaries
    ]


@app.get("/memory/procedures")
async def list_memory_procedures(
    status: str = "approved",
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_principal_id: Optional[str] = Header(default=None, alias="X-Principal-ID"),
) -> List[Dict]:
    auth = _auth_context(x_api_key, x_tenant_id, x_user_role, x_principal_id)
    _require_memory_user_id(x_principal_id)
    if status not in {"approved", "candidate"}:
        raise HTTPException(status_code=400, detail="unsupported procedure status")
    if status != "approved" and auth.user_role not in ADMIN_ROLES:
        raise HTTPException(status_code=403, detail="candidate procedures require operator/admin role")
    procedures = agent.long_term_memory.list_procedures(auth.tenant_id, status=status)
    return [
        {
            "procedure_id": item.procedure_id, "tenant_id": item.tenant_id,
            "agent_version": item.agent_version, "status": item.status,
            "title": item.title, "content": item.content, "evidence": item.evidence,
            "created_at": item.created_at.isoformat(),
            "approved_at": item.approved_at.isoformat() if item.approved_at else None,
        }
        for item in procedures
    ]


@app.get("/tools/manifest")
async def tool_manifest() -> Dict:
    return tool_registry.as_mcp_manifest()


@app.get("/metrics", response_class=PlainTextResponse)
async def metrics() -> str:
    return metrics_registry.render_prometheus()


@app.get("/metrics/snapshot")
async def metrics_snapshot() -> Dict:
    return metrics_registry.snapshot()


@app.post("/chat", response_model=ChatResponse)
async def chat(
    request: ChatRequest,
    raw_request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_principal_id: Optional[str] = Header(default=None, alias="X-Principal-ID"),
) -> ChatResponse:
    auth = _auth_context(x_api_key, x_tenant_id, x_user_role, x_principal_id, request.tenant_id)
    tenant_id = auth.tenant_id
    _rate_limit(raw_request, tenant_id)
    try:
        assert_safe_user_input(request.message)
    except UnsafeInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    task = AgentTask(
        query=request.message,
        session_id=request.session_id,
        tenant_id=tenant_id,
        user_id=auth.principal_id,
        data_user_id=auth.data_user_id,
        user_role=auth.user_role,
        scene="default",
        approval_id=request.approval_id,
    )
    result = await asyncio.to_thread(harness_runner.run, task)
    answer = result.answer

    trace_payload = trace_recorder.export_trace(result.request_id)
    store.save_trace(result.request_id, request.session_id, trace_payload, tenant_id=tenant_id)
    return ChatResponse(
        request_id=result.request_id,
        session_id=request.session_id,
        answer=answer,
        trace_url=f"/traces/{result.request_id}",
        status=result.state.status,
        approval_id=result.approval_id,
    )


_TERMINAL_STREAM_EVENTS = frozenset({"run_completed", "run_failed"})


def _parse_last_event_id(value: Optional[str]) -> int:
    if value is None or not value.strip():
        return 0
    try:
        sequence = int(value)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail="Last-Event-ID must be an integer") from exc
    if sequence < 0:
        raise HTTPException(status_code=400, detail="Last-Event-ID must be non-negative")
    return sequence


def _format_sse_event(event: AgentEvent) -> str:
    payload = dict(event.payload)
    payload["request_id"] = event.request_id
    payload["timestamp"] = event.timestamp
    if event.event_type in _TERMINAL_STREAM_EVENTS:
        payload.setdefault("trace_url", f"/traces/{event.request_id}")
    encoded = json.dumps(payload, ensure_ascii=False, separators=(",", ":"))
    return f"id: {event.sequence}\nevent: {event.event_type}\ndata: {encoded}\n\n"


def _save_stream_trace(request_id: str, session_id: str, tenant_id: str) -> None:
    try:
        trace_payload = trace_recorder.export_trace(request_id)
    except KeyError:
        return
    store.save_trace(request_id, session_id, trace_payload, tenant_id=tenant_id)


async def _legacy_runner_stream(task: AgentTask, last_event_id: int):
    """Compatibility for injected runners that predate the streaming contract."""
    result = await asyncio.to_thread(harness_runner.run, task)
    sequence = last_event_id
    if result.answer:
        sequence += 1
        yield AgentEvent(
            request_id=result.request_id,
            event_type="token_delta",
            sequence=sequence,
            timestamp=time.time(),
            payload={"delta": result.answer, "provisional": False},
        )
    sequence += 1
    terminal_type = (
        "run_completed"
        if result.state.status in {"completed", "pending_approval"}
        else "run_failed"
    )
    yield AgentEvent(
        request_id=result.request_id,
        event_type=terminal_type,
        sequence=sequence,
        timestamp=time.time(),
        payload={
            "status": result.state.status,
            "answer": result.answer,
            "approval_id": result.approval_id,
        },
    )


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    raw_request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_principal_id: Optional[str] = Header(default=None, alias="X-Principal-ID"),
    last_event_id: Optional[str] = Header(default=None, alias="Last-Event-ID"),
):
    """Stream sequenced harness events and replay missed events on reconnect."""
    auth = _auth_context(x_api_key, x_tenant_id, x_user_role, x_principal_id, request.tenant_id)
    tenant_id = auth.tenant_id
    _rate_limit(raw_request, tenant_id)
    try:
        assert_safe_user_input(request.message)
    except UnsafeInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    sequence_cursor = _parse_last_event_id(last_event_id)
    request_id = request.request_id or str(uuid4())
    task = AgentTask(
        query=request.message,
        session_id=request.session_id,
        tenant_id=tenant_id,
        user_id=auth.principal_id,
        data_user_id=auth.data_user_id,
        user_role=auth.user_role,
        scene="default",
        approval_id=request.approval_id,
        request_id=request_id,
        emit_events=True,
    )
    expected_identity = AgentRunner.stream_identity(task)
    existing_identity = event_bus.identity(request_id)
    if existing_identity is not None and existing_identity != expected_identity:
        raise HTTPException(status_code=409, detail="request_id is bound to another stream")

    async def event_stream():
        started_at = time.perf_counter()
        ttft_recorded = False
        disconnected = False
        stream_method = getattr(harness_runner, "run_stream", None)
        source = (
            stream_method(task, last_event_id=sequence_cursor)
            if callable(stream_method)
            else _legacy_runner_stream(task, sequence_cursor)
        )
        try:
            async for event in source:
                if await raw_request.is_disconnected():
                    disconnected = True
                    break
                if event.event_type == "token_delta" and not ttft_recorded:
                    metrics_registry.observe_ttft((time.perf_counter() - started_at) * 1000)
                    ttft_recorded = True
                if event.event_type in _TERMINAL_STREAM_EVENTS:
                    _save_stream_trace(request_id, request.session_id, tenant_id)
                yield _format_sse_event(event)
        except EventStreamConflictError:
            conflict = AgentEvent(
                request_id=request_id,
                event_type="run_failed",
                sequence=sequence_cursor + 1,
                timestamp=time.time(),
                payload={"status": "failed", "error": "stream_identity_conflict"},
            )
            yield _format_sse_event(conflict)
        except asyncio.CancelledError:
            event_bus.cancel(request_id)
            raise
        finally:
            if disconnected:
                event_bus.cancel(request_id)

    headers = {
        "Cache-Control": "no-cache",
        "X-Accel-Buffering": "no",
        "X-Request-ID": request_id,
    }
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


@app.post("/plan", response_model=PlanResponse)
async def plan_endpoint(
    request: PlanRequest,
    raw_request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_principal_id: Optional[str] = Header(default=None, alias="X-Principal-ID"),
) -> PlanResponse:
    auth = _auth_context(x_api_key, x_tenant_id, x_user_role, x_principal_id, request.tenant_id)
    tenant_id = auth.tenant_id
    _rate_limit(raw_request, tenant_id)
    request_id = str(uuid4())
    try:
        assert_safe_user_input(request.message)
    except UnsafeInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    plan_result = await asyncio.to_thread(
        agent.run_plan, request.message, request_id, tenant_id
    )
    try:
        trace_payload = trace_recorder.export_trace(request_id)
    except KeyError:
        trace_recorder.start_trace(request_id=request_id, session_id="planner")
        trace_payload = trace_recorder.export_trace(request_id)
    store.save_trace(request_id, "planner", trace_payload, tenant_id=tenant_id)
    return PlanResponse(
        request_id=request_id,
        plan=[{"id": t.id, "kind": t.kind, "description": t.description} for t in plan_result.plan],
        results=[
            {"id": r.id, "kind": r.kind, "success": r.success,
             "content": r.content, "error": r.error}
            for r in plan_result.results
        ],
        answer=plan_result.answer,
        trace_url=f"/traces/{request_id}",
    )


@app.post("/harness/run", response_model=HarnessRunResponse)
async def harness_run(
    request: HarnessRunRequest,
    raw_request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_principal_id: Optional[str] = Header(default=None, alias="X-Principal-ID"),
) -> HarnessRunResponse:
    auth = _auth_context(x_api_key, x_tenant_id, x_user_role, x_principal_id, request.tenant_id)
    tenant_id = auth.tenant_id
    _rate_limit(raw_request, tenant_id)
    try:
        assert_safe_user_input(request.message)
    except UnsafeInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    task = AgentTask(
        query=request.message,
        session_id=request.session_id,
        tenant_id=tenant_id,
        user_id=auth.principal_id,
        data_user_id=auth.data_user_id,
        user_role=auth.user_role,
        scene=request.scene,
        approval_id=request.approval_id,
    )
    result = await asyncio.to_thread(harness_runner.run, task)
    trace_payload = trace_recorder.export_trace(result.request_id)
    store.save_trace(result.request_id, request.session_id, trace_payload, tenant_id=tenant_id)
    return HarnessRunResponse(
        request_id=result.request_id,
        session_id=request.session_id,
        status=result.state.status,
        answer=result.answer,
        approval_id=result.approval_id,
        artifacts=[artifact.__dict__ for artifact in result.artifacts],
        verifier=result.verifier.__dict__ if result.verifier else None,
        trace_url=f"/traces/{result.request_id}",
    )


@app.get("/approvals")
async def list_approvals(
    raw_request: Request,
    status: Optional[Literal["pending", "approved", "denied"]] = None,
    limit: int = 100,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_principal_id: Optional[str] = Header(default=None, alias="X-Principal-ID"),
) -> List[Dict]:
    auth = _auth_context(x_api_key, x_tenant_id, x_user_role, x_principal_id)
    _require_approval_operator(auth)
    _rate_limit(raw_request, auth.tenant_id)
    return [
        record.__dict__
        for record in approval_store.list_approvals(
            auth.tenant_id,
            status=status,
            limit=max(1, min(limit, 200)),
        )
    ]


@app.get("/approvals/{approval_id}")
async def get_approval(
    approval_id: str,
    raw_request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_principal_id: Optional[str] = Header(default=None, alias="X-Principal-ID"),
) -> Dict:
    auth = _auth_context(x_api_key, x_tenant_id, x_user_role, x_principal_id)
    tenant_id = auth.tenant_id
    _rate_limit(raw_request, tenant_id)
    approval = _load_tenant_approval(approval_id, tenant_id)
    if (
        not auth.can_approve
        and approval.principal_id
        and approval.principal_id != auth.principal_id
    ):
        raise HTTPException(status_code=404, detail="approval not found")
    return approval.__dict__


@app.post("/approvals/{approval_id}/approve")
async def approve_approval(
    approval_id: str,
    request: ApprovalDecisionRequest,
    raw_request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_principal_id: Optional[str] = Header(default=None, alias="X-Principal-ID"),
) -> Dict:
    auth = _auth_context(x_api_key, x_tenant_id, x_user_role, x_principal_id)
    _require_approval_operator(auth)
    tenant_id = auth.tenant_id
    _rate_limit(raw_request, tenant_id)
    _load_tenant_approval(approval_id, tenant_id)
    return approval_store.approve(approval_id, auth.principal_id).__dict__


@app.post("/approvals/{approval_id}/deny")
async def deny_approval(
    approval_id: str,
    request: ApprovalDecisionRequest,
    raw_request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_principal_id: Optional[str] = Header(default=None, alias="X-Principal-ID"),
) -> Dict:
    auth = _auth_context(x_api_key, x_tenant_id, x_user_role, x_principal_id)
    _require_approval_operator(auth)
    tenant_id = auth.tenant_id
    _rate_limit(raw_request, tenant_id)
    _load_tenant_approval(approval_id, tenant_id)
    return approval_store.deny(approval_id, auth.principal_id).__dict__


@app.get("/artifacts/{request_id}")
async def list_artifacts(
    request_id: str,
    raw_request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
) -> Dict:
    _authorize(x_api_key)
    tenant_id = _resolve_tenant(None, x_tenant_id)
    _rate_limit(raw_request, tenant_id)
    artifacts = artifact_store.list_artifacts(request_id, tenant_id=tenant_id)
    return {"artifacts": [artifact.__dict__ for artifact in artifacts]}


@app.get("/artifact/{artifact_id}")
async def get_artifact(
    artifact_id: str,
    raw_request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
) -> Dict:
    _authorize(x_api_key)
    tenant_id = _resolve_tenant(None, x_tenant_id)
    _rate_limit(raw_request, tenant_id)
    try:
        artifact = artifact_store.get_artifact(artifact_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="artifact not found") from exc
    if artifact.tenant_id != tenant_id:
        raise HTTPException(status_code=404, detail="artifact not found")
    return {"artifact": artifact.__dict__}


@app.post("/judge")
async def judge_endpoint(
    request: JudgeRequest,
    raw_request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
) -> Dict:
    _authorize(x_api_key)
    tenant_id = _resolve_tenant(None, x_tenant_id)
    _rate_limit(raw_request, tenant_id)
    result = await asyncio.to_thread(
        evaluate_batch,
        [case.model_dump() for case in request.cases],
        LLMJudge(),
    )
    return {"cases": result.cases, "aggregate": result.aggregate}


@app.get("/traces/{request_id}")
async def get_trace(request_id: str) -> Dict:
    auth = _request_auth.get()
    if auth is None:
        raise HTTPException(status_code=401, detail="authentication required")
    try:
        return store.get_trace(request_id, tenant_id=auth.tenant_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="trace not found") from exc


@app.get("/traces/{request_id}/otel")
async def get_otel_trace(request_id: str) -> Dict:
    auth = _request_auth.get()
    if auth is None:
        raise HTTPException(status_code=401, detail="authentication required")
    try:
        trace_payload = store.get_trace(request_id, tenant_id=auth.tenant_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="trace not found") from exc
    return {"spans": otel_spans_from_trace_payload(trace_payload)}


@app.post("/mcp")
async def mcp_http(
    request: Dict,
    raw_request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_principal_id: Optional[str] = Header(default=None, alias="X-Principal-ID"),
) -> Dict:
    auth = _auth_context(x_api_key, x_tenant_id, x_user_role, x_principal_id)
    _rate_limit(raw_request, auth.tenant_id)
    return mcp_server.handle_jsonrpc(request, context=auth)
