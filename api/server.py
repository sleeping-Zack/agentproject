import asyncio
import json
import os
import time
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent.react_agent import ReactAgent
from agent.runner import AgentRunner, AgentTask, ReactAgentBackend
from agent.tools.agent_tools import (
    fetch_external_data,
    get_weather,
    rag_summarize,
    tool_registry,
)
from mcp_adapter.server import MCPToolServer
from observability.metrics import metrics_registry
from observability.tracing import trace_recorder
from rag.judge import LLMJudge, evaluate_batch
from safety.auth import ADMIN_ROLES, AuthContext, resolve_auth_context
from safety.security import UnsafeInputError, assert_safe_user_input
from services.approval_store import SQLiteApprovalStore
from services.artifact_store import SQLiteArtifactStore
from services.persistence import SQLiteStore
from services.rate_limit import RateLimiter

app = FastAPI(title="Sweeper Agent API", version="0.4.0")
store = SQLiteStore(os.getenv("AGENT_DB_PATH", "storage/agent.db"))
agent = ReactAgent(session_store=store)
approval_store = SQLiteApprovalStore(os.getenv("AGENT_APPROVAL_DB_PATH", "storage/approvals.db"))
artifact_store = SQLiteArtifactStore(os.getenv("AGENT_ARTIFACT_DB_PATH", "storage/artifacts.db"))
harness_runner = AgentRunner(
    backend=ReactAgentBackend(agent=agent),
    approval_store=approval_store,
    artifact_store=artifact_store,
    conversation_memory=agent.memory,
)
rate_limiter = RateLimiter(
    max_requests=int(os.getenv("AGENT_RATE_LIMIT_REQUESTS", "60")),
    window_seconds=int(os.getenv("AGENT_RATE_LIMIT_WINDOW_SECONDS", "60")),
)
mcp_server = MCPToolServer(
    tool_handlers={
        "rag_summarize": lambda args: rag_summarize.invoke({"query": args["query"]}),
        "get_weather": lambda args: get_weather.invoke({"city": args["city"]}),
        "fetch_external_data": lambda args: fetch_external_data.invoke(
            {"user_id": args["user_id"], "month": args["month"]}
        ),
    },
    policy=harness_runner.policy,
    approval_store=approval_store,
)


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = "default"
    stream: bool = False
    tenant_id: Optional[str] = None


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


class JudgeCase(BaseModel):
    query: str
    context: str = ""
    answer: str


class JudgeRequest(BaseModel):
    cases: List[JudgeCase]


def _expected_api_key() -> str:
    return os.getenv("AGENT_API_KEY", "dev-api-key")


def _authorize(api_key: Optional[str]) -> None:
    if api_key != _expected_api_key():
        raise HTTPException(status_code=401, detail="invalid api key")


def _auth_context(
    api_key: Optional[str],
    header_tenant_id: Optional[str] = None,
    header_user_role: Optional[str] = None,
    header_principal_id: Optional[str] = None,
    body_tenant_id: Optional[str] = None,
) -> AuthContext:
    _authorize(api_key)
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
    return header_tenant or body_tenant or "default"


def _require_approval_operator(context: AuthContext) -> None:
    if context.user_role not in ADMIN_ROLES:
        raise HTTPException(status_code=403, detail="approval requires operator/admin role")


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
        user_role=auth.user_role,
        scene="default",
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


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    raw_request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
    x_user_role: Optional[str] = Header(default=None, alias="X-User-Role"),
    x_principal_id: Optional[str] = Header(default=None, alias="X-Principal-ID"),
):
    """Server-Sent Events facade over the harness-controlled execution path."""
    auth = _auth_context(x_api_key, x_tenant_id, x_user_role, x_principal_id, request.tenant_id)
    tenant_id = auth.tenant_id
    _rate_limit(raw_request, tenant_id)
    try:
        assert_safe_user_input(request.message)
    except UnsafeInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def event_stream():
        start_time = time.perf_counter()
        task = AgentTask(
            query=request.message,
            session_id=request.session_id,
            tenant_id=tenant_id,
            user_role=auth.user_role,
            scene="default",
        )
        result = harness_runner.run(task)
        metrics_registry.observe_ttft((time.perf_counter() - start_time) * 1000)
        final = result.answer
        answer_payload = json.dumps(
            {
                "request_id": result.request_id,
                "delta": final,
                "full": final,
                "status": result.state.status,
                "approval_id": result.approval_id,
            },
            ensure_ascii=False,
        )
        yield f"event: answer\ndata: {answer_payload}\n\n"
        trace_payload = trace_recorder.export_trace(result.request_id)
        store.save_trace(result.request_id, request.session_id, trace_payload,
                         tenant_id=tenant_id)
        done_payload = json.dumps(
            {
                "request_id": result.request_id,
                "answer": final,
                "status": result.state.status,
                "approval_id": result.approval_id,
                "trace_url": f"/traces/{result.request_id}",
            },
            ensure_ascii=False,
        )
        yield f"event: done\ndata: {done_payload}\n\n"

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
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
    return _load_tenant_approval(approval_id, tenant_id).__dict__


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
    try:
        return store.get_trace(request_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="trace not found") from exc


@app.get("/traces/{request_id}/otel")
async def get_otel_trace(request_id: str) -> Dict:
    try:
        return {"spans": trace_recorder.export_otel_spans(request_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="trace not found") from exc


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
