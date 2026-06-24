import asyncio
import json
import os
import threading
import time
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request
from fastapi.responses import PlainTextResponse, StreamingResponse
from pydantic import BaseModel, Field

from agent.react_agent import ReactAgent
from agent.tools.agent_tools import (
    fetch_external_data,
    get_weather,
    rag,
    rag_summarize,
    tool_data_service,
    tool_registry,
)
from agent.workflows.report_workflow import ReportWorkflow
from mcp_adapter.server import MCPToolServer
from observability.event_bus import event_bus
from observability.metrics import metrics_registry
from observability.tracing import trace_recorder
from rag.judge import LLMJudge, evaluate_batch
from safety.security import UnsafeInputError, assert_safe_user_input
from services.persistence import SQLiteStore
from services.rate_limit import RateLimiter
from utils.streaming import get_final_response

app = FastAPI(title="Sweeper Agent API", version="0.4.0")
store = SQLiteStore(os.getenv("AGENT_DB_PATH", "storage/agent.db"))
agent = ReactAgent(session_store=store)
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
    }
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


class PlanRequest(BaseModel):
    message: str = Field(..., min_length=1)
    tenant_id: Optional[str] = None


class PlanResponse(BaseModel):
    request_id: str
    plan: List[Dict]
    results: List[Dict]
    answer: str
    trace_url: str


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
    return body_tenant or header_tenant or "default"


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
) -> ChatResponse:
    _authorize(x_api_key)
    tenant_id = _resolve_tenant(request.tenant_id, x_tenant_id)
    _rate_limit(raw_request, tenant_id)
    request_id = str(uuid4())
    try:
        assert_safe_user_input(request.message)
    except UnsafeInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def _run() -> str:
        if "报告" in request.message or "使用记录" in request.message:
            trace_recorder.start_trace(request_id=request_id, session_id=request.session_id)
            workflow = ReportWorkflow(tool_service=tool_data_service, rag_service=rag)
            return workflow.run(request.message)["answer"]
        chunks: List[str] = list(
            agent.execute_stream(
                request.message,
                session_id=request.session_id,
                request_id=request_id,
                tenant_id=tenant_id,
            )
        )
        return get_final_response(chunks)

    # Agent / 模型调用是 CPU+IO 阻塞型，扔进默认 threadpool，避免阻塞事件循环
    answer = await asyncio.to_thread(_run)

    store.save_session_message(request.session_id, "user", request.message,
                                tenant_id=tenant_id)
    store.save_session_message(request.session_id, "assistant", answer,
                                tenant_id=tenant_id)
    trace_payload = trace_recorder.export_trace(request_id)
    store.save_trace(request_id, request.session_id, trace_payload, tenant_id=tenant_id)
    return ChatResponse(
        request_id=request_id,
        session_id=request.session_id,
        answer=answer,
        trace_url=f"/traces/{request_id}",
    )


@app.post("/chat/stream")
async def chat_stream(
    request: ChatRequest,
    raw_request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
    x_tenant_id: Optional[str] = Header(default=None, alias="X-Tenant-ID"),
):
    """Server-Sent Events stream of incremental Agent output."""
    _authorize(x_api_key)
    tenant_id = _resolve_tenant(request.tenant_id, x_tenant_id)
    _rate_limit(raw_request, tenant_id)
    request_id = str(uuid4())
    try:
        assert_safe_user_input(request.message)
    except UnsafeInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def event_stream():
        emitted: List[str] = []
        previous = ""
        ttft_recorded = False
        start_time = time.perf_counter()
        last_emit = start_time
        HEARTBEAT_INTERVAL = float(os.getenv("AGENT_SSE_HEARTBEAT_SECONDS", "15"))

        # 在后台线程跑 Agent，主协程同时排空 message chunk 与 event_bus 中的 tool 事件
        chunk_queue: "asyncio.Queue[Optional[str]]" = None  # noqa: E501  仅用于类型暗示
        import queue as _queue
        chunk_q: _queue.Queue = _queue.Queue()

        def producer():
            try:
                for chunk in agent.execute_stream(
                    request.message,
                    session_id=request.session_id,
                    request_id=request_id,
                    tenant_id=tenant_id,
                ):
                    chunk_q.put(("chunk", chunk))
            except Exception as exc:
                chunk_q.put(("error", str(exc)))
            finally:
                chunk_q.put(("done", None))
                event_bus.close(request_id)

        producer_thread = threading.Thread(target=producer, daemon=True)
        producer_thread.start()

        done = False
        while not done:
            now = time.perf_counter()
            if now - last_emit >= HEARTBEAT_INTERVAL:
                yield "event: heartbeat\ndata: {}\n\n"
                last_emit = now

            event = event_bus.consume(request_id, timeout=0.2)
            if event and event != "closed":
                payload = json.dumps({"request_id": request_id, **event.data},
                                     ensure_ascii=False)
                yield f"event: {event.event}\ndata: {payload}\n\n"
                last_emit = time.perf_counter()
                continue

            try:
                kind, payload_raw = chunk_q.get_nowait()
            except _queue.Empty:
                continue

            if kind == "error":
                err = json.dumps({"request_id": request_id, "error": payload_raw},
                                 ensure_ascii=False)
                yield f"event: error\ndata: {err}\n\n"
                done = True
                break
            if kind == "done":
                done = True
                break

            chunk = payload_raw
            emitted.append(chunk)
            stripped = chunk.rstrip("\n")
            delta = stripped[len(previous):] if stripped.startswith(previous) else stripped
            previous = stripped
            if not delta:
                continue

            if not ttft_recorded:
                metrics_registry.observe_ttft((time.perf_counter() - start_time) * 1000)
                ttft_recorded = True

            payload = json.dumps(
                {"request_id": request_id, "delta": delta, "full": stripped},
                ensure_ascii=False,
            )
            yield f"event: answer\ndata: {payload}\n\n"
            last_emit = time.perf_counter()

        # 排空 event_bus 中可能晚到的事件
        while True:
            event = event_bus.consume(request_id, timeout=0.05)
            if not event or event == "closed":
                break
            payload = json.dumps({"request_id": request_id, **event.data},
                                 ensure_ascii=False)
            yield f"event: {event.event}\ndata: {payload}\n\n"

        final = get_final_response(emitted)
        store.save_session_message(request.session_id, "user", request.message,
                                    tenant_id=tenant_id)
        store.save_session_message(request.session_id, "assistant", final,
                                    tenant_id=tenant_id)
        trace_payload = trace_recorder.export_trace(request_id)
        store.save_trace(request_id, request.session_id, trace_payload,
                         tenant_id=tenant_id)
        done_payload = json.dumps(
            {"request_id": request_id, "answer": final,
             "trace_url": f"/traces/{request_id}"},
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
) -> PlanResponse:
    _authorize(x_api_key)
    tenant_id = _resolve_tenant(request.tenant_id, x_tenant_id)
    _rate_limit(raw_request, tenant_id)
    request_id = str(uuid4())
    try:
        assert_safe_user_input(request.message)
    except UnsafeInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    plan_result = await asyncio.to_thread(
        agent.run_plan, request.message, request_id, tenant_id
    )
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
async def mcp_http(request: Dict) -> Dict:
    return mcp_server.handle_jsonrpc(request)
