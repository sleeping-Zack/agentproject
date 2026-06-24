import json
import os
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


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    answer: str
    trace_url: str


class PlanRequest(BaseModel):
    message: str = Field(..., min_length=1)


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


def _rate_limit(request: Request) -> None:
    client_host = request.client.host if request.client else "unknown"
    if not rate_limiter.allow(client_host):
        raise HTTPException(status_code=429, detail="rate limit exceeded")


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/tools/manifest")
def tool_manifest() -> Dict:
    return tool_registry.as_mcp_manifest()


@app.get("/metrics", response_class=PlainTextResponse)
def metrics() -> str:
    return metrics_registry.render_prometheus()


@app.get("/metrics/snapshot")
def metrics_snapshot() -> Dict:
    return metrics_registry.snapshot()


@app.post("/chat", response_model=ChatResponse)
def chat(
    request: ChatRequest,
    raw_request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> ChatResponse:
    _authorize(x_api_key)
    _rate_limit(raw_request)
    request_id = str(uuid4())
    try:
        assert_safe_user_input(request.message)
    except UnsafeInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    if "报告" in request.message or "使用记录" in request.message:
        trace_recorder.start_trace(request_id=request_id, session_id=request.session_id)
        workflow = ReportWorkflow(tool_service=tool_data_service, rag_service=rag)
        result = workflow.run(request.message)
        answer = result["answer"]
    else:
        chunks: List[str] = list(
            agent.execute_stream(
                request.message,
                session_id=request.session_id,
                request_id=request_id,
            )
        )
        answer = get_final_response(chunks)

    store.save_session_message(request.session_id, "user", request.message)
    store.save_session_message(request.session_id, "assistant", answer)
    trace_payload = trace_recorder.export_trace(request_id)
    store.save_trace(request_id, request.session_id, trace_payload)
    return ChatResponse(
        request_id=request_id,
        session_id=request.session_id,
        answer=answer,
        trace_url=f"/traces/{request_id}",
    )


@app.post("/chat/stream")
def chat_stream(
    request: ChatRequest,
    raw_request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
):
    """Server-Sent Events stream of incremental Agent output.

    Emits one `data: {...}\\n\\n` line per chunk plus a final `event: done`
    sentinel. This is the format expected by EventSource on the client side.
    """
    _authorize(x_api_key)
    _rate_limit(raw_request)
    request_id = str(uuid4())
    try:
        assert_safe_user_input(request.message)
    except UnsafeInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    def event_stream():
        emitted: List[str] = []
        previous = ""
        try:
            for chunk in agent.execute_stream(
                request.message,
                session_id=request.session_id,
                request_id=request_id,
            ):
                emitted.append(chunk)
                stripped = chunk.rstrip("\n")
                delta = stripped[len(previous):] if stripped.startswith(previous) else stripped
                previous = stripped
                if not delta:
                    continue
                payload = json.dumps(
                    {"request_id": request_id, "delta": delta, "full": stripped},
                    ensure_ascii=False,
                )
                yield f"data: {payload}\n\n"
        except Exception as exc:
            payload = json.dumps({"request_id": request_id, "error": str(exc)},
                                 ensure_ascii=False)
            yield f"event: error\ndata: {payload}\n\n"
            return
        final = get_final_response(emitted)
        store.save_session_message(request.session_id, "user", request.message)
        store.save_session_message(request.session_id, "assistant", final)
        trace_payload = trace_recorder.export_trace(request_id)
        store.save_trace(request_id, request.session_id, trace_payload)
        done = json.dumps(
            {"request_id": request_id, "answer": final, "trace_url": f"/traces/{request_id}"},
            ensure_ascii=False,
        )
        yield f"event: done\ndata: {done}\n\n"

    headers = {"Cache-Control": "no-cache", "X-Accel-Buffering": "no"}
    return StreamingResponse(event_stream(), media_type="text/event-stream", headers=headers)


@app.post("/plan", response_model=PlanResponse)
def plan_endpoint(
    request: PlanRequest,
    raw_request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> PlanResponse:
    _authorize(x_api_key)
    _rate_limit(raw_request)
    request_id = str(uuid4())
    try:
        assert_safe_user_input(request.message)
    except UnsafeInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    plan_result = agent.run_plan(request.message, request_id=request_id)
    trace_payload = trace_recorder.export_trace(request_id)
    store.save_trace(request_id, "planner", trace_payload)
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
def judge_endpoint(
    request: JudgeRequest,
    raw_request: Request,
    x_api_key: Optional[str] = Header(default=None, alias="X-API-Key"),
) -> Dict:
    _authorize(x_api_key)
    _rate_limit(raw_request)
    result = evaluate_batch([case.model_dump() for case in request.cases], judge=LLMJudge())
    return {"cases": result.cases, "aggregate": result.aggregate}


@app.get("/traces/{request_id}")
def get_trace(request_id: str) -> Dict:
    try:
        return store.get_trace(request_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="trace not found") from exc


@app.get("/traces/{request_id}/otel")
def get_otel_trace(request_id: str) -> Dict:
    try:
        return {"spans": trace_recorder.export_otel_spans(request_id)}
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="trace not found") from exc


@app.post("/mcp")
def mcp_http(request: Dict) -> Dict:
    return mcp_server.handle_jsonrpc(request)
