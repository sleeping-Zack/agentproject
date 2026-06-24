import os
from typing import Dict, List, Optional
from uuid import uuid4

from fastapi import FastAPI, Header, HTTPException, Request
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
from observability.tracing import trace_recorder
from safety.security import UnsafeInputError, assert_safe_user_input
from services.persistence import SQLiteStore
from services.rate_limit import RateLimiter
from utils.streaming import get_final_response


app = FastAPI(title="Sweeper Agent API", version="0.3.0")
agent = ReactAgent()
store = SQLiteStore(os.getenv("AGENT_DB_PATH", "storage/agent.db"))
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
