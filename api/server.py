from typing import Dict, List
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from agent.react_agent import ReactAgent
from agent.tools.agent_tools import tool_registry
from observability.tracing import trace_recorder
from safety.security import UnsafeInputError, assert_safe_user_input
from utils.streaming import get_final_response


app = FastAPI(title="Sweeper Agent API", version="0.2.0")
agent = ReactAgent()


class ChatRequest(BaseModel):
    message: str = Field(..., min_length=1)
    session_id: str = "default"
    stream: bool = False


class ChatResponse(BaseModel):
    request_id: str
    session_id: str
    answer: str
    trace_url: str


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


@app.get("/tools/manifest")
def tool_manifest() -> Dict:
    return tool_registry.as_mcp_manifest()


@app.post("/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    request_id = str(uuid4())
    try:
        assert_safe_user_input(request.message)
    except UnsafeInputError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    chunks: List[str] = list(
        agent.execute_stream(
            request.message,
            session_id=request.session_id,
            request_id=request_id,
        )
    )
    return ChatResponse(
        request_id=request_id,
        session_id=request.session_id,
        answer=get_final_response(chunks),
        trace_url=f"/traces/{request_id}",
    )


@app.get("/traces/{request_id}")
def get_trace(request_id: str) -> Dict:
    try:
        return trace_recorder.export_trace(request_id)
    except KeyError as exc:
        raise HTTPException(status_code=404, detail="trace not found") from exc
