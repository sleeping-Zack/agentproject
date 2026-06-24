"""请求级上下文：通过 ContextVar 在调用链中传递 request_id / session_id / tenant / model。

任何一处只要 from observability.context import request_context 即可读取
当前请求的上下文，结构化日志 formatter 也基于这些字段生成 JSON 一行一事件，
让 trace 与日志通过 request_id 自然对齐。
"""
from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar, Token
from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class RequestContext:
    request_id: Optional[str] = None
    session_id: Optional[str] = None
    tenant_id: Optional[str] = None
    model: Optional[str] = None
    prompt_version: Optional[str] = None
    extra: Dict[str, str] = field(default_factory=dict)

    def as_dict(self) -> Dict[str, str]:
        data = {
            "request_id": self.request_id,
            "session_id": self.session_id,
            "tenant_id": self.tenant_id,
            "model": self.model,
            "prompt_version": self.prompt_version,
        }
        data = {k: v for k, v in data.items() if v}
        data.update(self.extra)
        return data


_current: ContextVar[RequestContext] = ContextVar("request_context", default=RequestContext())


def request_context() -> RequestContext:
    return _current.get()


def set_request_context(ctx: RequestContext) -> Token:
    return _current.set(ctx)


def reset_request_context(token: Token) -> None:
    _current.reset(token)


def update_request_context(**fields) -> None:
    """就地修改当前 ContextVar 的字段（无栈语义），常用于 prompt_version 这种延后才知道的字段。"""
    current = _current.get()
    merged = RequestContext(
        request_id=fields.get("request_id", current.request_id),
        session_id=fields.get("session_id", current.session_id),
        tenant_id=fields.get("tenant_id", current.tenant_id),
        model=fields.get("model", current.model),
        prompt_version=fields.get("prompt_version", current.prompt_version),
        extra={**current.extra, **{k: v for k, v in fields.items()
                                    if k not in {"request_id", "session_id", "tenant_id",
                                                 "model", "prompt_version"} and v is not None}},
    )
    _current.set(merged)


@contextmanager
def bind_request_context(**fields):
    current = _current.get()
    merged = RequestContext(
        request_id=fields.get("request_id", current.request_id),
        session_id=fields.get("session_id", current.session_id),
        tenant_id=fields.get("tenant_id", current.tenant_id),
        model=fields.get("model", current.model),
        prompt_version=fields.get("prompt_version", current.prompt_version),
        extra={**current.extra, **{k: v for k, v in fields.items()
                                    if k not in {"request_id", "session_id", "tenant_id",
                                                 "model", "prompt_version"} and v is not None}},
    )
    token = _current.set(merged)
    try:
        yield merged
    finally:
        _current.reset(token)
