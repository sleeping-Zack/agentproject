from __future__ import annotations

import json
from typing import Any, Dict, Iterable, Iterator, Optional

import requests


class AgentApiError(RuntimeError):
    pass


def parse_sse_lines(lines: Iterable[str]) -> Iterator[Dict[str, Any]]:
    event_id = ""
    event_type = "message"
    data_lines = []
    for raw_line in lines:
        line = raw_line.rstrip("\r\n")
        if not line:
            if data_lines:
                encoded = "\n".join(data_lines)
                try:
                    payload = json.loads(encoded)
                except json.JSONDecodeError:
                    payload = {"raw": encoded}
                yield {"id": event_id, "event": event_type, "data": payload}
            event_id = ""
            event_type = "message"
            data_lines = []
            continue
        if line.startswith(":"):
            continue
        field, _, value = line.partition(":")
        value = value[1:] if value.startswith(" ") else value
        if field == "id":
            event_id = value
        elif field == "event":
            event_type = value
        elif field == "data":
            data_lines.append(value)
    if data_lines:
        encoded = "\n".join(data_lines)
        try:
            payload = json.loads(encoded)
        except json.JSONDecodeError:
            payload = {"raw": encoded}
        yield {"id": event_id, "event": event_type, "data": payload}


class AgentApiClient:
    def __init__(
        self,
        base_url: str,
        api_key: str,
        tenant_id: str,
        user_id: str,
        *,
        user_role: str = "user",
        bearer_token: str = "",
        timeout: float = 60.0,
        session=None,
    ) -> None:
        self.base_url = base_url.rstrip("/")
        self.api_key = api_key
        self.tenant_id = tenant_id
        self.user_id = user_id
        self.user_role = user_role
        self.bearer_token = bearer_token
        self.timeout = timeout
        self.session = session or requests.Session()

    @property
    def headers(self) -> Dict[str, str]:
        headers = {"X-API-Key": self.api_key} if self.api_key else {}
        if self.bearer_token:
            headers["Authorization"] = f"Bearer {self.bearer_token}"
        return headers

    def health(self) -> Dict[str, Any]:
        return self._json("GET", "/health")

    def identity(self) -> Dict[str, Any]:
        return self._json("GET", "/auth/me")

    def tool_manifest(self) -> Dict[str, Any]:
        return self._json("GET", "/tools/manifest")

    def metrics_snapshot(self) -> Dict[str, Any]:
        return self._json("GET", "/metrics/snapshot")

    def list_memories(self, include_inactive: bool = False):
        return self._json(
            "GET", "/memory", params={"include_inactive": str(include_inactive).lower()}
        )

    def list_sessions(self, limit: int = 100):
        return self._json("GET", "/sessions", params={"limit": limit})

    def session_messages(self, session_id: str) -> Dict[str, Any]:
        return self._json("GET", f"/sessions/{session_id}/messages")

    def memory_events(self, limit: int = 100):
        return self._json("GET", "/memory/events", params={"limit": limit})

    def memory_summaries(self, limit: int = 100):
        return self._json("GET", "/memory/summaries", params={"limit": limit})

    def memory_procedures(self, status: str = "approved"):
        return self._json("GET", "/memory/procedures", params={"status": status})

    def remember(
        self,
        key: str,
        value: str,
        category: str,
        importance: float = 0.5,
    ) -> Dict[str, Any]:
        return self._json(
            "POST",
            "/memory",
            json={
                "key": key,
                "value": value,
                "category": category,
                "importance": importance,
            },
        )

    def forget(self, key: Optional[str] = None) -> Dict[str, Any]:
        return self._json("DELETE", "/memory", json={"key": key} if key else None)

    def review_memory(
        self, memory_id: str, decision: str
    ) -> Dict[str, Any]:
        if decision not in {"accept", "reject"}:
            raise ValueError("decision must be accept or reject")
        return self._json(
            "POST",
            f"/memory/{memory_id}/review",
            json={"decision": decision},
        )

    def plan(self, message: str) -> Dict[str, Any]:
        return self._json("POST", "/plan", json={"message": message})

    def approval(self, approval_id: str) -> Dict[str, Any]:
        return self._json("GET", f"/approvals/{approval_id}")

    def list_approvals(self, status: str = "pending", limit: int = 100):
        return self._json(
            "GET",
            "/approvals",
            params={"status": status, "limit": limit},
        )

    def decide_approval(
        self, approval_id: str, decision: str, decided_by: str
    ) -> Dict[str, Any]:
        if decision not in {"approve", "deny"}:
            raise ValueError("decision must be approve or deny")
        return self._json(
            "POST",
            f"/approvals/{approval_id}/{decision}",
            json={"decided_by": decided_by},
        )

    def artifacts(self, request_id: str):
        return self._json("GET", f"/artifacts/{request_id}")

    def artifact(self, artifact_id: str) -> Dict[str, Any]:
        return self._json("GET", f"/artifact/{artifact_id}")

    def trace(self, request_id: str, otel: bool = False) -> Dict[str, Any]:
        suffix = "/otel" if otel else ""
        return self._json("GET", f"/traces/{request_id}{suffix}")

    def chat_events(
        self,
        message: str,
        session_id: str,
        *,
        request_id: Optional[str] = None,
        approval_id: Optional[str] = None,
        last_event_id: Optional[int] = None,
    ) -> Iterator[Dict[str, Any]]:
        headers = dict(self.headers)
        if last_event_id is not None:
            headers["Last-Event-ID"] = str(last_event_id)
        response = self._request(
            "POST",
            "/chat/stream",
            headers=headers,
            json={
                "message": message,
                "session_id": session_id,
                **({"request_id": request_id} if request_id else {}),
                **({"approval_id": approval_id} if approval_id else {}),
            },
            stream=True,
        )
        response.encoding = "utf-8"
        try:
            yield from parse_sse_lines(response.iter_lines(decode_unicode=True))
        finally:
            response.close()

    def _json(self, method: str, path: str, **kwargs):
        response = self._request(method, path, **kwargs)
        try:
            return response.json()
        except ValueError as exc:
            raise AgentApiError(f"backend returned invalid JSON for {path}") from exc

    def _request(self, method: str, path: str, **kwargs):
        headers = kwargs.pop("headers", self.headers)
        try:
            response = self.session.request(
                method,
                f"{self.base_url}{path}",
                headers=headers,
                timeout=self.timeout,
                **kwargs,
            )
        except requests.RequestException as exc:
            raise AgentApiError(f"cannot reach backend at {self.base_url}") from exc
        if response.status_code >= 400:
            try:
                detail = response.json().get("detail")
            except (ValueError, AttributeError):
                detail = None
            raise AgentApiError(str(detail or response.text or f"HTTP {response.status_code}"))
        return response
