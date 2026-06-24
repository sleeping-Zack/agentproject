import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from typing import Dict, List, Optional

from safety.security import redact_sensitive


@dataclass
class TraceEvent:
    category: str
    name: str
    started_at: float
    duration_ms: float = 0
    metadata: Dict = field(default_factory=dict)
    error: Optional[str] = None

    def export(self) -> Dict:
        return {
            "category": self.category,
            "name": self.name,
            "started_at": self.started_at,
            "duration_ms": self.duration_ms,
            "metadata": redact_sensitive(self.metadata),
            "error": self.error,
        }


@dataclass
class Trace:
    request_id: str
    session_id: str
    started_at: float
    events: List[TraceEvent] = field(default_factory=list)


class TraceRecorder:
    def __init__(self) -> None:
        self._traces: Dict[str, Trace] = {}

    def start_trace(self, request_id: str, session_id: str) -> Trace:
        trace = Trace(request_id=request_id, session_id=session_id, started_at=time.time())
        self._traces[request_id] = trace
        return trace

    @contextmanager
    def span(self, request_id: str, category: str, name: str, metadata: Optional[Dict] = None):
        trace = self._traces[request_id]
        event = TraceEvent(category=category, name=name, started_at=time.time(), metadata=metadata or {})
        start = time.perf_counter()
        try:
            yield event
        except Exception as exc:
            event.error = str(exc)
            raise
        finally:
            event.duration_ms = round((time.perf_counter() - start) * 1000, 3)
            trace.events.append(event)

    def export_trace(self, request_id: str) -> Dict:
        trace = self._traces[request_id]
        return {
            "request_id": trace.request_id,
            "session_id": trace.session_id,
            "started_at": trace.started_at,
            "events": [event.export() for event in trace.events],
        }

    def export_otel_spans(self, request_id: str) -> List[Dict]:
        trace = self._traces[request_id]
        spans = []
        for index, event in enumerate(trace.events):
            spans.append(
                {
                    "trace_id": trace.request_id,
                    "span_id": f"{index + 1:016x}",
                    "name": f"{event.category}.{event.name}",
                    "start_time_unix_nano": int(event.started_at * 1_000_000_000),
                    "duration_ms": event.duration_ms,
                    "attributes": redact_sensitive(event.metadata),
                    "status": {"code": "ERROR" if event.error else "OK", "message": event.error or ""},
                }
            )
        return spans


trace_recorder = TraceRecorder()
