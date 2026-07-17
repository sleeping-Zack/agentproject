from __future__ import annotations

import atexit
import json
import os
from contextlib import nullcontext
from typing import Any, Dict


_tracer = None
_provider = None
_configured = False


def _enabled() -> bool:
    return os.getenv("AGENT_OTEL_ENABLED", "false").strip().lower() in {
        "1",
        "true",
        "yes",
        "on",
    }


def configure_telemetry(app=None):
    """Configure OTLP export once; return None when telemetry is disabled."""
    global _configured, _provider, _tracer
    if not _enabled():
        return None
    if _configured:
        return _tracer
    try:
        from opentelemetry import trace
        from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
    except ImportError as exc:  # pragma: no cover - production dependency
        raise RuntimeError(
            "OpenTelemetry export requires the 'production' dependency extra"
        ) from exc

    service_name = os.getenv("OTEL_SERVICE_NAME", "sweeper-agent")
    environment = os.getenv("AGENT_ENV", "local")
    _provider = TracerProvider(
        resource=Resource.create(
            {
                "service.name": service_name,
                "deployment.environment.name": environment,
            }
        )
    )
    exporter = OTLPSpanExporter(
        endpoint=os.getenv(
            "OTEL_EXPORTER_OTLP_TRACES_ENDPOINT",
            "http://127.0.0.1:4318/v1/traces",
        )
    )
    _provider.add_span_processor(BatchSpanProcessor(exporter))
    trace.set_tracer_provider(_provider)
    _tracer = trace.get_tracer("sweeper-agent")
    _configured = True
    atexit.register(_provider.shutdown)

    if app is not None and os.getenv(
        "AGENT_OTEL_INSTRUMENT_FASTAPI", "true"
    ).strip().lower() in {"1", "true", "yes", "on"}:
        try:
            from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
        except ImportError as exc:  # pragma: no cover - production dependency
            raise RuntimeError(
                "FastAPI instrumentation requires the 'production' dependency extra"
            ) from exc
        FastAPIInstrumentor.instrument_app(app)
    return _tracer


def start_otel_span(name: str, attributes: Dict[str, Any]):
    if _tracer is None:
        return nullcontext(None)
    return _tracer.start_as_current_span(
        name,
        attributes=_normalize_attributes(attributes),
        record_exception=False,
        set_status_on_exception=False,
    )


def mark_otel_error(span, error: BaseException) -> None:
    if span is None:
        return
    from opentelemetry.trace.status import Status, StatusCode

    span.record_exception(error)
    span.set_status(Status(StatusCode.ERROR, str(error)))


def emit_completed_otel_span(
    name: str,
    started_at: float,
    duration_ms: float,
    attributes: Dict[str, Any],
    error: str | None = None,
) -> None:
    if _tracer is None:
        return
    start_ns = int(started_at * 1_000_000_000)
    span = _tracer.start_span(
        name,
        start_time=start_ns,
        attributes=_normalize_attributes(attributes),
    )
    if error:
        from opentelemetry.trace.status import Status, StatusCode

        span.set_status(Status(StatusCode.ERROR, error))
        span.set_attribute("error.message", error)
    span.end(end_time=start_ns + int(duration_ms * 1_000_000))


def _normalize_attributes(attributes: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in attributes.items():
        if value is None:
            continue
        if isinstance(value, (bool, str, int, float)):
            normalized[str(key)] = value
            continue
        if isinstance(value, (list, tuple)) and all(
            isinstance(item, (bool, str, int, float)) for item in value
        ):
            normalized[str(key)] = list(value)
            continue
        normalized[str(key)] = json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            default=str,
        )
    return normalized
