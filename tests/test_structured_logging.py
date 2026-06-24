import json
import logging
from io import StringIO

import pytest

from observability.context import bind_request_context, request_context
from utils.logger_handler import JsonFormatter


@pytest.fixture
def json_logger():
    log = logging.getLogger("test_struct_logger")
    log.handlers.clear()
    log.setLevel(logging.DEBUG)
    buffer = StringIO()
    handler = logging.StreamHandler(buffer)
    handler.setFormatter(JsonFormatter())
    log.addHandler(handler)
    yield log, buffer
    log.handlers.clear()


def _last_payload(buffer: StringIO) -> dict:
    lines = [line for line in buffer.getvalue().splitlines() if line.strip()]
    return json.loads(lines[-1])


def test_json_log_includes_request_context(json_logger):
    log, buffer = json_logger
    with bind_request_context(request_id="req-1", session_id="s-1", tenant_id="tenant-a"):
        log.info("hello")
    payload = _last_payload(buffer)
    assert payload["msg"] == "hello"
    assert payload["request_id"] == "req-1"
    assert payload["session_id"] == "s-1"
    assert payload["tenant_id"] == "tenant-a"
    assert payload["level"] == "INFO"


def test_json_log_redacts_sensitive(json_logger):
    log, buffer = json_logger
    log.info("api_key=sk-secret123456789")
    payload = _last_payload(buffer)
    assert "sk-secret123456789" not in payload["msg"]


def test_bind_request_context_restores_on_exit():
    assert request_context().request_id is None
    with bind_request_context(request_id="req-x"):
        assert request_context().request_id == "req-x"
    assert request_context().request_id is None
