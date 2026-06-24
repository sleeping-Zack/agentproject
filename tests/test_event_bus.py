import time

from observability.event_bus import EventBus


def test_event_bus_publish_and_consume():
    bus = EventBus()
    bus.publish("req-1", "tool_start", {"tool": "rag_summarize"})
    event = bus.consume("req-1", timeout=1.0)
    assert event != "closed"
    assert event.event == "tool_start"
    assert event.data["tool"] == "rag_summarize"


def test_event_bus_close_signals_consumer():
    bus = EventBus()
    bus.close("req-2")
    result = bus.consume("req-2", timeout=1.0)
    assert result == "closed"


def test_event_bus_returns_none_on_timeout():
    bus = EventBus()
    start = time.time()
    result = bus.consume("req-3", timeout=0.1)
    assert result is None
    assert time.time() - start < 0.5


def test_event_bus_per_request_isolation():
    bus = EventBus()
    bus.publish("req-a", "tool_start", {"x": 1})
    bus.publish("req-b", "tool_start", {"x": 2})

    a = bus.consume("req-a", timeout=0.5)
    b = bus.consume("req-b", timeout=0.5)
    assert a.data["x"] == 1
    assert b.data["x"] == 2
