import time

from observability.event_bus import EventBus


def test_event_bus_publish_and_consume():
    bus = EventBus()
    bus.publish("req-1", "tool_start", {"tool": "rag_summarize"})
    event = bus.consume("req-1", timeout=1.0)
    assert event != "closed"
    assert event.event == "tool_start"
    assert event.data["tool"] == "rag_summarize"
    assert event.sequence == 1
    assert event.request_id == "req-1"


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


def test_event_sequence_and_replay_after_last_event_id():
    bus = EventBus()
    first = bus.publish("req-replay", "run_started")
    second = bus.publish("req-replay", "token_delta", {"delta": "A"})
    third = bus.publish("req-replay", "token_delta", {"delta": "B"})

    assert [first.sequence, second.sequence, third.sequence] == [1, 2, 3]
    assert [event.sequence for event in bus.replay("req-replay", after_sequence=1)] == [2, 3]


def test_cancel_flag_is_visible_to_producer():
    bus = EventBus()
    bus.open("req-cancel")
    assert not bus.is_cancelled("req-cancel")

    bus.cancel("req-cancel")

    assert bus.is_cancelled("req-cancel")


def test_closed_channel_keeps_replay_for_reconnect():
    bus = EventBus()
    bus.publish("req-closed", "run_completed", {"answer": "ok"})
    bus.close("req-closed")

    assert bus.consume("req-closed", timeout=1.0).event_type == "run_completed"
    assert bus.consume("req-closed", timeout=1.0) == "closed"
    assert bus.replay("req-closed", after_sequence=0)[0].payload["answer"] == "ok"
