import time

import pytest

from services.circuit_breaker import (CircuitBreaker, CircuitBreakerRegistry,
                                       CircuitOpenError, CircuitState)


def test_circuit_opens_after_threshold_failures():
    breaker = CircuitBreaker(failure_threshold=3, recovery_timeout=1.0)
    assert breaker.state == CircuitState.CLOSED
    for _ in range(3):
        breaker.record_failure()
    assert breaker.state == CircuitState.OPEN
    assert breaker.allow() is False


def test_circuit_recovers_via_half_open():
    breaker = CircuitBreaker(failure_threshold=2, recovery_timeout=0.05)
    breaker.record_failure()
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN

    time.sleep(0.1)
    assert breaker.allow() is True  # 进入 HALF_OPEN
    assert breaker.state == CircuitState.HALF_OPEN

    breaker.record_success()
    assert breaker.state == CircuitState.CLOSED


def test_half_open_failure_returns_to_open():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=0.05)
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN

    time.sleep(0.1)
    assert breaker.allow() is True
    breaker.record_failure()
    assert breaker.state == CircuitState.OPEN


def test_call_short_circuits_when_open():
    breaker = CircuitBreaker(failure_threshold=1, recovery_timeout=10)
    breaker.record_failure()

    with pytest.raises(CircuitOpenError):
        breaker.call(lambda: "should not run")


def test_call_records_success_and_failure():
    breaker = CircuitBreaker(failure_threshold=2)

    assert breaker.call(lambda: "ok") == "ok"
    assert breaker.state == CircuitState.CLOSED

    def boom():
        raise RuntimeError("boom")

    for _ in range(2):
        with pytest.raises(RuntimeError):
            breaker.call(boom)
    assert breaker.state == CircuitState.OPEN


def test_registry_reuses_breaker_by_name():
    reg = CircuitBreakerRegistry()
    a = reg.get("model:foo")
    b = reg.get("model:foo")
    assert a is b
    snapshot = reg.snapshot()
    assert "model:foo" in snapshot
