from agent.tools.retry import RetryPolicy, run_with_retry


def test_run_with_retry_succeeds_after_transient_failures():
    calls = {"n": 0}

    def flaky():
        calls["n"] += 1
        if calls["n"] < 3:
            raise RuntimeError("temporary")
        return "ok"

    policy = RetryPolicy(max_attempts=5, base_delay=0.0, jitter=0)
    result = run_with_retry(flaky, policy=policy)

    assert result == "ok"
    assert calls["n"] == 3


def test_run_with_retry_does_not_retry_permission_errors():
    calls = {"n": 0}

    def always_denied():
        calls["n"] += 1
        raise PermissionError("denied")

    policy = RetryPolicy(max_attempts=5, base_delay=0.0, jitter=0)
    try:
        run_with_retry(always_denied, policy=policy)
        assert False, "should have raised"
    except PermissionError:
        pass

    assert calls["n"] == 1


def test_run_with_retry_stops_after_max_attempts():
    calls = {"n": 0}

    def always_fails():
        calls["n"] += 1
        raise RuntimeError("nope")

    policy = RetryPolicy(max_attempts=3, base_delay=0.0, jitter=0)
    try:
        run_with_retry(always_fails, policy=policy)
        assert False, "should have raised"
    except RuntimeError:
        pass

    assert calls["n"] == 3


def test_on_retry_callback_invoked():
    events = []

    def flaky():
        if len(events) < 1:
            raise RuntimeError("boom")
        return 1

    def on_retry(attempt, exc, wait):
        events.append((attempt, type(exc).__name__))

    policy = RetryPolicy(max_attempts=3, base_delay=0.0, jitter=0)
    assert run_with_retry(flaky, policy=policy, on_retry=on_retry) == 1
    assert events == [(1, "RuntimeError")]
