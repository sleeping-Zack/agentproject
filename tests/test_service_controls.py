
from services.persistence import SQLiteStore
from services.rate_limit import RateLimiter


def test_sqlite_store_persists_session_and_trace(tmp_path):
    store = SQLiteStore(str(tmp_path / "agent.db"))

    store.save_session_message("s1", "user", "你好")
    store.save_trace("req-1", "s1", {"events": [{"name": "tool"}]})

    assert store.get_session_messages("s1") == [{"role": "user", "content": "你好"}]
    assert store.get_trace("req-1")["events"][0]["name"] == "tool"


def test_rate_limiter_blocks_after_limit():
    limiter = RateLimiter(max_requests=2, window_seconds=60)

    assert limiter.allow("client-1")
    assert limiter.allow("client-1")
    assert not limiter.allow("client-1")
