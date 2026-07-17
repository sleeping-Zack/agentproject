import os
import time
from collections import defaultdict, deque
from threading import RLock
from uuid import uuid4


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests = defaultdict(deque)
        self._lock = RLock()

    def allow(self, key: str) -> bool:
        with self._lock:
            now = time.time()
            queue = self._requests[key]
            while queue and now - queue[0] > self.window_seconds:
                queue.popleft()
            if len(queue) >= self.max_requests:
                return False
            queue.append(now)
            return True


class RedisRateLimiter:
    _ALLOW_SCRIPT = """
local key = KEYS[1]
local now = tonumber(ARGV[1])
local window = tonumber(ARGV[2])
local limit = tonumber(ARGV[3])
local member = ARGV[4]
redis.call('ZREMRANGEBYSCORE', key, '-inf', now - window)
if redis.call('ZCARD', key) >= limit then
    return 0
end
redis.call('ZADD', key, now, member)
redis.call('PEXPIRE', key, math.ceil(window))
return 1
"""

    def __init__(
        self,
        max_requests: int,
        window_seconds: int,
        redis_url: str = "redis://127.0.0.1:6379/0",
        key_prefix: str = "agent:rate-limit",
        client=None,
    ) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.key_prefix = key_prefix.rstrip(":")
        if client is None:
            try:
                import redis
            except ImportError as exc:  # pragma: no cover - production dependency
                raise RuntimeError(
                    "Redis rate limiter requires the 'production' dependency extra"
                ) from exc
            client = redis.Redis.from_url(redis_url, decode_responses=True)
        self.client = client
        self._allow_script = self.client.register_script(self._ALLOW_SCRIPT)

    def allow(self, key: str) -> bool:
        now_ms = int(time.time() * 1000)
        result = self._allow_script(
            keys=[f"{self.key_prefix}:{key}"],
            args=[
                now_ms,
                self.window_seconds * 1000,
                self.max_requests,
                f"{now_ms}:{uuid4()}",
            ],
        )
        return bool(int(result))


def create_rate_limiter(max_requests: int, window_seconds: int):
    backend = os.getenv("AGENT_RATE_LIMIT_BACKEND", "memory").strip().lower()
    if backend == "memory":
        return RateLimiter(max_requests=max_requests, window_seconds=window_seconds)
    if backend == "redis":
        return RedisRateLimiter(
            max_requests=max_requests,
            window_seconds=window_seconds,
            redis_url=os.getenv("AGENT_REDIS_URL", "redis://127.0.0.1:6379/0"),
            key_prefix=os.getenv("AGENT_RATE_LIMIT_KEY_PREFIX", "agent:rate-limit"),
        )
    raise ValueError(f"unsupported rate limit backend: {backend}")
