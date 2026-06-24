import time
from collections import defaultdict, deque


class RateLimiter:
    def __init__(self, max_requests: int, window_seconds: int) -> None:
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self._requests = defaultdict(deque)

    def allow(self, key: str) -> bool:
        now = time.time()
        queue = self._requests[key]
        while queue and now - queue[0] > self.window_seconds:
            queue.popleft()
        if len(queue) >= self.max_requests:
            return False
        queue.append(now)
        return True
