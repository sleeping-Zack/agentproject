"""Retry policy for tool invocations.

Wraps a callable with bounded exponential backoff. Distinguishes between
transient errors (worth retrying) and permanent ones (PermissionError /
ValueError from safety layer) which we want to surface immediately.
"""
from __future__ import annotations

import random
import time
from dataclasses import dataclass
from typing import Callable, Tuple, Type


@dataclass
class RetryPolicy:
    max_attempts: int = 3
    base_delay: float = 0.2
    max_delay: float = 2.0
    jitter: float = 0.1
    non_retryable: Tuple[Type[BaseException], ...] = (
        PermissionError,
        ValueError,
    )

    def is_retryable(self, exc: BaseException) -> bool:
        return not isinstance(exc, self.non_retryable)

    def delay_for(self, attempt: int) -> float:
        backoff = min(self.max_delay, self.base_delay * (2 ** (attempt - 1)))
        return backoff + random.uniform(0, self.jitter)


def run_with_retry(
    func: Callable,
    *,
    policy: RetryPolicy,
    on_retry: Callable[[int, BaseException, float], None] | None = None,
):
    last_exc: BaseException | None = None
    for attempt in range(1, policy.max_attempts + 1):
        try:
            return func()
        except BaseException as exc:
            last_exc = exc
            if attempt >= policy.max_attempts or not policy.is_retryable(exc):
                raise
            wait = policy.delay_for(attempt)
            if on_retry is not None:
                on_retry(attempt, exc, wait)
            time.sleep(wait)
    if last_exc is not None:
        raise last_exc
