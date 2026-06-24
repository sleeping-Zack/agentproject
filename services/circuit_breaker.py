"""三态断路器：CLOSED → OPEN → HALF_OPEN → CLOSED。

应用场景：
    - 模型调用：连续失败 N 次后熔断，触发 ModelRouter 降级到备用模型
    - 工具调用：单个工具熔断后短路返回兜底说明，避免拖累整条链路
    - 外部依赖：RAG 向量库探测连续失败 → 降级为纯模型直答

为什么自己写不用 pybreaker：1) 零依赖；2) 状态需要被 /metrics 暴露，自己写
能直接 hook metrics_registry；3) 体量很小（一百多行），自己实现更可控。

阈值含义：
    - failure_threshold：连续失败多少次进入 OPEN
    - recovery_timeout：OPEN 状态多久后允许一次半开探测
    - half_open_max_calls：HALF_OPEN 状态最多允许多少次试探
"""
from __future__ import annotations

import time
from enum import Enum
from threading import RLock
from typing import Callable, Optional

from observability.metrics import metrics_registry


class CircuitState(Enum):
    CLOSED = "closed"
    OPEN = "open"
    HALF_OPEN = "half_open"


class CircuitOpenError(RuntimeError):
    """断路器开路时调用方应该立刻降级。"""


class CircuitBreaker:
    def __init__(
        self,
        name: str = "default",
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
        half_open_max_calls: int = 1,
    ) -> None:
        self.name = name
        self.failure_threshold = failure_threshold
        self.recovery_timeout = recovery_timeout
        self.half_open_max_calls = half_open_max_calls
        self._state = CircuitState.CLOSED
        self._failure_count = 0
        self._opened_at: Optional[float] = None
        self._half_open_calls = 0
        self._lock = RLock()

    @property
    def state(self) -> CircuitState:
        return self._state

    @property
    def failure_count(self) -> int:
        return self._failure_count

    def _transition(self, new_state: CircuitState) -> None:
        if self._state == new_state:
            return
        self._state = new_state
        metrics_registry.inc_counter(
            "agent_circuit_state_transition_total",
            {"name": self.name, "state": new_state.value},
        )

    def allow(self) -> bool:
        """非阻塞地判断当前是否允许调用。供 ModelRouter 在选模型时做健康度过滤。"""
        with self._lock:
            if self._state == CircuitState.CLOSED:
                return True
            if self._state == CircuitState.OPEN:
                if self._opened_at and time.time() - self._opened_at >= self.recovery_timeout:
                    self._transition(CircuitState.HALF_OPEN)
                    self._half_open_calls = 0
                else:
                    return False
            if self._state == CircuitState.HALF_OPEN:
                if self._half_open_calls < self.half_open_max_calls:
                    self._half_open_calls += 1
                    return True
                return False
            return True

    def record_success(self) -> None:
        with self._lock:
            self._failure_count = 0
            if self._state in (CircuitState.HALF_OPEN, CircuitState.OPEN):
                self._transition(CircuitState.CLOSED)
                self._opened_at = None
                self._half_open_calls = 0

    def record_failure(self) -> None:
        with self._lock:
            self._failure_count += 1
            if self._state == CircuitState.HALF_OPEN:
                self._transition(CircuitState.OPEN)
                self._opened_at = time.time()
                return
            if self._failure_count >= self.failure_threshold:
                self._transition(CircuitState.OPEN)
                self._opened_at = time.time()

    def call(self, fn: Callable, *args, **kwargs):
        """直接保护一次同步调用：开路则抛 CircuitOpenError，调用方负责降级。"""
        if not self.allow():
            metrics_registry.inc_counter(
                "agent_circuit_short_circuit_total", {"name": self.name},
            )
            raise CircuitOpenError(f"circuit [{self.name}] is open")
        try:
            result = fn(*args, **kwargs)
        except Exception:
            self.record_failure()
            raise
        else:
            self.record_success()
            return result


class CircuitBreakerRegistry:
    """按名字管理一组断路器；同名复用，便于把指标聚合到同一维度。"""

    def __init__(self) -> None:
        self._breakers: dict[str, CircuitBreaker] = {}
        self._lock = RLock()

    def get(
        self,
        name: str,
        failure_threshold: int = 5,
        recovery_timeout: float = 30.0,
    ) -> CircuitBreaker:
        with self._lock:
            breaker = self._breakers.get(name)
            if breaker is None:
                breaker = CircuitBreaker(
                    name=name,
                    failure_threshold=failure_threshold,
                    recovery_timeout=recovery_timeout,
                )
                self._breakers[name] = breaker
            return breaker

    def snapshot(self) -> dict:
        with self._lock:
            return {n: {"state": b.state.value, "failures": b.failure_count}
                    for n, b in self._breakers.items()}


breaker_registry = CircuitBreakerRegistry()
