"""Thread-safe, pre-call budget reservations for agent work.

The manager owns the accounting state for one agent run.  Callers reserve
capacity before performing external work and then either commit the actual
usage or release the reservation when no call was made.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from threading import RLock
from typing import Callable, Literal, Optional
from uuid import uuid4


ReservationKind = Literal["model", "tool"]


class BudgetExceeded(ValueError):
    """Raised before a call that cannot fit inside the remaining budget."""

    def __init__(self, reason: str) -> None:
        super().__init__(reason)
        self.reason = reason


@dataclass(frozen=True)
class Reservation:
    """An immutable handle for capacity held by a :class:`BudgetManager`."""

    reservation_id: str
    manager_id: str
    kind: ReservationKind
    reserved_tokens: int = 0
    reserved_cost: float = 0.0
    estimated_input_tokens: int = 0
    max_output_tokens: int = 0
    tool_name: Optional[str] = None


@dataclass
class _ReservationState:
    reservation: Reservation
    status: Literal["reserved", "committed", "released"] = "reserved"


class BudgetManager:
    """Atomically account for model, tool, step, cost, and deadline budgets.

    A manager is intentionally scoped to a single request.  Its lock protects
    both the committed counters and outstanding reservations, so parallel
    planner tasks cannot all observe and spend the same remaining capacity.
    """

    def __init__(
        self,
        *,
        max_steps: int = 8,
        max_tool_calls: int = 5,
        max_tokens: int = 8000,
        max_cost: float = 1.0,
        deadline: Optional[float] = None,
        deadline_seconds: Optional[float] = None,
        max_duration_seconds: Optional[float] = None,
        used_steps: int = 0,
        used_tool_calls: int = 0,
        used_tokens: int = 0,
        used_cost: float = 0.0,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if deadline_seconds is not None and max_duration_seconds is not None:
            raise ValueError("set only one of deadline_seconds and max_duration_seconds")
        duration = deadline_seconds if deadline_seconds is not None else max_duration_seconds
        if deadline is not None and duration is not None:
            raise ValueError("set either deadline or a duration, not both")

        integer_values = {
            "max_steps": max_steps,
            "max_tool_calls": max_tool_calls,
            "max_tokens": max_tokens,
            "used_steps": used_steps,
            "used_tool_calls": used_tool_calls,
            "used_tokens": used_tokens,
        }
        if any(int(value) < 0 for value in integer_values.values()):
            raise ValueError("budget counters and limits must be non-negative")
        if float(max_cost) < 0 or float(used_cost) < 0:
            raise ValueError("budget costs must be non-negative")
        if duration is not None and float(duration) < 0:
            raise ValueError("deadline duration must be non-negative")

        self._id = str(uuid4())
        self._lock = RLock()
        self._clock = clock
        self._max_steps = int(max_steps)
        self._max_tool_calls = int(max_tool_calls)
        self._max_tokens = int(max_tokens)
        self._max_cost = float(max_cost)
        self._deadline = (
            float(deadline)
            if deadline is not None
            else (clock() + float(duration) if duration is not None else None)
        )

        self._used_steps = int(used_steps)
        self._used_tool_calls = int(used_tool_calls)
        self._used_tokens = int(used_tokens)
        self._used_cost = round(float(used_cost), 6)
        self._used_model_calls = 0
        self._model_cache_hits = 0

        self._reserved_tool_calls = 0
        self._reserved_tokens = 0
        self._reserved_cost = 0.0
        self._reservations: dict[str, _ReservationState] = {}

    @property
    def max_steps(self) -> int:
        return self._max_steps

    @property
    def max_tool_calls(self) -> int:
        return self._max_tool_calls

    @property
    def max_tokens(self) -> int:
        return self._max_tokens

    @property
    def max_cost(self) -> float:
        return self._max_cost

    @property
    def deadline(self) -> Optional[float]:
        return self._deadline

    @property
    def used_steps(self) -> int:
        with self._lock:
            return self._used_steps

    @property
    def used_tool_calls(self) -> int:
        with self._lock:
            return self._used_tool_calls

    @property
    def used_tokens(self) -> int:
        with self._lock:
            return self._used_tokens

    @property
    def used_cost(self) -> float:
        with self._lock:
            return self._used_cost

    @property
    def remaining_tokens(self) -> int:
        with self._lock:
            return max(0, self._max_tokens - self._used_tokens - self._reserved_tokens)

    @property
    def remaining_cost(self) -> float:
        with self._lock:
            return max(
                0.0,
                round(self._max_cost - self._used_cost - self._reserved_cost, 6),
            )

    @property
    def remaining_tool_calls(self) -> int:
        with self._lock:
            return max(
                0,
                self._max_tool_calls - self._used_tool_calls - self._reserved_tool_calls,
            )

    def remaining_output_tokens(self, estimated_input_tokens: int = 0) -> int:
        """Return the output cap available after reserving the estimated prompt."""

        estimated_input_tokens = max(0, int(estimated_input_tokens))
        return max(0, self.remaining_tokens - estimated_input_tokens)

    def remaining_seconds(self) -> Optional[float]:
        if self._deadline is None:
            return None
        return max(0.0, self._deadline - self._clock())

    def check_deadline(self) -> None:
        if self._deadline is not None and self._clock() >= self._deadline:
            raise BudgetExceeded("deadline_exceeded")

    def reserve_model_call(
        self,
        estimated_tokens: Optional[int] = None,
        estimated_cost: float = 0.0,
        *,
        estimated_input_tokens: int = 0,
        max_output_tokens: Optional[int] = None,
    ) -> Reservation:
        """Reserve prompt/output tokens and estimated cost for one model call.

        ``estimated_tokens`` is retained as a compact compatibility form for
        callers that only know a total estimate.  New call sites should pass
        prompt and maximum output tokens separately so the returned reservation
        exposes the exact provider-side output cap.
        """

        estimated_input_tokens = max(0, int(estimated_input_tokens))
        if estimated_tokens is not None:
            estimated_tokens = max(0, int(estimated_tokens))
        if max_output_tokens is not None:
            max_output_tokens = max(0, int(max_output_tokens))
        estimated_cost = max(0.0, float(estimated_cost))

        with self._lock:
            self._check_deadline_locked()
            remaining = self._max_tokens - self._used_tokens - self._reserved_tokens
            if estimated_tokens is not None and max_output_tokens is None:
                requested_total = estimated_tokens
                output_cap = max(0, requested_total - estimated_input_tokens)
            else:
                requested_output = remaining if max_output_tokens is None else max_output_tokens
                output_cap = min(requested_output, max(0, remaining - estimated_input_tokens))
                requested_total = estimated_input_tokens + output_cap

            if estimated_input_tokens >= remaining or output_cap <= 0 or requested_total <= 0:
                raise BudgetExceeded("max_tokens_exceeded")
            if requested_total > remaining:
                raise BudgetExceeded("max_tokens_exceeded")
            if self._used_cost + self._reserved_cost + estimated_cost > self._max_cost:
                raise BudgetExceeded("max_cost_exceeded")

            reservation = Reservation(
                reservation_id=str(uuid4()),
                manager_id=self._id,
                kind="model",
                reserved_tokens=requested_total,
                reserved_cost=round(estimated_cost, 6),
                estimated_input_tokens=estimated_input_tokens,
                max_output_tokens=output_cap,
            )
            self._reserved_tokens += reservation.reserved_tokens
            self._reserved_cost = round(
                self._reserved_cost + reservation.reserved_cost,
                6,
            )
            self._reservations[reservation.reservation_id] = _ReservationState(reservation)
            return reservation

    def reserve_tool_call(self, tool_name: Optional[str] = None) -> Reservation:
        """Atomically hold one tool-call slot."""

        with self._lock:
            self._check_deadline_locked()
            if self._used_tool_calls + self._reserved_tool_calls >= self._max_tool_calls:
                raise BudgetExceeded("max_tool_calls_exceeded")
            reservation = Reservation(
                reservation_id=str(uuid4()),
                manager_id=self._id,
                kind="tool",
                tool_name=tool_name,
            )
            self._reserved_tool_calls += 1
            self._reservations[reservation.reservation_id] = _ReservationState(reservation)
            return reservation

    def commit(
        self,
        reservation: Reservation,
        *,
        actual_tokens: Optional[int] = None,
        actual_cost: Optional[float] = None,
        cache_hit: bool = False,
    ) -> bool:
        """Commit actual usage exactly once; repeated terminal calls are no-ops."""

        with self._lock:
            state = self._reservation_state(reservation)
            if state.status != "reserved":
                return False
            self._remove_reserved_locked(reservation)
            if reservation.kind == "model":
                if cache_hit:
                    self._model_cache_hits += 1
                else:
                    tokens = reservation.reserved_tokens if actual_tokens is None else actual_tokens
                    cost = reservation.reserved_cost if actual_cost is None else actual_cost
                    self._used_tokens += max(0, int(tokens))
                    self._used_cost = round(self._used_cost + max(0.0, float(cost)), 6)
                    self._used_model_calls += 1
            else:
                self._used_tool_calls += 1
            state.status = "committed"
            return True

    def commit_model_call(
        self,
        reservation: Reservation,
        *,
        actual_tokens: Optional[int] = None,
        actual_cost: Optional[float] = None,
        cache_hit: bool = False,
    ) -> bool:
        self._require_kind(reservation, "model")
        return self.commit(
            reservation,
            actual_tokens=actual_tokens,
            actual_cost=actual_cost,
            cache_hit=cache_hit,
        )

    def commit_tool_call(self, reservation: Reservation) -> bool:
        self._require_kind(reservation, "tool")
        return self.commit(reservation)

    def release(self, reservation: Reservation) -> bool:
        """Release held capacity exactly once when no external call occurred."""

        with self._lock:
            state = self._reservation_state(reservation)
            if state.status != "reserved":
                return False
            self._remove_reserved_locked(reservation)
            state.status = "released"
            return True

    def release_model_call(self, reservation: Reservation) -> bool:
        self._require_kind(reservation, "model")
        return self.release(reservation)

    def release_tool_call(self, reservation: Reservation) -> bool:
        self._require_kind(reservation, "tool")
        return self.release(reservation)

    def record_step(self, count: int = 1) -> None:
        with self._lock:
            self._used_steps += max(0, int(count))

    def record_tool_call(self, count: int = 1) -> None:
        """Compatibility accounting for legacy callers without reservations."""

        with self._lock:
            self._used_tool_calls += max(0, int(count))

    def record_tokens(self, count: int) -> None:
        with self._lock:
            self._used_tokens += max(0, int(count))

    def record_cost(self, amount: float) -> None:
        with self._lock:
            self._used_cost = round(self._used_cost + max(0.0, float(amount)), 6)

    def stop_reason(self) -> Optional[str]:
        """Return the legacy post-accounting stop reason, if any."""

        with self._lock:
            if self._used_steps >= self._max_steps:
                return "max_steps_exceeded"
            if self._used_tool_calls >= self._max_tool_calls:
                return "max_tool_calls_exceeded"
            if self._used_tokens >= self._max_tokens:
                return "max_tokens_exceeded"
            if self._used_cost >= self._max_cost:
                return "max_cost_exceeded"
            if self._deadline is not None and self._clock() >= self._deadline:
                return "deadline_exceeded"
            return None

    def snapshot(self) -> dict[str, int | float | str | None]:
        """Return a consistent point-in-time view for diagnostics and tests."""

        with self._lock:
            remaining_tokens = max(
                0,
                self._max_tokens - self._used_tokens - self._reserved_tokens,
            )
            remaining_tools = max(
                0,
                self._max_tool_calls - self._used_tool_calls - self._reserved_tool_calls,
            )
            remaining_cost = max(
                0.0,
                round(self._max_cost - self._used_cost - self._reserved_cost, 6),
            )
            return {
                "manager_id": self._id,
                "max_steps": self._max_steps,
                "max_tool_calls": self._max_tool_calls,
                "max_tokens": self._max_tokens,
                "max_cost": self._max_cost,
                "deadline": self._deadline,
                "used_steps": self._used_steps,
                "used_tool_calls": self._used_tool_calls,
                "used_model_calls": self._used_model_calls,
                "used_tokens": self._used_tokens,
                "used_cost": self._used_cost,
                "model_cache_hits": self._model_cache_hits,
                "reserved_tool_calls": self._reserved_tool_calls,
                "reserved_model_calls": sum(
                    1
                    for state in self._reservations.values()
                    if state.status == "reserved" and state.reservation.kind == "model"
                ),
                "reserved_tokens": self._reserved_tokens,
                "reserved_cost": self._reserved_cost,
                "remaining_tool_calls": remaining_tools,
                "remaining_tokens": remaining_tokens,
                "remaining_output_tokens": remaining_tokens,
                "remaining_cost": remaining_cost,
                "remaining_seconds": self.remaining_seconds(),
            }

    def _check_deadline_locked(self) -> None:
        if self._deadline is not None and self._clock() >= self._deadline:
            raise BudgetExceeded("deadline_exceeded")

    def _reservation_state(self, reservation: Reservation) -> _ReservationState:
        if reservation.manager_id != self._id:
            raise ValueError("reservation belongs to a different budget manager")
        state = self._reservations.get(reservation.reservation_id)
        if state is None or state.reservation != reservation:
            raise ValueError("unknown reservation")
        return state

    @staticmethod
    def _require_kind(reservation: Reservation, kind: ReservationKind) -> None:
        if reservation.kind != kind:
            raise ValueError(f"expected a {kind} reservation")

    def _remove_reserved_locked(self, reservation: Reservation) -> None:
        if reservation.kind == "model":
            self._reserved_tokens -= reservation.reserved_tokens
            self._reserved_cost = round(
                self._reserved_cost - reservation.reserved_cost,
                6,
            )
        else:
            self._reserved_tool_calls -= 1
