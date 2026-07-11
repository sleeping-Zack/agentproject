from concurrent.futures import ThreadPoolExecutor

import pytest

from agent.budget import BudgetExceeded, BudgetManager


def test_parallel_tool_reservations_do_not_oversell():
    manager = BudgetManager(max_tool_calls=3)

    def reserve_and_commit():
        try:
            reservation = manager.reserve_tool_call("worker")
        except BudgetExceeded:
            return False
        manager.commit_tool_call(reservation)
        return True

    with ThreadPoolExecutor(max_workers=12) as pool:
        admitted = list(pool.map(lambda _: reserve_and_commit(), range(12)))

    assert sum(admitted) == 3
    snapshot = manager.snapshot()
    assert snapshot["used_tool_calls"] == 3
    assert snapshot["reserved_tool_calls"] == 0


def test_parallel_model_reservations_do_not_oversell_tokens():
    manager = BudgetManager(max_tokens=100)

    def reserve():
        try:
            return manager.reserve_model_call(estimated_tokens=30)
        except BudgetExceeded:
            return None

    with ThreadPoolExecutor(max_workers=10) as pool:
        reservations = list(pool.map(lambda _: reserve(), range(10)))

    admitted = [reservation for reservation in reservations if reservation is not None]
    assert len(admitted) == 3
    assert manager.snapshot()["reserved_tokens"] == 90

    for reservation in admitted:
        manager.release_model_call(reservation)
    assert manager.snapshot()["reserved_tokens"] == 0


def test_reservation_terminal_operations_are_idempotent():
    manager = BudgetManager(max_tokens=100, max_tool_calls=1)
    model = manager.reserve_model_call(estimated_tokens=40, estimated_cost=0.1)

    assert manager.commit_model_call(model, actual_tokens=12, actual_cost=0.02)
    assert not manager.commit_model_call(model, actual_tokens=99, actual_cost=0.9)
    assert not manager.release_model_call(model)
    assert manager.snapshot()["used_tokens"] == 12
    assert manager.snapshot()["used_cost"] == 0.02

    tool = manager.reserve_tool_call("rag")
    assert manager.release_tool_call(tool)
    assert not manager.release_tool_call(tool)
    assert not manager.commit_tool_call(tool)
    assert manager.snapshot()["used_tool_calls"] == 0


def test_deadline_rejects_before_reserving():
    now = [100.0]
    manager = BudgetManager(
        max_tokens=100,
        max_tool_calls=1,
        deadline=101.0,
        clock=lambda: now[0],
    )
    now[0] = 101.0

    with pytest.raises(BudgetExceeded, match="deadline_exceeded"):
        manager.reserve_model_call(estimated_tokens=10)
    with pytest.raises(BudgetExceeded, match="deadline_exceeded"):
        manager.reserve_tool_call("rag")

    assert manager.snapshot()["reserved_tokens"] == 0
    assert manager.snapshot()["reserved_tool_calls"] == 0


def test_model_cache_hit_releases_tokens_without_charging_usage():
    manager = BudgetManager(max_tokens=100)
    reservation = manager.reserve_model_call(estimated_tokens=50)

    manager.commit_model_call(reservation, actual_tokens=50, cache_hit=True)

    snapshot = manager.snapshot()
    assert snapshot["used_tokens"] == 0
    assert snapshot["reserved_tokens"] == 0
    assert snapshot["model_cache_hits"] == 1


def test_model_reservation_caps_output_to_remaining_tokens():
    manager = BudgetManager(max_tokens=20, used_tokens=5)

    reservation = manager.reserve_model_call(
        estimated_input_tokens=4,
        max_output_tokens=100,
    )

    assert reservation.max_output_tokens == 11
    assert reservation.reserved_tokens == 15
