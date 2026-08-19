from __future__ import annotations

import pytest

from evaluation_analysis.statistics import (
    paired_binary_test,
    paired_bootstrap_mean_delta,
    paired_sign_test,
    percentile,
)


def test_percentile_uses_linear_interpolation() -> None:
    values = [0.0, 10.0, 20.0, 30.0]

    assert percentile(values, 0.0) == 0.0
    assert percentile(values, 0.25) == 7.5
    assert percentile(values, 0.5) == 15.0
    assert percentile(values, 1.0) == 30.0


def test_bootstrap_mean_delta_is_paired_deterministic_and_rounded() -> None:
    baseline = [1.0, 2.0, 3.0, 4.0]
    candidate = [2.0, 4.0, 4.0, 6.0]

    result = paired_bootstrap_mean_delta(
        baseline,
        candidate,
        iterations=500,
        confidence_level=0.95,
        seed=7,
    )

    assert result == paired_bootstrap_mean_delta(
        baseline,
        candidate,
        iterations=500,
        confidence_level=0.95,
        seed=7,
    )
    assert result["n"] == 4
    assert result["baseline_mean"] == 2.5
    assert result["candidate_mean"] == 4.0
    assert result["delta"] == 1.5
    assert result["confidence_interval"]["level"] == 0.95
    assert result["confidence_interval"]["lower"] <= result["delta"]
    assert result["confidence_interval"]["upper"] >= result["delta"]
    assert all(
        value == round(value, 6)
        for value in (
            result["baseline_mean"],
            result["candidate_mean"],
            result["delta"],
            result["confidence_interval"]["lower"],
            result["confidence_interval"]["upper"],
        )
    )


def test_bootstrap_preserves_constant_paired_delta() -> None:
    result = paired_bootstrap_mean_delta(
        [1, 2, 3],
        [3, 4, 5],
        iterations=100,
        confidence_level=0.9,
        seed=11,
    )

    assert result["delta"] == 2.0
    assert result["confidence_interval"] == {"level": 0.9, "lower": 2.0, "upper": 2.0}


@pytest.mark.parametrize(
    ("baseline", "candidate", "iterations", "confidence_level"),
    [
        ([], [], 10, 0.95),
        ([1], [1, 2], 10, 0.95),
        ([1], [1], 0, 0.95),
        ([1], [1], True, 0.95),
        ([1], [1], 10, 0.0),
        ([1], [1], 10, 1.0),
        ([float("nan")], [1], 10, 0.95),
    ],
)
def test_bootstrap_rejects_invalid_inputs(
    baseline: list[float],
    candidate: list[float],
    iterations: int,
    confidence_level: float,
) -> None:
    with pytest.raises(ValueError):
        paired_bootstrap_mean_delta(
            baseline,
            candidate,
            iterations=iterations,
            confidence_level=confidence_level,
            seed=1,
        )


def test_bootstrap_accepts_positional_parameters_and_rejects_invalid_seed() -> None:
    result = paired_bootstrap_mean_delta([1], [2], 10, 0.95, 1)

    assert result["delta"] == 1.0
    with pytest.raises(ValueError):
        paired_bootstrap_mean_delta([1], [2], 10, 0.95, True)


def test_sign_test_ignores_ties_and_uses_exact_two_sided_probability() -> None:
    result = paired_sign_test(
        baseline=[1, 2, 3, 4, 5, 6, 7, 8],
        candidate=[2, 3, 4, 5, 4, 6, 7, 8],
    )

    assert result == {
        "improved": 4,
        "regressed": 1,
        "ties": 3,
        "effective_pairs": 5,
        "two_sided_p_value": 0.375,
    }


def test_sign_test_with_only_ties_has_unit_p_value() -> None:
    assert paired_sign_test([1, 2], [1, 2]) == {
        "improved": 0,
        "regressed": 0,
        "ties": 2,
        "effective_pairs": 0,
        "two_sided_p_value": 1.0,
    }


def test_exact_sign_test_caps_doubled_tail_probability_at_one() -> None:
    result = paired_sign_test([0, 0, 0, 0], [1, 1, 0, -1])

    assert result["two_sided_p_value"] == 1.0


def test_exact_sign_test_detects_one_sided_extreme() -> None:
    result = paired_sign_test([0] * 10, [1] * 10)

    assert result["two_sided_p_value"] == 0.001953


def test_binary_test_counts_candidate_and_baseline_wins() -> None:
    result = paired_binary_test(
        baseline=[False, False, True, True, False],
        candidate=[True, True, False, True, False],
    )

    assert result == {
        "candidate_wins": 2,
        "baseline_wins": 1,
        "ties": 2,
        "effective_pairs": 3,
        "two_sided_p_value": 1.0,
    }


@pytest.mark.parametrize("function", [paired_sign_test, paired_binary_test])
def test_paired_tests_reject_empty_or_unequal_inputs(function) -> None:
    with pytest.raises(ValueError):
        function([], [])
    with pytest.raises(ValueError):
        function([1], [1, 2])


def test_binary_test_rejects_non_boolean_values() -> None:
    with pytest.raises(ValueError):
        paired_binary_test([0, 1], [False, True])


@pytest.mark.parametrize("q", [-0.01, 1.01])
def test_percentile_rejects_invalid_quantile(q: float) -> None:
    with pytest.raises(ValueError):
        percentile([1, 2], q)
