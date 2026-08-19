"""Small, dependency-free statistics helpers for paired evaluation runs."""

from __future__ import annotations

import math
import random
from collections.abc import Sequence
from numbers import Real


def _rounded(value: float) -> float:
    rounded = round(value, 6)
    return 0.0 if rounded == 0 else rounded


def _numeric_pairs(
    baseline: Sequence[Real], candidate: Sequence[Real]
) -> tuple[list[float], list[float]]:
    if not baseline or len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must be non-empty and have equal length")

    normalized: list[list[float]] = [[], []]
    for target, values in zip(normalized, (baseline, candidate)):
        for value in values:
            if isinstance(value, bool) or not isinstance(value, Real):
                raise ValueError("paired values must be finite numbers")
            number = float(value)
            if not math.isfinite(number):
                raise ValueError("paired values must be finite numbers")
            target.append(number)
    return normalized[0], normalized[1]


def percentile(values: Sequence[Real], quantile: float) -> float:
    """Return a linearly interpolated percentile for a quantile in ``[0, 1]``."""

    if not values:
        raise ValueError("values must be non-empty")
    if isinstance(quantile, bool) or not isinstance(quantile, Real):
        raise ValueError("quantile must be between 0 and 1")
    quantile = float(quantile)
    if not math.isfinite(quantile) or not 0 <= quantile <= 1:
        raise ValueError("quantile must be between 0 and 1")

    ordered: list[float] = []
    for value in values:
        if isinstance(value, bool) or not isinstance(value, Real):
            raise ValueError("values must contain only finite numbers")
        number = float(value)
        if not math.isfinite(number):
            raise ValueError("values must contain only finite numbers")
        ordered.append(number)
    ordered.sort()

    position = (len(ordered) - 1) * quantile
    lower_index = math.floor(position)
    upper_index = math.ceil(position)
    if lower_index == upper_index:
        return _rounded(ordered[lower_index])
    weight = position - lower_index
    return _rounded(ordered[lower_index] * (1 - weight) + ordered[upper_index] * weight)


def _exact_two_sided_binomial_p_value(wins: int, losses: int) -> float:
    effective_pairs = wins + losses
    if effective_pairs == 0:
        return 1.0
    tail = min(wins, losses)
    probability = 2 * sum(math.comb(effective_pairs, k) for k in range(tail + 1))
    probability /= 2**effective_pairs
    return _rounded(min(1.0, probability))


def paired_bootstrap_mean_delta(
    baseline: Sequence[Real],
    candidate: Sequence[Real],
    iterations: int,
    confidence_level: float,
    seed: int,
) -> dict[str, object]:
    """Estimate a confidence interval by resampling paired case deltas."""

    baseline_values, candidate_values = _numeric_pairs(baseline, candidate)
    if isinstance(iterations, bool) or not isinstance(iterations, int) or iterations <= 0:
        raise ValueError("iterations must be a positive integer")
    if isinstance(confidence_level, bool) or not isinstance(confidence_level, Real):
        raise ValueError("confidence_level must be between 0 and 1")
    confidence_level = float(confidence_level)
    if not math.isfinite(confidence_level) or not 0 < confidence_level < 1:
        raise ValueError("confidence_level must be between 0 and 1")
    if isinstance(seed, bool) or not isinstance(seed, int):
        raise ValueError("seed must be an integer")

    differences = [candidate - base for base, candidate in zip(baseline_values, candidate_values)]
    sample_size = len(differences)
    rng = random.Random(seed)
    sampled_deltas = [
        sum(differences[rng.randrange(sample_size)] for _ in range(sample_size)) / sample_size
        for _ in range(iterations)
    ]
    alpha = 1 - confidence_level

    return {
        "n": sample_size,
        "baseline_mean": _rounded(sum(baseline_values) / sample_size),
        "candidate_mean": _rounded(sum(candidate_values) / sample_size),
        "delta": _rounded(sum(differences) / sample_size),
        "confidence_interval": {
            "level": _rounded(confidence_level),
            "lower": percentile(sampled_deltas, alpha / 2),
            "upper": percentile(sampled_deltas, 1 - alpha / 2),
        },
    }


def paired_sign_test(
    baseline: Sequence[Real], candidate: Sequence[Real]
) -> dict[str, int | float]:
    """Run an exact paired sign test, excluding equal pairs from the test."""

    baseline_values, candidate_values = _numeric_pairs(baseline, candidate)
    improved = sum(candidate > base for base, candidate in zip(baseline_values, candidate_values))
    regressed = sum(candidate < base for base, candidate in zip(baseline_values, candidate_values))
    ties = len(baseline_values) - improved - regressed

    return {
        "improved": improved,
        "regressed": regressed,
        "ties": ties,
        "effective_pairs": improved + regressed,
        "two_sided_p_value": _exact_two_sided_binomial_p_value(improved, regressed),
    }


def paired_binary_test(
    baseline: Sequence[bool], candidate: Sequence[bool]
) -> dict[str, int | float]:
    """Run an exact paired test over discordant binary outcomes."""

    if not baseline or len(baseline) != len(candidate):
        raise ValueError("baseline and candidate must be non-empty and have equal length")
    if any(type(value) is not bool for value in (*baseline, *candidate)):
        raise ValueError("paired binary values must be booleans")

    candidate_wins = sum(not base and current for base, current in zip(baseline, candidate))
    baseline_wins = sum(base and not current for base, current in zip(baseline, candidate))
    ties = len(baseline) - candidate_wins - baseline_wins

    return {
        "candidate_wins": candidate_wins,
        "baseline_wins": baseline_wins,
        "ties": ties,
        "effective_pairs": candidate_wins + baseline_wins,
        "two_sided_p_value": _exact_two_sided_binomial_p_value(
            candidate_wins, baseline_wins
        ),
    }
