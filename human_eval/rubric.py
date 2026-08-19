from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence

import yaml


DEFAULT_RUBRIC_PATH = Path("config/evaluation_rubric.yml")


@dataclass(frozen=True)
class EvaluatedAnnotation:
    payload: Dict[str, Any]
    overall_score: Optional[float]
    passed: Optional[bool]


class RubricEvaluator:
    def __init__(self, path: Path | str = DEFAULT_RUBRIC_PATH) -> None:
        self.path = Path(path)
        self.config = yaml.safe_load(self.path.read_text(encoding="utf-8"))
        self.version = str(self.config["rubric_version"])
        self.dimensions = {
            str(item["id"]): dict(item) for item in self.config["dimensions"]
        }
        self.dimension_ids = tuple(self.dimensions)
        self.always_applicable = {
            dimension_id
            for dimension_id, item in self.dimensions.items()
            if item.get("always_applicable")
        }
        self.veto_rules = {
            str(item["id"]): dict(item) for item in self.config["veto_rules"]
        }
        self.invalid_reasons = set(self.config["invalid_case_reasons"])
        self.pass_conditions = dict(self.config["overall"]["pass_conditions"])
        self.precision = int(self.config["overall"].get("precision", 4))

    def public_definition(self) -> Dict[str, Any]:
        """Return the reviewer-visible rubric without internal pass thresholds."""
        return {
            "rubric_version": self.version,
            "scale": self.config["scale"],
            "dimensions": [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "question": item["question"],
                    "always_applicable": bool(item.get("always_applicable")),
                    "applicable_when": item.get("applicable_when"),
                    "anchors": item["anchors"],
                }
                for item in self.config["dimensions"]
            ],
            "veto_rules": [
                {
                    "id": item["id"],
                    "name": item["name"],
                    "description": item["description"],
                    "required_evidence": item["required_evidence"],
                    "forces_scores": item["forces_scores"],
                }
                for item in self.config["veto_rules"]
            ],
            "invalid_case_reasons": sorted(self.invalid_reasons),
            "confidence_levels": ["high", "medium", "low"],
        }

    def evaluate(self, submission: Mapping[str, Any]) -> EvaluatedAnnotation:
        valid = submission.get("valid")
        if type(valid) is not bool:
            raise ValueError("valid must be a boolean")
        scores = submission.get("scores")
        if not isinstance(scores, Mapping) or set(scores) != set(self.dimension_ids):
            raise ValueError("scores must contain every rubric dimension exactly once")
        vetoes = submission.get("vetoes")
        if not isinstance(vetoes, list) or len(vetoes) != len(set(vetoes)):
            raise ValueError("vetoes must be a unique list")
        unknown_vetoes = set(vetoes) - set(self.veto_rules)
        if unknown_vetoes:
            raise ValueError(f"unknown vetoes: {sorted(unknown_vetoes)}")
        rationales = submission.get("rationales")
        if not isinstance(rationales, Mapping):
            raise ValueError("rationales must be an object")
        confidence = submission.get("confidence")
        if confidence not in {"high", "medium", "low"}:
            raise ValueError("confidence must be high, medium, or low")
        duration_seconds = submission.get("duration_seconds")
        if type(duration_seconds) not in {int, float} or not 0 < duration_seconds <= 86400:
            raise ValueError("duration_seconds must be in (0, 86400]")

        invalid_reason = submission.get("invalid_reason")
        normalized_scores: Dict[str, Optional[int]] = {}
        if not valid:
            if invalid_reason not in self.invalid_reasons:
                raise ValueError("invalid annotations require a supported invalid_reason")
            if any(value is not None for value in scores.values()):
                raise ValueError("invalid annotations must use null for every score")
            if vetoes:
                raise ValueError("invalid annotations cannot declare vetoes")
            self._require_rationale(rationales, "validity")
            normalized_scores = {dimension_id: None for dimension_id in self.dimension_ids}
        else:
            if invalid_reason is not None:
                raise ValueError("valid annotations cannot declare invalid_reason")
            for dimension_id, value in scores.items():
                if value is None:
                    if dimension_id in self.always_applicable:
                        raise ValueError(f"{dimension_id} is always applicable")
                    normalized_scores[dimension_id] = None
                    continue
                if type(value) is not int or value not in {0, 1, 2, 3}:
                    raise ValueError(f"{dimension_id} must be an integer from 0 to 3 or null")
                normalized_scores[dimension_id] = value
                if value <= 1:
                    self._require_rationale(rationales, dimension_id)
            for veto_id in vetoes:
                self._require_rationale(rationales, f"veto:{veto_id}")
                for dimension_id, forced_score in self.veto_rules[veto_id][
                    "forces_scores"
                ].items():
                    if normalized_scores.get(dimension_id) != forced_score:
                        raise ValueError(
                            f"veto {veto_id} requires {dimension_id}={forced_score}"
                        )
        if confidence == "low":
            self._require_rationale(rationales, "confidence")

        payload = {
            "valid": valid,
            "invalid_reason": invalid_reason,
            "scores": normalized_scores,
            "vetoes": list(vetoes),
            "rationales": {str(key): str(value).strip() for key, value in rationales.items()},
            "confidence": confidence,
            "duration_seconds": float(duration_seconds),
        }
        if not valid:
            return EvaluatedAnnotation(payload=payload, overall_score=None, passed=None)
        overall_score = self.overall_score(normalized_scores)
        passed = self.case_passed(normalized_scores, vetoes, overall_score=overall_score)
        return EvaluatedAnnotation(
            payload=payload,
            overall_score=overall_score,
            passed=passed,
        )

    def overall_score(self, scores: Mapping[str, Optional[float]]) -> float:
        weighted_sum = 0.0
        weight_sum = 0.0
        for dimension_id, value in scores.items():
            if value is None:
                continue
            weight = float(self.dimensions[dimension_id]["weight"])
            weighted_sum += float(value) * weight
            weight_sum += weight
        if weight_sum <= 0:
            raise ValueError("at least one rubric dimension must be applicable")
        return round(weighted_sum / weight_sum, self.precision)

    def case_passed(
        self,
        scores: Mapping[str, Optional[float]],
        vetoes: Sequence[str],
        *,
        overall_score: Optional[float] = None,
    ) -> bool:
        if vetoes:
            return False
        overall = self.overall_score(scores) if overall_score is None else overall_score
        if overall < float(self.pass_conditions["minimum_overall_score"]):
            return False
        for dimension_id, minimum in self.pass_conditions[
            "minimum_dimension_scores"
        ].items():
            value = scores.get(dimension_id)
            if value is not None and float(value) < float(minimum):
                return False
        return True

    @staticmethod
    def _require_rationale(rationales: Mapping[str, Any], key: str) -> None:
        value = rationales.get(key)
        if not isinstance(value, str) or not value.strip():
            raise ValueError(f"rationale required for {key}")


def disagreement_triggers(
    first: Mapping[str, Any],
    second: Mapping[str, Any],
    *,
    minimum_score_gap: int = 2,
) -> list[str]:
    triggers: list[str] = []
    if bool(first["valid"]) != bool(second["valid"]):
        triggers.append("validity_disagreement")
    elif not first["valid"] and first.get("invalid_reason") != second.get("invalid_reason"):
        triggers.append("invalid_reason_disagreement")
    if first.get("passed") != second.get("passed"):
        triggers.append("pass_fail_disagreement")
    if set(first.get("vetoes") or []) != set(second.get("vetoes") or []):
        triggers.append("veto_disagreement")
    first_scores = first.get("scores") or {}
    second_scores = second.get("scores") or {}
    for dimension_id in sorted(set(first_scores) | set(second_scores)):
        left = first_scores.get(dimension_id)
        right = second_scores.get(dimension_id)
        if (left is None) != (right is None):
            triggers.append(f"applicability_disagreement:{dimension_id}")
        elif left is not None and right is not None and abs(left - right) >= minimum_score_gap:
            triggers.append(f"score_gap:{dimension_id}")
    if first.get("confidence") == "low" or second.get("confidence") == "low":
        triggers.append("low_confidence")
    return list(dict.fromkeys(triggers))


def quadratic_weighted_kappa(pairs: Sequence[tuple[int, int]]) -> Optional[float]:
    if not pairs:
        return None
    categories = 4
    observed = [[0.0 for _ in range(categories)] for _ in range(categories)]
    first_counts = [0.0] * categories
    second_counts = [0.0] * categories
    for first, second in pairs:
        observed[first][second] += 1.0
        first_counts[first] += 1.0
        second_counts[second] += 1.0
    total = float(len(pairs))
    weighted_observed = 0.0
    weighted_expected = 0.0
    for first in range(categories):
        for second in range(categories):
            weight = ((first - second) / (categories - 1)) ** 2
            weighted_observed += weight * observed[first][second] / total
            expected = (first_counts[first] / total) * (second_counts[second] / total)
            weighted_expected += weight * expected
    if weighted_expected == 0:
        return 1.0 if weighted_observed == 0 else None
    return round(1.0 - weighted_observed / weighted_expected, 4)
