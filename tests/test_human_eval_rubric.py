import pytest

from human_eval.rubric import (
    RubricEvaluator,
    disagreement_triggers,
    quadratic_weighted_kappa,
)


def _scores(default=3):
    return {
        "task_completion": default,
        "factual_correctness": default,
        "tool_use": default,
        "instruction_following": default,
        "groundedness": default,
        "safety": default,
        "response_quality": default,
    }


def _submission(**overrides):
    payload = {
        "valid": True,
        "invalid_reason": None,
        "scores": _scores(),
        "vetoes": [],
        "rationales": {},
        "confidence": "high",
        "duration_seconds": 45,
    }
    payload.update(overrides)
    return payload


def test_rubric_evaluates_valid_and_invalid_annotations():
    rubric = RubricEvaluator()

    valid = rubric.evaluate(_submission())
    invalid = rubric.evaluate(
        _submission(
            valid=False,
            invalid_reason="missing_reference",
            scores=_scores(None),
            rationales={"validity": "缺少用于核验结论的参考材料"},
        )
    )

    assert valid.overall_score == 3.0
    assert valid.passed is True
    assert invalid.overall_score is None
    assert invalid.passed is None


def test_rubric_rejects_low_score_without_evidence_rationale():
    rubric = RubricEvaluator()
    scores = _scores()
    scores["task_completion"] = 1

    with pytest.raises(ValueError, match="rationale required for task_completion"):
        rubric.evaluate(_submission(scores=scores))


def test_rubric_enforces_veto_score_and_low_confidence_rationale():
    rubric = RubricEvaluator()
    scores = _scores()
    scores["safety"] = 0
    evaluated = rubric.evaluate(
        _submission(
            scores=scores,
            vetoes=["unsafe_action"],
            rationales={
                "safety": "回答给出了可能损坏设备的可执行操作",
                "veto:unsafe_action": "Trace 与安全规则显示该操作不应执行",
            },
        )
    )
    assert evaluated.passed is False

    with pytest.raises(ValueError, match="rationale required for confidence"):
        rubric.evaluate(_submission(confidence="low"))


def test_disagreement_triggers_cover_validity_veto_score_and_confidence():
    first = RubricEvaluator().evaluate(_submission()).payload
    first.update({"passed": True, "overall_score": 3.0})
    scores = _scores()
    scores["safety"] = 0
    second = RubricEvaluator().evaluate(
        _submission(
            scores=scores,
            vetoes=["unsafe_action"],
            confidence="low",
            rationales={
                "safety": "存在严重安全风险",
                "veto:unsafe_action": "输出包含危险操作步骤",
                "confidence": "参考材料对设备型号描述不完整",
            },
        )
    ).payload
    second.update({"passed": False, "overall_score": 2.7})

    triggers = disagreement_triggers(first, second)

    assert "pass_fail_disagreement" in triggers
    assert "veto_disagreement" in triggers
    assert "score_gap:safety" in triggers
    assert "low_confidence" in triggers


def test_quadratic_weighted_kappa_handles_perfect_and_empty_samples():
    assert quadratic_weighted_kappa([]) is None
    assert quadratic_weighted_kappa([(0, 0), (1, 1), (2, 2), (3, 3)]) == 1.0
    assert quadratic_weighted_kappa([(0, 3), (3, 0)]) == -1.0
