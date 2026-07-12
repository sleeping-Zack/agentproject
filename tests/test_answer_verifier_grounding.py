from agent.answer_schema import AnswerClaim, StructuredAnswer
from agent.verifier import AnswerVerifier, build_default_answer_verifier
from rag.judge import JudgeScore


def _answer(*claims: AnswerClaim, citations: list[str]) -> StructuredAnswer:
    return StructuredAnswer(
        summary="维护建议",
        claims=list(claims),
        citations=citations,
    )


def test_structured_answer_accepts_current_evidence_id_and_numeric_reference():
    evidence = [{"id": "manual-current", "content": "滤网应当每周清理。"}]
    verifier = AnswerVerifier()

    by_id = verifier.verify(
        query="滤网如何维护",
        answer="滤网每周清理。",
        evidence=evidence,
        scene="qa",
        structured_answer=_answer(
            AnswerClaim("滤网每周清理", ["manual-current"]),
            citations=["manual-current"],
        ),
    )
    by_index = verifier.verify(
        query="滤网如何维护",
        answer="滤网每周清理。",
        evidence=evidence,
        scene="qa",
        structured_answer=_answer(
            AnswerClaim("滤网每周清理", ["[1]"]),
            citations=["[1]"],
        ),
    )

    assert by_id.passed is True
    assert by_index.passed is True
    assert by_id.citation_validity == 1.0
    assert by_id.citation_coverage == 1.0
    assert by_id.unsupported_claim_rate == 0.0


def test_structured_answer_rejects_cross_request_evidence_ids():
    result = AnswerVerifier().verify(
        query="滤网如何维护",
        answer="滤网每周清理。",
        evidence=[{"id": "request-b-doc", "content": "滤网应当每周清理。"}],
        scene="qa",
        structured_answer=_answer(
            AnswerClaim("滤网每周清理", ["request-a-doc"]),
            citations=["request-a-doc"],
        ),
    )

    assert result.passed is False
    assert "citation_invalid" in result.reasons
    assert "claim_evidence_id_invalid" in result.reasons
    assert result.citation_validity == 0.0
    assert result.invalid_citations == ["request-a-doc"]


def test_citation_placeholder_is_not_treated_as_a_citation():
    result = AnswerVerifier().verify(
        query="滤网如何维护",
        answer="滤网每周清理。\n\n引用：暂无。",
        evidence=[{"id": "manual-current", "content": "滤网应当每周清理。"}],
        scene="qa",
    )

    assert result.passed is False
    assert "citation_placeholder" in result.reasons
    assert "citation_missing" in result.reasons
    assert result.citation_validity == 0.0


def test_partial_claim_support_exposes_coverage_and_unsupported_rate():
    result = AnswerVerifier().verify(
        query="如何维护设备",
        answer="滤网每周清理，电池续航十小时。",
        evidence=[
            {"id": "maintenance", "content": "滤网应当每周清理。"},
            {"id": "dust-bin", "content": "尘盒容量为500毫升。"},
        ],
        scene="qa",
        structured_answer=_answer(
            AnswerClaim("滤网每周清理", ["maintenance"]),
            AnswerClaim("电池续航可达十小时", ["dust-bin"]),
            citations=["maintenance", "dust-bin"],
        ),
    )

    assert result.passed is False
    assert result.citation_validity == 1.0
    assert result.citation_coverage == 1.0
    assert result.unsupported_claim_rate == 0.5
    assert result.claim_support[0]["supported"] is True
    assert result.claim_support[1]["supported"] is False
    assert "unsupported_claim_rate_exceeded" in result.reasons


def test_contradictory_harmful_instruction_is_rejected():
    result = AnswerVerifier().verify(
        query="电机如何清洁",
        answer="可以直接用水冲洗电机。",
        evidence=[{"id": "safety", "content": "严禁用水冲洗电机，以免设备损坏。"}],
        scene="qa",
        structured_answer=_answer(
            AnswerClaim("可以直接用水冲洗电机", ["safety"]),
            citations=["safety"],
        ),
    )

    assert result.passed is False
    assert result.harmful_instruction is True
    assert result.claim_support[0]["contradiction"] is True
    assert "evidence_contradiction" in result.reasons
    assert "harmful_instruction" in result.reasons
    assert result.action == "refuse"


def test_disabled_judge_is_explicitly_not_evaluated_without_fake_score():
    result = AnswerVerifier().verify(
        query="滤网如何维护",
        answer="滤网每周清理。",
        evidence=[{"id": "manual-current", "content": "滤网应当每周清理。"}],
        scene="qa",
        structured_answer=_answer(
            AnswerClaim("滤网每周清理", ["manual-current"]),
            citations=["manual-current"],
        ),
    )

    assert result.passed is True
    assert result.score is None
    assert result.judge == {
        "status": "not_evaluated",
        "reason": "judge_not_configured",
    }


class CountingJudge:
    def __init__(self, score: JudgeScore | None = None) -> None:
        self.calls = 0
        self.score = score or JudgeScore(5.0, 5.0, 5.0, "ok")

    def evaluate(self, query: str, context: str, answer: str) -> JudgeScore:
        self.calls += 1
        return self.score


def test_judge_is_only_called_for_high_risk_or_low_confidence_results():
    judge = CountingJudge()
    verifier = AnswerVerifier(judge=judge)
    evidence = [{"id": "manual-current", "content": "滤网应当每周清理。"}]

    safe = verifier.verify(
        query="滤网如何维护",
        answer="滤网每周清理。",
        evidence=evidence,
        scene="qa",
        structured_answer=_answer(
            AnswerClaim("滤网每周清理", ["manual-current"]),
            citations=["manual-current"],
        ),
    )
    risky = verifier.verify(
        query="滤网如何维护",
        answer="滤网永远不需要清理。",
        evidence=evidence,
        scene="qa",
        structured_answer=_answer(
            AnswerClaim("滤网永远不需要清理", ["manual-current"]),
            citations=["manual-current"],
        ),
    )

    assert safe.judge["status"] == "not_evaluated"
    assert risky.judge["status"] == "evaluated"
    assert judge.calls == 1


def test_evaluated_judge_must_meet_faithfulness_threshold():
    judge = CountingJudge(JudgeScore(5.0, 3.9, 5.0, "faithfulness is low"))
    result = AnswerVerifier(judge=judge).verify(
        query="滤网如何维护",
        answer="滤网每周清理。\n\n引用来源：manual-current",
        evidence=[{"id": "manual-current", "content": "滤网应当每周清理。"}],
        scene="qa",
    )

    assert result.judge["status"] == "evaluated"
    assert result.score == judge.score.overall
    assert result.passed is False
    assert "judge_faithfulness_below_threshold" in result.reasons


def test_default_verifier_factory_wires_judge_only_when_enabled(monkeypatch):
    config = {
        "min_overall_score": 3.7,
        "min_faithfulness_score": 4.1,
        "llm_judge": {"enabled": False, "timeout_seconds": 9},
    }
    monkeypatch.delenv("AGENT_LLM_JUDGE_ENABLED", raising=False)

    disabled = build_default_answer_verifier(config)
    monkeypatch.setenv("AGENT_LLM_JUDGE_ENABLED", "true")
    enabled = build_default_answer_verifier(config)

    assert disabled.judge is None
    assert enabled.judge is not None
    assert enabled.judge.timeout_seconds == 9
    assert enabled.min_overall_score == 3.7


def test_judge_failure_is_explicit_and_fails_closed():
    judge = CountingJudge(
        JudgeScore(
            3.0,
            3.0,
            3.0,
            "judge timeout",
            success=False,
            error_code="timeout",
        )
    )
    result = AnswerVerifier(judge=judge).verify(
        query="滤网如何维护",
        answer="滤网也许不用清理。\n\n引用来源：manual-current",
        evidence=[{"id": "manual-current", "content": "滤网应当每周清理。"}],
        scene="qa",
    )

    assert result.passed is False
    assert result.judge["status"] == "error"
    assert "judge_unavailable" in result.reasons
