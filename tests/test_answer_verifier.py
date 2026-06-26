from agent.verifier import AnswerVerifier
from rag.judge import JudgeScore


class StubJudge:
    def __init__(self, score: JudgeScore) -> None:
        self.score = score

    def evaluate(self, query: str, context: str, answer: str) -> JudgeScore:
        return self.score


def test_answer_verifier_accepts_grounded_answer_with_citation():
    verifier = AnswerVerifier(
        judge=StubJudge(JudgeScore(4.5, 4.2, 4.0, "ok")),
        min_overall_score=3.5,
    )

    result = verifier.verify(
        query="怎么保养滤网",
        answer="建议每周清理一次滤网。\n\n引用来源：manual-1",
        evidence=[{"id": "manual-1", "content": "滤网每周清理"}],
    )

    assert result.passed is True
    assert result.action == "accept"
    assert result.score >= 3.5


def test_answer_verifier_rejects_low_evidence_answer():
    verifier = AnswerVerifier(
        judge=StubJudge(JudgeScore(2.0, 2.0, 2.0, "unsupported")),
        min_overall_score=3.5,
    )

    result = verifier.verify(
        query="怎么保养滤网",
        answer="随便清理即可",
        evidence=[{"id": "manual-1", "content": "滤网每周清理"}],
    )

    assert result.passed is False
    assert result.action == "retry"
    assert "citation_missing" in result.reasons
