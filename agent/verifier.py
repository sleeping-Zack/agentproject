from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

from rag.judge import JudgeScore, LLMJudge


@dataclass
class VerifyResult:
    passed: bool
    action: str
    score: float
    reasons: List[str] = field(default_factory=list)
    judge: Dict[str, Any] = field(default_factory=dict)


class AnswerVerifier:
    def __init__(
        self,
        judge: Optional[LLMJudge] = None,
        min_overall_score: float = 3.5,
        require_citation: bool = True,
    ) -> None:
        self.judge = judge
        self.min_overall_score = min_overall_score
        self.require_citation = require_citation

    def verify(
        self,
        query: str,
        answer: str,
        evidence: List[Dict[str, Any]],
    ) -> VerifyResult:
        reasons: List[str] = []
        context = "\n".join(str(item.get("content", "")) for item in evidence)

        citation_present = self._has_citation(answer, evidence)
        if self.require_citation and evidence and not citation_present:
            reasons.append("citation_missing")
        if evidence and not context.strip():
            reasons.append("evidence_empty")
        if not answer.strip():
            reasons.append("answer_empty")

        score = JudgeScore(5.0, 5.0, 5.0, "judge disabled")
        if self.judge is not None:
            score = self.judge.evaluate(query=query, context=context, answer=answer)
            if score.overall < self.min_overall_score:
                reasons.append("judge_score_below_threshold")

        passed = not reasons
        if passed:
            return VerifyResult(
                passed=True,
                action="accept",
                score=score.overall,
                judge=score.to_dict(),
            )
        action = "retry" if "judge_score_below_threshold" in reasons else "refuse"
        return VerifyResult(
            passed=False,
            action=action,
            score=score.overall,
            reasons=reasons,
            judge=score.to_dict(),
        )

    @staticmethod
    def _has_citation(answer: str, evidence: List[Dict[str, Any]]) -> bool:
        if "引用" in answer or "source" in answer.lower():
            return True
        for item in evidence:
            evidence_id = str(item.get("id") or item.get("source") or "").strip()
            if evidence_id and evidence_id in answer:
                return True
        return False
