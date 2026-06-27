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
        scene: str = "general",
        tool_results: Optional[List[Dict[str, Any]]] = None,
        artifacts: Optional[List[Dict[str, Any]]] = None,
    ) -> VerifyResult:
        reasons: List[str] = []
        tool_results = tool_results or []
        artifacts = artifacts or []
        context = "\n".join(str(item.get("content", "")) for item in evidence)

        citation_present = self._has_citation(answer, evidence)
        if scene in {"rag", "rag_qa", "qa"}:
            if not evidence:
                reasons.append("evidence_required")
            if not citation_present:
                reasons.append("citation_missing")
        if scene in {"report", "monthly_report"}:
            has_report_artifact = any(
                artifact.get("type") in {"usage_record", "report", "tool_results"}
                or artifact.get("artifact_type") in {"usage_record", "report", "tool_results"}
                for artifact in artifacts
            )
            if not tool_results and not has_report_artifact:
                reasons.append("report_support_required")
        if self.require_citation and evidence and not citation_present:
            reasons.append("citation_missing")
        if evidence and not context.strip():
            reasons.append("evidence_empty")
        if not answer.strip():
            reasons.append("answer_empty")
        if answer.strip().startswith("请求未执行") and scene == "general":
            reasons.append("unexpected_refusal")

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
        retry_reasons = {"judge_score_below_threshold"}
        action = "retry" if retry_reasons.intersection(reasons) else "refuse"
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
