"""LLM-as-Judge evaluation for generated answers.

Industry-standard practice for evaluating Agent / RAG output is to use a
strong LLM as a judge that scores responses on dimensions such as
correctness, faithfulness (groundedness w.r.t. retrieved context) and
completeness. This module gives the project that capability while remaining
test-friendly:

    * `JudgeClient` accepts any callable `invoker(prompt) -> str` so unit
      tests can plug in a deterministic stub without burning API quota.
    * `LLMJudge.evaluate(...)` parses the judge's structured response into
      typed scores and falls back to safe defaults on parse failure.
    * `evaluate_batch(...)` aggregates per-case results into mean scores
      that can be uploaded to metrics_registry as gauges, so the same
      observability stack covers retrieval AND generation quality.
"""
from __future__ import annotations

import json
import re
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional

from observability.metrics import metrics_registry

JUDGE_PROMPT_TEMPLATE = """你是一名严格的中文 Agent 评测员，需要根据以下维度对回答打分（1-5 分，5 分最好）：
1. correctness：回答是否事实正确，是否回应了用户的问题
2. faithfulness：回答是否忠实于提供的参考资料，没有捏造
3. completeness：回答是否覆盖了用户问题中的关键点

只输出 JSON，键为：correctness, faithfulness, completeness, rationale。
rationale 用一句话中文，不要超过 80 字。

[问题]
{query}

[参考资料]
{context}

[被评测的回答]
{answer}

请直接输出 JSON："""


@dataclass
class JudgeScore:
    correctness: float
    faithfulness: float
    completeness: float
    rationale: str = ""

    @property
    def overall(self) -> float:
        return round(
            (self.correctness + self.faithfulness + self.completeness) / 3, 4
        )

    def to_dict(self) -> Dict[str, float]:
        return {
            "correctness": self.correctness,
            "faithfulness": self.faithfulness,
            "completeness": self.completeness,
            "overall": self.overall,
            "rationale": self.rationale,
        }


@dataclass
class JudgeBatchResult:
    cases: List[Dict] = field(default_factory=list)
    aggregate: Dict[str, float] = field(default_factory=dict)


def _coerce_score(value, default: float = 3.0) -> float:
    try:
        score = float(value)
    except (TypeError, ValueError):
        return default
    if score < 1:
        return 1.0
    if score > 5:
        return 5.0
    return round(score, 4)


def _parse_judge_response(raw: str) -> JudgeScore:
    text = (raw or "").strip()
    if not text:
        return JudgeScore(3.0, 3.0, 3.0, rationale="empty judge response")
    match = re.search(r"\{.*\}", text, re.DOTALL)
    if match:
        text = match.group(0)
    try:
        payload = json.loads(text)
    except json.JSONDecodeError:
        return JudgeScore(3.0, 3.0, 3.0, rationale="failed to parse judge json")
    return JudgeScore(
        correctness=_coerce_score(payload.get("correctness")),
        faithfulness=_coerce_score(payload.get("faithfulness")),
        completeness=_coerce_score(payload.get("completeness")),
        rationale=str(payload.get("rationale", "")).strip(),
    )


class LLMJudge:
    """Wrap any text-in/text-out callable as a structured grader."""

    def __init__(self, invoker: Optional[Callable[[str], str]] = None) -> None:
        self._invoker = invoker

    def _default_invoker(self, prompt: str) -> str:
        from model.factory import chat_model

        response = chat_model.invoke(prompt)
        content = getattr(response, "content", response)
        return str(content)

    def evaluate(self, query: str, context: str, answer: str) -> JudgeScore:
        prompt = JUDGE_PROMPT_TEMPLATE.format(
            query=query, context=context or "（无）", answer=answer or "（空）",
        )
        invoker = self._invoker or self._default_invoker
        try:
            raw = invoker(prompt)
        except Exception as exc:
            return JudgeScore(3.0, 3.0, 3.0, rationale=f"judge error: {exc}")
        return _parse_judge_response(raw)


def evaluate_batch(
    cases: List[Dict],
    judge: Optional[LLMJudge] = None,
    record_metrics: bool = True,
) -> JudgeBatchResult:
    """Score a list of cases. Each case is {query, context, answer, ...}."""

    judge = judge or LLMJudge()
    scored = []
    totals = {"correctness": 0.0, "faithfulness": 0.0, "completeness": 0.0, "overall": 0.0}
    for case in cases:
        score = judge.evaluate(case.get("query", ""), case.get("context", ""),
                               case.get("answer", ""))
        record = {
            "query": case.get("query", ""),
            "answer": case.get("answer", ""),
            "score": score.to_dict(),
        }
        scored.append(record)
        for key in ("correctness", "faithfulness", "completeness", "overall"):
            totals[key] += record["score"][key]

    aggregate: Dict[str, float] = {"case_count": float(len(scored))}
    if scored:
        for key, total in totals.items():
            aggregate[key] = round(total / len(scored), 4)
            if record_metrics:
                metrics_registry.set_gauge(
                    "agent_judge_score",
                    aggregate[key],
                    {"metric": key},
                )
    return JudgeBatchResult(cases=scored, aggregate=aggregate)
