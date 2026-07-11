from __future__ import annotations

import re
import unicodedata
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

from agent.answer_schema import AnswerClaim, ClaimSupport, StructuredAnswer
from rag.judge import LLMJudge


_RAG_SCENES = {"rag", "rag_qa", "qa"}
_REPORT_SCENES = {"report", "monthly_report"}
_NUMERIC_CITATION_RE = re.compile(r"\[\s*(\d+)\s*\]")
_CITATION_HEADER_RE = re.compile(
    r"^\s*(?:引用(?:来源)?|参考(?:来源|资料)?|sources?)\s*[:：]\s*(.*)$",
    re.IGNORECASE,
)
_PLACEHOLDER_RE = re.compile(
    r"^(?:暂无|无|没有|无可用(?:来源|引用)?|none|n\s*/?\s*a|not\s+available)[。.!\s]*$",
    re.IGNORECASE,
)
_NEGATION_RE = re.compile(
    r"(?:严禁|禁止|切勿|不得|不能|不要|不应|避免|无需|不可|请勿|"
    r"must\s+not|do\s+not|don't|never|cannot|can't|should\s+not|shouldn't|"
    r"prohibit(?:ed)?|avoid)",
    re.IGNORECASE,
)
_HARMFUL_PATTERNS = (
    re.compile(
        r"(?:用水冲洗|水洗|浸泡|泡水).{0,12}(?:电机|主机|电池|充电座|电源|传感器)"
        r"|(?:电机|主机|电池|充电座|电源|传感器).{0,12}(?:用水冲洗|水洗|浸泡|泡水)"
    ),
    re.compile(r"(?:短接|刺穿|挤压|加热|焚烧|明火烧).{0,10}(?:电池|电源)"),
    re.compile(r"(?:绕过|关闭|禁用|拆除).{0,10}(?:安全|保护|防护|传感器|断电)"),
    re.compile(
        r"(?:rinse|wash|submerge|soak).{0,24}(?:motor|battery|dock|power\s*supply)"
        r"|(?:motor|battery|dock|power\s*supply).{0,24}(?:rinse|wash|submerge|soak)",
        re.IGNORECASE,
    ),
    re.compile(
        r"(?:short|puncture|burn|heat).{0,18}(?:battery|power\s*supply)"
        r"|(?:bypass|disable|remove).{0,18}(?:safety|protection|sensor)",
        re.IGNORECASE,
    ),
)
_BOILERPLATE_PHRASES = (
    "建议",
    "应该",
    "应当",
    "可以",
    "需要",
    "请注意",
    "请",
    "务必",
    "直接",
    "进行",
    "一次",
)
_ENGLISH_STOP_WORDS = {
    "a",
    "an",
    "and",
    "are",
    "be",
    "can",
    "could",
    "it",
    "must",
    "of",
    "should",
    "the",
    "to",
    "will",
}


@dataclass(frozen=True)
class _EvidenceRecord:
    evidence_id: str
    content: str


@dataclass
class VerifyResult:
    passed: bool
    action: str
    score: Optional[float]
    reasons: List[str] = field(default_factory=list)
    judge: Dict[str, Any] = field(default_factory=dict)
    citation_validity: float = 1.0
    citation_coverage: float = 1.0
    unsupported_claim_rate: float = 0.0
    harmful_instruction: bool = False
    claim_support: List[Dict[str, Any]] = field(default_factory=list)
    invalid_citations: List[str] = field(default_factory=list)

    @property
    def quality(self) -> Dict[str, Any]:
        """Stable quality payload for traces, API adapters, and tests."""

        return {
            "citation_validity": self.citation_validity,
            "citation_coverage": self.citation_coverage,
            "unsupported_claim_rate": self.unsupported_claim_rate,
            "harmful_instruction": self.harmful_instruction,
            "claim_support": self.claim_support,
            "invalid_citations": self.invalid_citations,
            "judge": self.judge,
        }


class AnswerVerifier:
    def __init__(
        self,
        judge: Optional[LLMJudge] = None,
        min_overall_score: float = 3.5,
        require_citation: bool = True,
        min_faithfulness_score: float = 4.0,
    ) -> None:
        self.judge = judge
        self.min_overall_score = min_overall_score
        self.require_citation = require_citation
        self.min_faithfulness_score = min_faithfulness_score

    def verify(
        self,
        query: str,
        answer: str,
        evidence: List[Dict[str, Any]],
        scene: str = "general",
        tool_results: Optional[List[Dict[str, Any]]] = None,
        artifacts: Optional[List[Dict[str, Any]]] = None,
        structured_answer: Optional[StructuredAnswer] = None,
    ) -> VerifyResult:
        """Run structural, deterministic grounding, then selective semantic checks."""

        reasons: List[str] = []
        tool_results = tool_results or []
        artifacts = artifacts or []
        records = self._normalise_evidence(evidence)
        evidence_by_id = {item.evidence_id: item for item in records}
        context = "\n".join(
            f"[{item.evidence_id}] {item.content}" for item in records if item.content.strip()
        )

        structured, schema_errors = self._coerce_structured_answer(structured_answer)
        for reason in schema_errors:
            self._add_reason(reasons, reason)
        effective_answer = self._effective_answer(answer, structured)
        verification_text = self._verification_text(effective_answer, structured)
        if not effective_answer.strip():
            self._add_reason(reasons, "answer_empty")
        if records and not context.strip():
            self._add_reason(reasons, "evidence_empty")
        if effective_answer.strip().startswith("请求未执行") and scene == "general":
            self._add_reason(reasons, "unexpected_refusal")

        if scene in _RAG_SCENES and not records:
            self._add_reason(reasons, "evidence_required")
        if scene in _REPORT_SCENES and not self._has_report_support(tool_results, artifacts):
            self._add_reason(reasons, "report_support_required")

        if structured is not None:
            citation_refs = list(structured.citations)
            placeholder_citation = any(_PLACEHOLDER_RE.fullmatch(ref.strip()) for ref in citation_refs)
        else:
            citation_refs, placeholder_citation = self._legacy_citation_refs(answer, records)

        resolved_citations, invalid_citations = self._resolve_references(citation_refs, records)
        citation_required = scene in _RAG_SCENES or (self.require_citation and bool(records))
        citation_validity = self._citation_validity(
            citation_refs,
            invalid_citations,
            citation_required=citation_required,
        )
        if placeholder_citation:
            self._add_reason(reasons, "citation_placeholder")
        if citation_required and not resolved_citations:
            self._add_reason(reasons, "citation_missing")
        if invalid_citations:
            self._add_reason(reasons, "citation_invalid")
        if citation_required and citation_validity < 1.0:
            self._add_reason(reasons, "citation_validity_below_threshold")

        claims = self._claims_for_verification(
            answer=answer,
            structured=structured,
            records=records,
            fallback_evidence_ids=resolved_citations,
        )
        grounding_required = bool(records) or scene in _RAG_SCENES
        support_results: List[ClaimSupport] = []
        covered_claims = 0
        unsupported_claims = 0
        declared_citation_ids = set(resolved_citations)
        claim_reference_errors: List[str] = []

        if grounding_required:
            if not claims and effective_answer.strip():
                self._add_reason(reasons, "claims_missing")
            for claim in claims:
                resolved_ids, invalid_ids = self._resolve_references(claim.evidence_ids, records)
                if resolved_ids:
                    covered_claims += 1
                if invalid_ids:
                    claim_reference_errors.extend(invalid_ids)
                    self._add_reason(reasons, "claim_evidence_id_invalid")
                if structured is not None and any(
                    evidence_id not in declared_citation_ids for evidence_id in resolved_ids
                ):
                    self._add_reason(reasons, "claim_citation_missing")

                support = self._evaluate_claim(claim.text, resolved_ids, evidence_by_id)
                support_results.append(support)
                if not support.supported:
                    unsupported_claims += 1
                if support.contradiction:
                    self._add_reason(reasons, "evidence_contradiction")

        invalid_citations = self._dedupe(invalid_citations + claim_reference_errors)
        if structured is not None and claim_reference_errors:
            all_structured_refs = list(citation_refs)
            all_structured_refs.extend(
                ref for claim in structured.claims for ref in claim.evidence_ids
            )
            _, all_invalid_refs = self._resolve_references(all_structured_refs, records)
            citation_validity = self._citation_validity(
                all_structured_refs,
                all_invalid_refs,
                citation_required=citation_required,
            )
            self._add_reason(reasons, "citation_invalid")
            if citation_required and citation_validity < 1.0:
                self._add_reason(reasons, "citation_validity_below_threshold")
        if claims and grounding_required:
            citation_coverage = round(covered_claims / len(claims), 4)
            unsupported_claim_rate = round(unsupported_claims / len(claims), 4)
        elif grounding_required:
            citation_coverage = 0.0
            unsupported_claim_rate = 1.0 if effective_answer.strip() else 0.0
        else:
            citation_coverage = 1.0
            unsupported_claim_rate = 0.0

        if grounding_required and citation_coverage < 0.9:
            self._add_reason(reasons, "citation_coverage_below_threshold")
        if grounding_required and unsupported_claim_rate > 0.05:
            self._add_reason(reasons, "unsupported_claim_rate_exceeded")

        harmful_instruction = self._contains_harmful_instruction(verification_text)
        if harmful_instruction:
            self._add_reason(reasons, "harmful_instruction")

        high_risk = bool(
            harmful_instruction
            or invalid_citations
            or unsupported_claims
            or any(item.contradiction for item in support_results)
        )
        low_confidence = structured is None or not support_results or any(
            item.confidence < 0.75 for item in support_results if item.supported
        )
        judge_payload: Dict[str, Any]
        overall_score: Optional[float] = None
        if self.judge is None:
            judge_payload = {
                "status": "not_evaluated",
                "reason": "judge_not_configured",
            }
        elif high_risk or low_confidence:
            judge_score = self.judge.evaluate(
                query=query,
                context=context,
                answer=effective_answer,
            )
            overall_score = judge_score.overall
            judge_payload = {"status": "evaluated", **judge_score.to_dict()}
            if judge_score.overall < self.min_overall_score:
                self._add_reason(reasons, "judge_score_below_threshold")
            if judge_score.faithfulness < self.min_faithfulness_score:
                self._add_reason(reasons, "judge_faithfulness_below_threshold")
        else:
            judge_payload = {
                "status": "not_evaluated",
                "reason": "deterministic_checks_high_confidence",
            }

        claim_support_payload = [item.__dict__.copy() for item in support_results]
        passed = not reasons
        action = self._action_for(reasons) if not passed else "accept"
        return VerifyResult(
            passed=passed,
            action=action,
            score=overall_score,
            reasons=reasons,
            judge=judge_payload,
            citation_validity=citation_validity,
            citation_coverage=citation_coverage,
            unsupported_claim_rate=unsupported_claim_rate,
            harmful_instruction=harmful_instruction,
            claim_support=claim_support_payload,
            invalid_citations=invalid_citations,
        )

    @staticmethod
    def _normalise_evidence(evidence: Sequence[Any]) -> List[_EvidenceRecord]:
        records: List[_EvidenceRecord] = []
        for item in evidence:
            if isinstance(item, Mapping):
                evidence_id = item.get("id") or item.get("evidence_id") or item.get("source")
                content = item.get("content", "")
            else:
                evidence_id = (
                    getattr(item, "id", None)
                    or getattr(item, "evidence_id", None)
                    or getattr(item, "source", None)
                )
                content = getattr(item, "content", "")
            evidence_id = str(evidence_id or "").strip()
            if evidence_id:
                records.append(_EvidenceRecord(evidence_id, str(content or "")))
        return records

    @staticmethod
    def _coerce_structured_answer(
        value: Optional[StructuredAnswer],
    ) -> Tuple[Optional[StructuredAnswer], List[str]]:
        if value is None:
            return None, []
        if isinstance(value, StructuredAnswer):
            if (
                not isinstance(value.summary, str)
                or not isinstance(value.claims, list)
                or not isinstance(value.citations, list)
            ):
                return None, ["structured_answer_invalid"]
            if not all(isinstance(item, AnswerClaim) for item in value.claims):
                return None, ["structured_answer_invalid"]
            if not all(isinstance(item, str) for item in value.citations):
                return None, ["structured_answer_invalid"]
            if any(
                not isinstance(claim.text, str)
                or not isinstance(claim.evidence_ids, list)
                for claim in value.claims
            ):
                return None, ["structured_answer_invalid"]
            if any(
                not isinstance(ref, str)
                for claim in value.claims
                for ref in claim.evidence_ids
            ):
                return None, ["structured_answer_invalid"]
            return value, []
        if not isinstance(value, Mapping):
            return None, ["structured_answer_invalid"]
        try:
            raw_claims = value.get("claims", [])
            raw_citations = value.get("citations", [])
            summary = value.get("summary", "")
            if (
                not isinstance(summary, str)
                or not isinstance(raw_claims, list)
                or not isinstance(raw_citations, list)
            ):
                raise TypeError
            claims = []
            for item in raw_claims:
                if not isinstance(item, Mapping):
                    raise TypeError
                text = item.get("text", "")
                evidence_ids = item.get("evidence_ids", [])
                if not isinstance(text, str) or not isinstance(evidence_ids, list):
                    raise TypeError
                if not all(isinstance(ref, str) for ref in evidence_ids):
                    raise TypeError
                claims.append(AnswerClaim(text=text, evidence_ids=list(evidence_ids)))
            if not all(isinstance(ref, str) for ref in raw_citations):
                raise TypeError
        except (AttributeError, TypeError):
            return None, ["structured_answer_invalid"]
        return StructuredAnswer(summary, claims, list(raw_citations)), []

    @staticmethod
    def _effective_answer(answer: str, structured: Optional[StructuredAnswer]) -> str:
        if answer.strip() or structured is None:
            return answer
        parts = [structured.summary]
        parts.extend(claim.text for claim in structured.claims)
        return "\n".join(part for part in parts if part.strip())

    @staticmethod
    def _verification_text(answer: str, structured: Optional[StructuredAnswer]) -> str:
        if structured is None:
            return answer
        parts = [answer, structured.summary]
        parts.extend(claim.text for claim in structured.claims)
        return "\n".join(dict.fromkeys(part for part in parts if part.strip()))

    @staticmethod
    def _has_report_support(
        tool_results: Sequence[Mapping[str, Any]],
        artifacts: Sequence[Mapping[str, Any]],
    ) -> bool:
        if tool_results:
            return True
        return any(
            artifact.get("type") in {"usage_record", "report", "tool_results"}
            or artifact.get("artifact_type") in {"usage_record", "report", "tool_results"}
            for artifact in artifacts
        )

    @classmethod
    def _legacy_citation_refs(
        cls,
        answer: str,
        records: Sequence[_EvidenceRecord],
    ) -> Tuple[List[str], bool]:
        refs = [match.group(0) for match in _NUMERIC_CITATION_RE.finditer(answer)]
        for record in records:
            refs.extend(record.evidence_id for _ in cls._id_matches(answer, record.evidence_id))

        placeholder = False
        valid_ids = {record.evidence_id for record in records}
        for line in answer.splitlines():
            header = _CITATION_HEADER_RE.match(line)
            if not header:
                continue
            payload = header.group(1).strip()
            if _PLACEHOLDER_RE.fullmatch(payload):
                placeholder = True
                continue
            for token in re.split(r"[,，、;；\s]+", payload):
                token = token.strip(".。()（）")
                if not token or _NUMERIC_CITATION_RE.fullmatch(token):
                    continue
                if token in valid_ids or any(item in token for item in valid_ids):
                    continue
                refs.append(token)
        return refs, placeholder

    @staticmethod
    def _id_matches(text: str, evidence_id: str) -> List[re.Match[str]]:
        if not evidence_id:
            return []
        pattern = re.compile(rf"(?<![\w]){re.escape(evidence_id)}(?![\w])")
        return list(pattern.finditer(text))

    @staticmethod
    def _resolve_references(
        refs: Sequence[str],
        records: Sequence[_EvidenceRecord],
    ) -> Tuple[List[str], List[str]]:
        valid_ids = {record.evidence_id for record in records}
        resolved: List[str] = []
        invalid: List[str] = []
        for raw_ref in refs:
            ref = str(raw_ref).strip()
            numeric = _NUMERIC_CITATION_RE.fullmatch(ref)
            if numeric:
                index = int(numeric.group(1))
                if 1 <= index <= len(records):
                    resolved.append(records[index - 1].evidence_id)
                else:
                    invalid.append(ref)
            elif ref in valid_ids:
                resolved.append(ref)
            else:
                invalid.append(ref)
        return AnswerVerifier._dedupe(resolved), AnswerVerifier._dedupe(invalid)

    @staticmethod
    def _citation_validity(
        refs: Sequence[str],
        invalid_refs: Sequence[str],
        *,
        citation_required: bool,
    ) -> float:
        if not refs:
            return 0.0 if citation_required else 1.0
        invalid_values = set(invalid_refs)
        invalid_count = sum(1 for ref in refs if str(ref).strip() in invalid_values)
        return round((len(refs) - invalid_count) / len(refs), 4)

    @classmethod
    def _claims_for_verification(
        cls,
        *,
        answer: str,
        structured: Optional[StructuredAnswer],
        records: Sequence[_EvidenceRecord],
        fallback_evidence_ids: Sequence[str],
    ) -> List[AnswerClaim]:
        if structured is not None:
            return [
                AnswerClaim(claim.text.strip(), list(claim.evidence_ids))
                for claim in structured.claims
                if claim.text.strip()
            ]

        content_lines: List[str] = []
        for line in answer.splitlines():
            if _CITATION_HEADER_RE.match(line):
                break
            content_lines.append(line)
        content = "\n".join(content_lines)
        pieces = re.split(r"(?<=[。！？!?；;])|(?<=[.])\s+|\n+", content)
        claims: List[AnswerClaim] = []
        for piece in pieces:
            raw = piece.strip()
            if not raw:
                continue
            refs = [match.group(0) for match in _NUMERIC_CITATION_RE.finditer(raw)]
            for record in records:
                refs.extend(record.evidence_id for _ in cls._id_matches(raw, record.evidence_id))
            claim_text = _NUMERIC_CITATION_RE.sub("", raw)
            for record in records:
                claim_text = claim_text.replace(record.evidence_id, "")
            claim_text = re.sub(r"^\s*(?:[-*#>]|\d+[.)、])\s*", "", claim_text).strip()
            claim_text = claim_text.strip("。.!！?？;；:：")
            if not re.search(r"[A-Za-z0-9\u4e00-\u9fff]", claim_text):
                continue
            claims.append(
                AnswerClaim(
                    text=claim_text,
                    evidence_ids=refs or list(fallback_evidence_ids),
                )
            )
        return claims

    @classmethod
    def _evaluate_claim(
        cls,
        claim: str,
        evidence_ids: Sequence[str],
        evidence_by_id: Mapping[str, _EvidenceRecord],
    ) -> ClaimSupport:
        if not evidence_ids:
            return ClaimSupport(
                claim=claim,
                supported=False,
                reason="claim_has_no_evidence",
            )
        candidate_sentences: List[str] = []
        for evidence_id in evidence_ids:
            record = evidence_by_id.get(evidence_id)
            if record is not None:
                candidate_sentences.extend(cls._split_sentences(record.content))
        if not candidate_sentences:
            return ClaimSupport(
                claim=claim,
                supported=False,
                reason="evidence_content_empty",
                evidence_ids=list(evidence_ids),
            )

        best_score = 0.0
        best_sentence = ""
        for sentence in candidate_sentences:
            score = cls._lexical_support_score(claim, sentence)
            if score > best_score:
                best_score = score
                best_sentence = sentence
        contradiction = best_score >= 0.45 and cls._polarity_conflicts(claim, best_sentence)
        if contradiction:
            return ClaimSupport(
                claim=claim,
                supported=False,
                reason="evidence_contradicts_claim",
                evidence_ids=list(evidence_ids),
                confidence=round(best_score, 4),
                contradiction=True,
            )
        if best_score >= 0.58:
            return ClaimSupport(
                claim=claim,
                supported=True,
                reason="supported_by_evidence",
                evidence_ids=list(evidence_ids),
                confidence=round(best_score, 4),
            )
        return ClaimSupport(
            claim=claim,
            supported=False,
            reason="evidence_does_not_support_claim",
            evidence_ids=list(evidence_ids),
            confidence=round(best_score, 4),
        )

    @staticmethod
    def _split_sentences(text: str) -> List[str]:
        return [
            sentence.strip()
            for sentence in re.split(r"(?<=[。！？!?；;.])\s*|\n+", text)
            if sentence.strip()
        ]

    @classmethod
    def _lexical_support_score(cls, claim: str, evidence: str) -> float:
        claim_topic = cls._topic_text(claim)
        evidence_topic = cls._topic_text(evidence)
        if not claim_topic or not evidence_topic:
            return 0.0
        if claim_topic in evidence_topic:
            return 1.0

        claim_words = cls._english_words(claim_topic)
        evidence_words = set(cls._english_words(evidence_topic))
        word_score = (
            sum(word in evidence_words for word in claim_words) / len(claim_words)
            if claim_words
            else 0.0
        )

        claim_chinese = "".join(re.findall(r"[\u4e00-\u9fff]", claim_topic))
        evidence_chinese = "".join(re.findall(r"[\u4e00-\u9fff]", evidence_topic))
        chinese_score = 0.0
        if claim_chinese:
            claim_chars = set(claim_chinese)
            evidence_chars = set(evidence_chinese)
            char_score = len(claim_chars & evidence_chars) / len(claim_chars)
            claim_bigrams = cls._ngrams(claim_chinese, 2)
            evidence_bigrams = set(cls._ngrams(evidence_chinese, 2))
            bigram_score = (
                sum(item in evidence_bigrams for item in claim_bigrams) / len(claim_bigrams)
                if claim_bigrams
                else char_score
            )
            chinese_score = 0.6 * char_score + 0.4 * bigram_score
        return round(max(word_score, chinese_score), 4)

    @staticmethod
    def _topic_text(text: str) -> str:
        result = unicodedata.normalize("NFKC", text).lower()
        result = _NUMERIC_CITATION_RE.sub("", result)
        for phrase in _BOILERPLATE_PHRASES:
            result = result.replace(phrase, "")
        result = _NEGATION_RE.sub("", result)
        return re.sub(r"[^a-z0-9\u4e00-\u9fff]+", " ", result).strip()

    @staticmethod
    def _english_words(text: str) -> List[str]:
        words = []
        for word in re.findall(r"[a-z0-9]+", text.lower()):
            if word in _ENGLISH_STOP_WORDS:
                continue
            if len(word) > 4 and word.endswith("ing"):
                word = word[:-3]
            elif len(word) > 3 and word.endswith("ed"):
                word = word[:-2]
            elif len(word) > 4 and word.endswith("ly"):
                word = word[:-2]
            elif len(word) > 3 and word.endswith("s"):
                word = word[:-1]
            words.append(word)
        return words

    @staticmethod
    def _ngrams(text: str, size: int) -> List[str]:
        if len(text) < size:
            return [text] if text else []
        return [text[index : index + size] for index in range(len(text) - size + 1)]

    @staticmethod
    def _polarity_conflicts(claim: str, evidence: str) -> bool:
        return bool(_NEGATION_RE.search(claim)) != bool(_NEGATION_RE.search(evidence))

    @staticmethod
    def _contains_harmful_instruction(answer: str) -> bool:
        for sentence in AnswerVerifier._split_sentences(answer):
            for clause in re.split(r"[，,]", sentence):
                if _NEGATION_RE.search(clause):
                    continue
                if any(pattern.search(clause) for pattern in _HARMFUL_PATTERNS):
                    return True
        return False

    @staticmethod
    def _action_for(reasons: Sequence[str]) -> str:
        hard_refusal_reasons = {
            "answer_empty",
            "citation_invalid",
            "citation_placeholder",
            "claim_evidence_id_invalid",
            "evidence_contradiction",
            "harmful_instruction",
        }
        if hard_refusal_reasons.intersection(reasons):
            return "refuse"
        retry_reasons = {
            "citation_coverage_below_threshold",
            "judge_faithfulness_below_threshold",
            "judge_score_below_threshold",
            "unsupported_claim_rate_exceeded",
        }
        return "retry" if retry_reasons.intersection(reasons) else "refuse"

    @staticmethod
    def _add_reason(reasons: List[str], reason: str) -> None:
        if reason not in reasons:
            reasons.append(reason)

    @staticmethod
    def _dedupe(values: Sequence[str]) -> List[str]:
        return list(dict.fromkeys(values))
