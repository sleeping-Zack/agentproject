from __future__ import annotations

from dataclasses import dataclass, field
from typing import List


@dataclass
class AnswerClaim:
    """An atomic answer statement and the evidence references supporting it."""

    text: str
    evidence_ids: List[str] = field(default_factory=list)


@dataclass
class StructuredAnswer:
    """Machine-checkable representation of a generated answer."""

    summary: str
    claims: List[AnswerClaim] = field(default_factory=list)
    citations: List[str] = field(default_factory=list)


@dataclass
class ClaimSupport:
    """Deterministic grounding decision for one atomic claim."""

    claim: str
    supported: bool
    reason: str
    evidence_ids: List[str] = field(default_factory=list)
    confidence: float = 0.0
    contradiction: bool = False
